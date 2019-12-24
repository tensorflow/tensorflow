#include "tensorflow/core/lib/io/prefetched_inputstream.h"

#include <atomic>
#include <deque>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

namespace internal {

TaskInfo NextTask(const size_t file_size, const size_t task_size,
                  size_t* task_id, size_t* task_start) {
  size_t start = *task_start;
  size_t length = std::min<size_t>(task_size, file_size - *task_start);
  *task_start += length;
  bool is_last = *task_start == file_size;
  return {(*task_id)++, start, length, is_last};
}

PrefetchTask::PrefetchTask(BytesHandle bytes, RAFHandle raf, TaskInfo task_info)
    : bytes_(std::move(bytes)),
      raf_(std::move(raf)),
      info_(task_info),
      is_cancled(false),
      fill_done_(),
      r_cursor_(0),
      w_cursor_(0) {}

Status PrefetchTask::Fill(size_t n) {
  CHECK(w_cursor_ < info_.length_) << "Read after full";
  uint64 offset = static_cast<uint64>(info_.start_ + w_cursor_);
  char* scratch = &bytes_[w_cursor_];
  n = std::min<size_t>(n, info_.length_ - w_cursor_);  // EOF-free guarantee
  StringPiece result;
  Status s = raf_->Read(offset, n, &result, scratch);
  if (!s.ok()) {
    io_status_ = s;
    return s;
  }
  if (result.length() > 0 && result.data() != scratch) {
    memmove(scratch, result.data(), result.size());
  }
  w_cursor_ += n;
  return Status::OK();
}

size_t PrefetchTask::Read(size_t n, tstring* target) {
  size_t bytes_to_copy = std::min<size_t>(n, info_.length_ - r_cursor_);
  target->append(&bytes_[r_cursor_], bytes_to_copy);
  r_cursor_ += bytes_to_copy;
  return bytes_to_copy;
}

bool PrefetchTask::ReadLine(bool include_eol, string* target) {
  while (r_cursor_ < info_.length_) {
    char c = bytes_[r_cursor_++];
    if (c == '\n') {
      if (include_eol) {
        *target += c;
      }
      return true;
    }
    if (c != '\r') {  // CR is ignored.
      *target += c;
    }
  }
  return false;
}

}  // namespace internal

namespace {

constexpr size_t kMinTaskSize = 64 * 1024;  // 64 KB

/// Drive the task task to fill the buffer while check for cancelation.
void RunTask(internal::PrefetchTask* task) {
  // while filling is not finished and task not cancelled.
  while (!task->IsFullFilled() && !task->IsCancled()) {
    Status s = task->Fill(128 * 1024);
    if (!s.ok()) break;
  }
  task->NotifyFillDone();
}

}  // namespace

Status PrefetchedInputStream::New(
    const string& fname, size_t max_threads, size_t buf_size,
    std::unique_ptr<PrefetchedInputStream>* result) {
  size_t max_tasks = max_threads * 2;
  size_t task_size = std::max(buf_size / max_tasks, kMinTaskSize);
  uint64 file_size;
  TF_RETURN_IF_ERROR(Env::Default()->GetFileSize(fname, &file_size));
  result->reset(new PrefetchedInputStream(fname, max_threads, max_tasks,
                                          buf_size, file_size, task_size));
  return Status::OK();
}

PrefetchedInputStream::PrefetchedInputStream(const string& fname,
                                             size_t max_threads,
                                             size_t max_tasks, size_t buf_size,
                                             size_t file_size, size_t task_size)
    : fname_(fname),
      max_threads_(max_threads),
      max_tasks_(max_tasks),
      buf_size_(buf_size),
      file_size_(file_size),
      task_size_(task_size),
      pos_(0),
      next_id_(0),
      next_start_(0),
      status_() {
  auto bytes_factory = [task_size](internal::BytesHandle* handle) {
    handle->reset(new char[task_size]);
    return Status::OK();
  };
  bytes_pool_ = absl::make_unique<internal::Pool<char[]>>(bytes_factory);

  auto file_factory = [this](internal::RAFHandle* handle) {
    return Env::Default()->NewRandomAccessFile(this->fname_, handle);
  };
  file_pool_ = absl::make_unique<internal::Pool<RandomAccessFile>>(
      std::move(file_factory));

  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), "prefetch", static_cast<int>(max_threads_));
  task_queue_ = absl::make_unique<std::deque<internal::PrefetchTask>>();
}

PrefetchedInputStream::~PrefetchedInputStream() {
  for (auto& task : *task_queue_) {
    task.Cancel();
  }

  for (auto& task : *task_queue_) {
    task.WaitForFillingDone();
  }
}

Status PrefetchedInputStream::ReadNBytes(int64 bytes_to_read, tstring* result) {
  result->clear();
  result->reserve(bytes_to_read);
  auto read_fn = [result, bytes_to_read](internal::PrefetchTask& task) mutable {
    size_t n = task.Read(bytes_to_read, result);
    bytes_to_read -= n;
    return bytes_to_read <= 0;
  };
  return DoRead(std::move(read_fn));
}

Status PrefetchedInputStream::DoRead(ReadFuncType read_func) {
  // result->clear();
  if (!status_.ok()) return status_;

  // result->reserve(bytes_to_read);
  bool is_finish = false;
  while (!is_finish) {
    if (task_queue_->empty() || !task_queue_->front().IsReadable()) {
      TF_RETURN_IF_ERROR(SchedulePrefetch());
      std::this_thread::yield();
      continue;
    }

    auto& task = task_queue_->front();
    if (!task.GetIOStatus().ok()) return task.GetIOStatus();

    is_finish = read_func(task);
    pos_ = task.Tell();

    if (task.IsReadExhausted()) {
      bool eof = task.IsFileEnd();
      DestroyQueueFront();
      if (eof) {
        status_ = errors::OutOfRange("");
        return is_finish ? Status::OK() : status_;
      }
      TF_RETURN_IF_ERROR(SchedulePrefetch());
    }
  }
  return Status::OK();
}

Status PrefetchedInputStream::SchedulePrefetch() {
  if (task_queue_->size() >= max_tasks_) return Status::OK();

  size_t n_schedule = max_tasks_ - task_queue_->size();
  for (size_t i = 0; i < n_schedule; i++) {
    auto info =
        internal::NextTask(file_size_, task_size_, &next_id_, &next_start_);
    if (info.length_ == 0) {
      return Status::OK();
    }
    internal::RAFHandle raf;
    TF_RETURN_IF_ERROR(file_pool_->Borrow(&raf));
    internal::BytesHandle bytes;
    TF_RETURN_IF_ERROR(bytes_pool_->Borrow(&bytes));
    task_queue_->emplace_back(std::move(bytes), std::move(raf), info);
    // Safe to observe task task in deque with raw pointer as far as remains in
    // the dequeue. See:
    //     https://en.cppreference.com/w/cpp/container/deque#Invalidation_notes
    internal::PrefetchTask* task = &(task_queue_->back());
    thread_pool_->Schedule([task]() { RunTask(task); });
  }
  return Status::OK();
}

Status PrefetchedInputStream::ReadLine(string* result) {
  result->clear();
  auto read_fn = [result](internal::PrefetchTask& task) mutable {
    return task.ReadLine(/* include_eol */ false, result);
  };
  return DoRead(std::move(read_fn));
}

string PrefetchedInputStream::ReadLineAsString() {
  string result;
  auto read_fn = [&result](internal::PrefetchTask& task) {
    return task.ReadLine(/* include_eol */ true, &result);
  };
  DoRead(std::move(read_fn)).IgnoreError();
  return result;
}

Status PrefetchedInputStream::ReadAll(string* result) {
  result->clear();
  auto read_fn = [result](internal::PrefetchTask& task) mutable {
    task.ReadLine(/* include_eol */ true, result);
    return false;
  };
  Status s = DoRead(std::move(read_fn));
  if (errors::IsOutOfRange(s)) {
    return Status::OK();
  }
  return s;
}

Status PrefetchedInputStream::SkipNBytes(int64 bytes_to_skip) {
  return Seek(static_cast<int64>(pos_) + bytes_to_skip);
}

Status PrefetchedInputStream::Seek(int64 position) {
  if (position < 0) {
    return errors::InvalidArgument("Seeking to a negative position: ",
                                   position);
  }

  size_t new_pos = static_cast<size_t>(position);
  if (!task_queue_->empty() &&
      task_queue_->front().GetTaskInfo().start_ <= new_pos &&
      new_pos < task_queue_->back().GetTaskInfo().EndPos()) {
    // if position falls into current task queue.
    while (task_queue_->front().GetTaskInfo().EndPos() <= new_pos) {
      auto& front = task_queue_->front();
      front.Cancel();
      front.WaitForFillingDone();
      task_queue_->pop_front();
    }
    auto& front_info = task_queue_->front().GetTaskInfo();
    task_queue_->front().ResetReadCursor(new_pos - front_info.start_);
    pos_ = new_pos;
    return Status::OK();
  }

  return ResetTo(new_pos);
}

bool PrefetchedInputStream::DestroyQueueFront() {
  if (!task_queue_->empty()) {
    auto& task = task_queue_->front();
    auto bytes = task.MoveOutBytes();
    auto raf = task.MoveOutRAF();
    bytes_pool_->Return(std::move(bytes));
    file_pool_->Return(std::move(raf));
    task_queue_->pop_front();
    return true;
  }
  return false;
}

Status PrefetchedInputStream::Reset() { return ResetTo(0); }

Status PrefetchedInputStream::ResetTo(size_t offset) {
  for (auto& task : *task_queue_) {
    task.Cancel();
  }

  for (auto& task : *task_queue_) {
    task.WaitForFillingDone();
  }

  while (DestroyQueueFront()) {
  };

  if (offset > file_size_) {
    status_ =
        errors::OutOfRange("Invalid pos set to TaskGenerator, pos: ", offset,
                           ", file size: ", file_size_);
    return status_;
  }

  pos_ = offset;
  next_start_ = offset;

  status_ = Status::OK();
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow