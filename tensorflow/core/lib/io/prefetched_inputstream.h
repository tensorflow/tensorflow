#ifndef TENSORFLOW_CORE_LIB_IO_PREFETCHED_INPUTSTREAM_H_
#define TENSORFLOW_CORE_LIB_IO_PREFETCHED_INPUTSTREAM_H_

#include <deque>
#include <functional>
#include <memory>
#include <thread>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace io {

// Do NOT depend on internal implementation details!
namespace internal {

typedef std::unique_ptr<char[]> BytesHandle;
typedef std::unique_ptr<RandomAccessFile> RAFHandle;

/// \brief Information for a single parallel prefetch task.
struct TaskInfo {
  const size_t id_;         // Continuous increment id
  const size_t start_;      // Start offset in file
  const size_t length_;     // Length to read, usually a fixed size unless
                            // is_file_end_ is true.
  const bool is_file_end_;  // If this task reaches the end of file, usually the
                            // last task.

  size_t EndPos() const { return start_ + length_; }
};

/// \brief Generate the next TaskInfo.
///
/// \param file_size The total length of file.
/// \param task_size Reading length of each task.
/// \param task_id The current task_id, it will be increased by 1 after
///                invocation.
/// \param task_start The current start offset in file, it will be increased by
///                   task_size if EOF not reached, otherwise increase to
///                   file_size.
TaskInfo NextTask(const size_t file_size, const size_t task_size,
                  size_t* task_id, size_t* task_start);

/// \brief A simple pool for object reuse.
///
/// NOT thread safe!!
template <typename T>
class Pool {
 private:
  typedef std::unique_ptr<T> HandleType;
  typedef std::function<Status(HandleType*)> FactoryType;

 public:
  Pool(FactoryType factory)
      : factory_(std::move(factory)), num_created_(0), num_borrowed_(0) {}

  void Return(HandleType&& handle) {
    CHECK(num_borrowed_ > 0);
    num_borrowed_ -= 1;
    pool_.push_back(std::move(handle));
  }

  Status Borrow(HandleType* result) {
    num_borrowed_ += 1;
    if (pool_.empty()) {
      num_created_ += 1;
      return factory_(result);
    } else {
      *result = std::move(pool_.front());
      pool_.pop_front();
      return Status::OK();
    }
  }

  size_t NumCreated() { return num_created_; }
  size_t NumBorrowed() { return num_borrowed_; }

 private:
  FactoryType factory_;
  std::deque<HandleType> pool_;
  size_t num_created_;
  size_t num_borrowed_;
};

/// \brief PrefetchTask is the minimum unit for parallel reading.
///
/// PrefetchTask is the key data structure we're working on. The life cycle of a
/// PrefetchTask looks like:
///   1. Create with a bytes buffer to fill data into, a file handle to read
///   data from and task info describing range in file.
///   2. After creation, it'll be pushed into a queue for later reading. And a
///   thread will also be scheduled to drive the this task to read file and
///   fill buffer.
///   3. Meanwhile, the task may be canceled, and the atomic `is_cancled_`
///   variable will be set to *true*.
///   4. After the true-value `is_cancled_` variable been observed by reading
///   thread or the filling task finished (both successfully or abnormally), the
///   `fill_done_` notification will be fired.
///   5. The consuming thread won't start to consuming a task's buffer
///   until it is notified.
///   6. When the data is exhausted, the task is removed from queue and
///   it's buffer and file object are returned to corresponding pool. This task
///   task dies.
class PrefetchTask {
 public:
  PrefetchTask(BytesHandle bytes, RAFHandle raf, TaskInfo task_info);

  /// \brief Fill the internal bytes buffer with data read from file.
  ///        Write cursor will increase as a side effect.
  /// \param n Number of bytes requested to fill.
  /// \return Error status if something wrong when reading file.
  Status Fill(size_t n);

  /// \brief Try to read n bytes into target string.
  ///        Read cursor will increase as a side effect.
  /// \param n bytes try to read.
  /// \param target target string. It will be appended but NOT cleared!
  /// \return bytes read indeed.
  size_t Read(size_t n, tstring* target);

  /// \brief Try to read a line into target string.
  ///        Read cursor will increase as a side effect.
  /// \param include_eol if the trailing LF should be included in target string.
  /// \param target target string. It will NOT cleared before writing!
  /// \return true if LF reached. Otherwise the buffer is exhausted without a
  ///         single LF encountered.
  bool ReadLine(bool include_eol, string* target);

  /// \brief Check if all expected bytes have been filled into buffer as task
  /// info describled.
  bool IsFullFilled() { return w_cursor_ == info_.length_; }

  /// \brief Check if all bytes in the buffer have been read.
  bool IsReadExhausted() { return r_cursor_ == info_.length_; }

  /// \brief Check if this task is readable.
  ///
  /// Note:
  ///  1. Since this function is on the critical reading path, we shot-circuit
  ///  it by check r_cursor_: if we're already reding, it must be readable.
  ///  2. fill_done_ has been notified!
  bool IsReadable() { return r_cursor_ > 0 || fill_done_.HasBeenNotified(); }

  /// \brief Check if filling has been canceled.
  bool IsCancled() { return is_cancled.load(); }

  /// \brief Wait for this task to finish.
  void WaitForFillingDone() { fill_done_.WaitForNotification(); }

  /// \brief Send the notification that filling is done.
  void NotifyFillDone() { fill_done_.Notify(); }

  /// \brief Set is_cancled to true.
  void Cancel() { is_cancled.store(true); }

  /// \brief Reset read cursor to pos.
  void ResetReadCursor(size_t pos) {
    CHECK(pos < info_.length_)
        << "Invalid pos: " << pos << ", task length: " << info_.length_;
    r_cursor_ = pos;
  }

  /// \brief Get the current reading offset in file.
  size_t Tell() const { return info_.start_ + r_cursor_; }

  const TaskInfo& GetTaskInfo() const { return info_; }
  const Status& GetIOStatus() const { return io_status_; }

  bool IsFileEnd() const { return info_.is_file_end_; }

  /// \brief Take ownership of buffer bytes.
  ///        This task cant be used for filling or reading any more!!
  BytesHandle MoveOutBytes() { return std::move(bytes_); }

  /// \brief Take ownership of underlaying file.
  ///        This task cant be used for filling or reading any more!!
  RAFHandle MoveOutRAF() { return std::move(raf_); }

 private:
  BytesHandle bytes_;  // The byte array to buffer data, borrow from bytes pool.
  RAFHandle raf_;      // The raf to read data, borrow from file pool.
  const TaskInfo info_;          // Description info of this task.
  std::atomic<bool> is_cancled;  // If the task has been cancled.
  Notification fill_done_;       // Notification of filling done.
  size_t r_cursor_;              // Next reading position.
  size_t w_cursor_;              // Next filling position.
  Status io_status_;  // Not ok if any error occurs when running this task.

  TF_DISALLOW_COPY_AND_ASSIGN(PrefetchTask);
};
}  // namespace internal

/// \brief An inputstream which prefetch file content in parallel and
/// background.
///
/// Please note that this class is NOT thread safe!! All fields are operated
/// with single-thread assumption. Synchronization happens on state_ of the
/// head task in queue.
class PrefetchedInputStream : public InputStreamInterface {
 public:
  static Status New(const string& fname, size_t max_threads,
                    size_t total_buf_size,
                    std::unique_ptr<PrefetchedInputStream>* result);

  ~PrefetchedInputStream() override;

  /// \brief Read some bytes into target string.
  /// \param bytes_to_read number of bytes to read.
  /// \param result the target string. It will always be clear before filling.
  /// \return OK if success and error otherwise.
  Status ReadNBytes(int64 bytes_to_read, tstring* result) override;

  /// \brief Skip some bytes forward.
  /// \param bytes_to_skip number of bytes to skip.
  /// \return OK if success and error otherwise.
  Status SkipNBytes(int64 bytes_to_skip) override;

  /// \brief Get current reading offset in file.
  int64 Tell() const override { return pos_; }

  /// \brief Seek to an random offset in file.
  /// \param position offset to seek.
  /// \return OK if success and error otherwise.
  Status Seek(int64 position);

  /// \brief Read a whole line into result until EOF or LF is read.
  /// \param result the target string. It will always be overwritten.
  /// \return OK if success or OUT_OF_RANGE if at the end of the file or other
  /// errors.
  Status ReadLine(string* result);

  /// \brief Return a line of data until EOF or LF is read. The LF is included.
  string ReadLineAsString();

  /// \brief Read all content into result string.
  ///
  /// Note: This may require large amount of memory, don't use this method
  ///       unless you know what you're doing.
  Status ReadAll(string* result);

  /// \brief Reset stream to initial status.
  /// \return OK if success and error otherwise.
  Status Reset() override;

 private:
  /// \brief Try to schedule more background prefetch tasks.
  Status SchedulePrefetch();

  typedef std::function<bool(internal::PrefetchTask&)> ReadFuncType;
  Status DoRead(ReadFuncType read_func);

  /// \brief Reset this stream to initial status with specified offset.
  Status ResetTo(size_t offset);

  /// \brief Destroy the head task in queue. Return true if there is a
  /// front in queue and it is destroyed.
  bool DestroyQueueFront();

 private:
  PrefetchedInputStream(const string& fname, size_t max_threads,
                        size_t max_tasks, size_t buf_size, size_t file_size,
                        size_t task_size);

 private:
  const string fname_;        // file to read.
  const size_t max_threads_;  // max number of threads.
  const size_t max_tasks_;    // max_threads_ * 2.
  const size_t buf_size_;     // buffer size of all threads in total.
  const size_t file_size_;    // size of the whole file to read.
  const size_t task_size_;    // size of each generated task.

  size_t pos_;         // current pos from begining of the file.
  size_t next_id_;     // task id generator.
  size_t next_start_;  // offset in file of next task.

  // std::unique_ptr<internal::TaskGenerator>
  //     task_gen_;  // task sequence generator.

  std::unique_ptr<internal::Pool<char[]>>
      bytes_pool_;  // byte arrays for reuse.

  std::unique_ptr<internal::Pool<RandomAccessFile>>
      file_pool_;  // randome access files for reuse.

  std::unique_ptr<std::deque<internal::PrefetchTask>> task_queue_;  // the queue

  std::unique_ptr<thread::ThreadPool>
      thread_pool_;  // thread pool for parallel reading.

  Status status_;  // last error io status

  TF_DISALLOW_COPY_AND_ASSIGN(PrefetchedInputStream);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_PREFETCHED_INPUTSTREAM_H_