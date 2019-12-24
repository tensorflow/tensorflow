#include "tensorflow/core/lib/io/prefetched_inputstream.h"

#include <chrono>
#include <string>
#include <thread>
#include <utility>

#include "absl/memory/memory.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace io {

TEST(PrefetchedInputStream, NextTask1) {
  const size_t file_size = 100;
  const size_t task_size = 10;
  size_t task_id = 0;
  size_t task_start = 0;

  for (size_t i = 0; i < 10; i++) {
    auto task = internal::NextTask(file_size, task_size, &task_id, &task_start);
    EXPECT_EQ(task.id_, i);
    EXPECT_EQ(task.start_, i * 10);
    EXPECT_EQ(task.length_, 10);
    EXPECT_EQ(task.EndPos(), i * 10 + 10);
  }
  for (size_t i = 0; i < 5; i++) {
    auto task = internal::NextTask(file_size, task_size, &task_id, &task_start);
    EXPECT_EQ(task.id_, 10 + i);
    EXPECT_EQ(task.start_, 100);
    EXPECT_EQ(task.length_, 0);
    EXPECT_EQ(task.EndPos(), 100);
  }
}

TEST(PrefetchedInputStream, NextTask2) {
  const size_t file_size = 10;
  const size_t task_size = 4;
  size_t task_id = 0;
  size_t task_start = 0;

  auto task0 = internal::NextTask(file_size, task_size, &task_id, &task_start);
  EXPECT_EQ(task0.id_, 0);
  EXPECT_EQ(task0.start_, 0);
  EXPECT_EQ(task0.length_, 4);
  EXPECT_EQ(task0.EndPos(), 4);

  auto task1 = internal::NextTask(file_size, task_size, &task_id, &task_start);
  EXPECT_EQ(task1.id_, 1);
  EXPECT_EQ(task1.start_, 4);
  EXPECT_EQ(task1.length_, 4);
  EXPECT_EQ(task1.EndPos(), 8);

  auto task2 = internal::NextTask(file_size, task_size, &task_id, &task_start);
  EXPECT_EQ(task2.id_, 2);
  EXPECT_EQ(task2.start_, 8);
  EXPECT_EQ(task2.length_, 2);
  EXPECT_EQ(task2.EndPos(), 10);
}

TEST(Pool, BorrowAndReturn) {
  auto factory = [](std::unique_ptr<string>* result) {
    static int counter = 1;
    result->reset(new string(std::to_string(counter++)));
    return Status::OK();
  };
  internal::Pool<string> pool(std::move(factory));
  std::unique_ptr<string> result1;
  TF_EXPECT_OK(pool.Borrow(&result1));
  EXPECT_EQ("1", *result1);
  EXPECT_EQ(1, pool.NumCreated());
  EXPECT_EQ(1, pool.NumBorrowed());

  std::unique_ptr<string> result2;
  TF_EXPECT_OK(pool.Borrow(&result2));
  EXPECT_EQ("2", *result2);
  EXPECT_EQ(2, pool.NumCreated());
  EXPECT_EQ(2, pool.NumBorrowed());

  pool.Return(std::move(result1));
  EXPECT_EQ(2, pool.NumCreated());
  EXPECT_EQ(1, pool.NumBorrowed());

  pool.Return(std::move(result2));
  EXPECT_EQ(2, pool.NumCreated());
  EXPECT_EQ(0, pool.NumBorrowed());

  std::unique_ptr<string> result3;
  TF_EXPECT_OK(pool.Borrow(&result3));
  EXPECT_EQ("1", *result3);
  EXPECT_EQ(2, pool.NumCreated());
  EXPECT_EQ(1, pool.NumBorrowed());

  pool.Return(std::move(result3));
  EXPECT_EQ(2, pool.NumCreated());
  EXPECT_EQ(0, pool.NumBorrowed());
}

TEST(Pool, InvalidReturn) {
  auto factory = [](std::unique_ptr<string>* result) {
    static int counter = 1;
    result->reset(new string(std::to_string(counter++)));
    return Status::OK();
  };
  internal::Pool<string> pool(std::move(factory));
  std::unique_ptr<string> result;
  ASSERT_DEATH(pool.Return(std::move(result)),
               "Check failed: num_borrowed_ > 0");
}

namespace {
constexpr size_t _1_MB = 1024 * 1024;
const string kFileUnit("0123456789");
const string kFileUnitX2("01234567890123456789");
const string kFileUnitEOL("0123456789\n");

Status PrepareDataFile(Env* env, size_t file_size, bool eol, string* fname) {
  *fname = testing::TmpDir() + "/prefetched_inputstream_" +
           std::to_string(random::New64());
  std::unique_ptr<WritableFile> w_file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(*fname, &w_file));
  const string& unit = eol ? kFileUnitEOL : kFileUnit;
  for (int i = 0; i < file_size / unit.length(); i++) {
    TF_RETURN_IF_ERROR(w_file->Append(unit));
  }
  TF_RETURN_IF_ERROR(w_file->Append(unit.substr(0, file_size % unit.length())));
  TF_RETURN_IF_ERROR(w_file->Close());
  return Status::OK();
}

Status NewTask(size_t file_size, size_t task_size, bool eol,
               std::unique_ptr<internal::PrefetchTask>* result) {
  Env* env = Env::Default();
  string fname;
  TF_RETURN_IF_ERROR(PrepareDataFile(env, file_size, eol, &fname));
  internal::RAFHandle raf;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &raf));
  internal::BytesHandle bytes = absl::make_unique<char[]>(task_size);
  internal::TaskInfo info = {0, 0, task_size, task_size >= file_size};
  result->reset(
      new internal::PrefetchTask(std::move(bytes), std::move(raf), info));
  return Status::OK();
}

}  // namespace

TEST(PrefetchTask, FillAndReadSingleThread) {
  std::unique_ptr<internal::PrefetchTask> task;
  TF_ASSERT_OK((NewTask(100, 100, /* eol */ false, &task)));

  EXPECT_FALSE(task->IsFullFilled());  // Not started yet
  EXPECT_FALSE(task->IsReadable());

  TF_ASSERT_OK(task->Fill(60));
  EXPECT_FALSE(task->IsFullFilled());
  EXPECT_FALSE(task->IsReadable());

  TF_ASSERT_OK(task->Fill(60));
  EXPECT_FALSE(task->IsReadable());
  EXPECT_TRUE(task->IsFullFilled());

  task->NotifyFillDone();
  EXPECT_TRUE(task->IsReadable());

  tstring result;
  size_t actual_read = task->Read(200, &result);  // read 200 bytes,
  EXPECT_EQ(100, actual_read);                    // but only 100 available.
  EXPECT_EQ(100, result.length());
  for (int i = 0; i < 100 / kFileUnit.length(); i++) {
    StringPiece piece(result);
    auto sub = piece.substr(i * kFileUnit.length(), kFileUnit.length());
    EXPECT_EQ(0, sub.compare(kFileUnit.c_str()));
  }

  EXPECT_TRUE(task->IsReadExhausted());
}

TEST(PrefetchTask, FillAndReadMultiThread) {
  std::unique_ptr<internal::PrefetchTask> task;
  TF_ASSERT_OK((NewTask(100, 100, /* eol */ false, &task)));
  auto task_ptr = task.get();

  std::thread read_thread([task_ptr]() {
    EXPECT_FALSE(task_ptr->IsFullFilled());  // Not started yet
    TF_ASSERT_OK(task_ptr->Fill(60));
    EXPECT_FALSE(task_ptr->IsFullFilled());

    TF_ASSERT_OK(task_ptr->Fill(60));
    EXPECT_TRUE(task_ptr->IsFullFilled());
    task_ptr->NotifyFillDone();
  });

  while (!task->IsReadable()) {
    std::this_thread::yield();
  }

  tstring result;
  size_t actual_read = task->Read(200, &result);  // read 200 bytes,
  EXPECT_EQ(100, actual_read);                    // but only 100 available.
  EXPECT_EQ(100, result.length());
  for (int i = 0; i < 100 / kFileUnit.length(); i++) {
    StringPiece piece(result);
    auto sub = piece.substr(i * kFileUnit.length(), kFileUnit.length());
    EXPECT_EQ(0, sub.compare(kFileUnit.c_str()));
  }

  EXPECT_TRUE(task->IsReadExhausted());
  read_thread.join();
}

TEST(PrefetchTask, FillAndReadLine) {
  std::unique_ptr<internal::PrefetchTask> task;
  TF_ASSERT_OK((NewTask(100, 100, /* eol */ true, &task)));
  TF_ASSERT_OK(task->Fill(100));
  task->NotifyFillDone();

  while (!task->IsReadable()) {
    std::this_thread::yield();
  }

  string result;
  for (size_t i = 0; i < 100 / kFileUnitEOL.length(); i++) {
    result.clear();
    EXPECT_TRUE(task->ReadLine(false, &result));
    EXPECT_EQ(kFileUnit, result);
  }

  result.clear();
  EXPECT_FALSE(task->ReadLine(false, &result));
  EXPECT_EQ(kFileUnit.substr(0, 1), result);
  EXPECT_TRUE(task->IsReadExhausted());
}

TEST(PrefetchTask, Cancelation) {
  std::unique_ptr<internal::PrefetchTask> task;
  TF_ASSERT_OK((NewTask(_1_MB, 100, /* eol */ false, &task)));
  auto task_ptr = task.get();

  std::thread read_thread([task_ptr]() {
    while (!task_ptr->IsFullFilled() && !task_ptr->IsCancled()) {
      Status s = task_ptr->Fill(128 * 1024);
      // slow read
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (!s.ok()) break;
    }
    task_ptr->NotifyFillDone();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  task->Cancel();
  task->WaitForFillingDone();
  EXPECT_TRUE(task->IsCancled());
  EXPECT_TRUE(task->IsReadable());
  read_thread.join();
}

class SequencialReadFixture : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    file_len_ = GetParam();
    TF_EXPECT_OK(
        PrepareDataFile(Env::Default(), file_len_, /* eol */ false, &fname_));
  }

  void TearDown() override {}

 protected:
  size_t file_len_;
  string fname_;
};

TEST_P(SequencialReadFixture, SequencialRead) {
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname_, /* max_threads */ 4,
                                          /* buf_size */ 396, &stream));

  // Normal read.
  tstring result;
  for (size_t i = 0; i < file_len_ / kFileUnit.length(); i++) {
    EXPECT_EQ(i * kFileUnit.length(), stream->Tell());
    TF_ASSERT_OK(stream->ReadNBytes(kFileUnit.length(), &result));
    EXPECT_EQ(kFileUnit, result);
  }

  // Read to file end.
  Status s = stream->ReadNBytes(kFileUnit.length(), &result);
  EXPECT_TRUE(errors::IsOutOfRange(s));
  if (file_len_ % kFileUnit.length() == 0) {
    EXPECT_TRUE(result.empty());
  } else {
    EXPECT_EQ(kFileUnit.substr(0, file_len_ % kFileUnit.length()), result);
  }

  // Further reading always results in OutOfRange error.
  for (size_t i = 0; i < 10; i++) {
    EXPECT_EQ(file_len_, stream->Tell());
    Status s = stream->ReadNBytes(kFileUnit.length(), &result);
    EXPECT_TRUE(errors::IsOutOfRange(s));
    EXPECT_TRUE(result.empty());
  }
}

TEST(TestPrefetchedInputStream, ReadLine) {
  constexpr size_t file_len = 1024;
  string fname;
  TF_EXPECT_OK(
      PrepareDataFile(Env::Default(), file_len, /* eol */ true, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 396, &stream));
  // Normal read.
  string result;
  for (size_t i = 0; i < file_len / kFileUnitEOL.length(); i++) {
    EXPECT_EQ(i * kFileUnitEOL.length(), stream->Tell());
    TF_EXPECT_OK(stream->ReadLine(&result));
    EXPECT_EQ(kFileUnit, result);  // no trailing LF.
  }

  // Read to file end.
  Status s = stream->ReadLine(&result);
  EXPECT_TRUE(errors::IsOutOfRange(s));
  EXPECT_EQ(kFileUnit.substr(0, file_len % kFileUnitEOL.length()), result);

  // Further reading always results in OutOfRange error.
  for (size_t i = 0; i < 10; i++) {
    EXPECT_EQ(file_len, stream->Tell());
    Status s = stream->ReadLine(&result);
    EXPECT_TRUE(errors::IsOutOfRange(s));
    EXPECT_TRUE(result.empty());
  }
}

TEST(TestPrefetchedInputStream, ReadLineAsString) {
  constexpr size_t file_len = 1024;
  string fname;
  TF_EXPECT_OK(
      PrepareDataFile(Env::Default(), file_len, /* eol */ true, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 396, &stream));
  // Normal read.
  for (size_t i = 0; i < file_len / kFileUnitEOL.length(); i++) {
    EXPECT_EQ(i * kFileUnitEOL.length(), stream->Tell());
    string result = stream->ReadLineAsString();
    EXPECT_EQ(kFileUnitEOL, result);  // trailing LF included
  }

  // Read to file end.
  string result = stream->ReadLineAsString();
  EXPECT_EQ(kFileUnit.substr(0, file_len % kFileUnitEOL.length()), result);

  // Further reading always results in OutOfRange error.
  for (size_t i = 0; i < 10; i++) {
    EXPECT_EQ(file_len, stream->Tell());
    Status s = stream->ReadLine(&result);
    EXPECT_TRUE(errors::IsOutOfRange(s));
    EXPECT_TRUE(result.empty());
  }
}

TEST(TestPrefetchedInputStream, ReadAll) {
  constexpr size_t file_len = 1024;
  string fname;
  TF_EXPECT_OK(
      PrepareDataFile(Env::Default(), file_len, /* eol */ true, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 396, &stream));
  // Normal read.
  string result;
  EXPECT_OK(stream->ReadAll(&result));
  EXPECT_EQ(file_len, result.length());
  EXPECT_EQ(file_len, stream->Tell());

  // Further reading always results in OutOfRange error.
  Status s = stream->ReadLine(&result);
  EXPECT_TRUE(errors::IsOutOfRange(s));
  EXPECT_TRUE(result.empty());
}

INSTANTIATE_TEST_CASE_P(PrefetchedInputStreamSuit, SequencialReadFixture,
                        ::testing::Values(1000, 1024, _1_MB));

TEST(TestPrefetchedInputStream, LargeRead) {
  string fname;
  PrepareDataFile(Env::Default(), _1_MB, /* eol */ false, &fname);
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 400, &stream));
  tstring result;
  TF_EXPECT_OK(stream->ReadNBytes(_1_MB, &result));
  EXPECT_EQ(_1_MB, result.length());

  TF_EXPECT_OK(stream->Reset());
  Status s = stream->ReadNBytes(_1_MB * 2, &result);
  EXPECT_EQ(_1_MB, result.length());
  EXPECT_TRUE(errors::IsOutOfRange(s));
}

// TODO(xiafei.qiuxf): more sophisticated error tests.
TEST(TestPrefetchedInputStream, SeqReadError) {
  string fname;
  PrepareDataFile(Env::Default(), 10, /* eol */ false, &fname);
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 400, &stream));

  // Init OK, now let's remove the file.
  TF_ASSERT_OK(Env::Default()->DeleteFile(fname));

  tstring result;
  Status s = stream->ReadNBytes(10, &result);
  EXPECT_TRUE(!s.ok());
}

TEST(TestPrefetchedInputStream, Reset) {
  string fname;
  TF_ASSERT_OK(PrepareDataFile(Env::Default(), 1024, /* eol */ false, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 400, &stream));
  tstring result;
  for (size_t i = 0; i < 100; i++) {
    TF_EXPECT_OK(stream->Reset());
    EXPECT_EQ(0, stream->Tell());
    TF_EXPECT_OK(stream->ReadNBytes(kFileUnit.length(), &result));
    ASSERT_EQ(kFileUnit, result);
    EXPECT_EQ(kFileUnit.length(), stream->Tell());
  }
}

TEST(TestPrefetchedInputStream, SeekOnTaskBoundary) {
  string fname;
  TF_ASSERT_OK(PrepareDataFile(Env::Default(), 1024, /* eol */ false, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 400, &stream));
  tstring result;
  TF_EXPECT_OK(stream->ReadNBytes(kFileUnit.length(), &result));

  // wait all buffers ready.
  sleep(1);

  TF_EXPECT_OK(stream->Seek(100));
  TF_EXPECT_OK(stream->ReadNBytes(kFileUnit.length(), &result));
  ASSERT_EQ(kFileUnit, result);

  TF_EXPECT_OK(stream->Seek(400));
  TF_EXPECT_OK(stream->ReadNBytes(kFileUnit.length(), &result));
  ASSERT_EQ(kFileUnit, result);
}

TEST(TestPrefetchedInputStream, RandomSeek) {
  string fname;
  TF_ASSERT_OK(PrepareDataFile(Env::Default(), 1024, /* eol */ false, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 400, &stream));
  tstring result;
  for (size_t i = 0; i < 1000; i++) {
    size_t pos = static_cast<size_t>(random::New64() % (1024 - 10));
    TF_EXPECT_OK(stream->Seek(pos));
    EXPECT_EQ(pos, stream->Tell());
    TF_EXPECT_OK(stream->ReadNBytes(kFileUnit.length(), &result));
    ASSERT_EQ(kFileUnitX2.substr(pos % 10, kFileUnit.length()), result);
    EXPECT_EQ(pos + kFileUnit.length(), stream->Tell());
  }
}

TEST(TestPrefetchedInputStream, SkipNBytes) {
  string fname;
  TF_ASSERT_OK(PrepareDataFile(Env::Default(), 2000, /* eol */ false, &fname));
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 396, &stream));
  tstring result;
  for (size_t i = 0; i < 2000 / kFileUnit.length() / 2; i++) {
    TF_EXPECT_OK(stream->SkipNBytes(kFileUnit.length()));
    TF_EXPECT_OK(stream->ReadNBytes(kFileUnit.length(), &result));
    ASSERT_EQ(kFileUnit, result);
  }
}

TEST(TestPrefetchedInputStream, SeekToEOF) {
  string fname;
  TF_ASSERT_OK(PrepareDataFile(Env::Default(), 1024, /* eol */ false, &fname));
  // same as BufferedInputStream.
  std::unique_ptr<PrefetchedInputStream> stream;
  TF_ASSERT_OK(PrefetchedInputStream::New(fname, /* max_threads */ 4,
                                          /* buf_size */ 400, &stream));
  TF_EXPECT_OK(stream->Seek(0));
  TF_EXPECT_OK(stream->Seek(1024));
  Status s = stream->Seek(1025);
  EXPECT_TRUE(errors::IsOutOfRange(s));
}

void BenchStream(const string& bench_name, const string& fname,
                 size_t file_size, size_t threads) {
  std::unique_ptr<InputStreamInterface> stream;
  std::unique_ptr<RandomAccessFile> raf;
  if (bench_name == "prefetched") {
    std::unique_ptr<PrefetchedInputStream> tmp_stream;
    TF_ASSERT_OK(PrefetchedInputStream::New(fname, threads, threads * _1_MB,
                                            &tmp_stream));
    stream.reset(tmp_stream.release());
  } else {
    TF_ASSERT_OK(Env::Default()->NewRandomAccessFile(fname, &raf));
    stream.reset(new BufferedInputStream(raf.release(), threads * _1_MB));
  }

  auto start = std::chrono::system_clock::now();
  tstring result;
  size_t total_read = 0;
  while (true) {
    Status s = stream->ReadNBytes(1024, &result);
    if (s.ok()) {
      total_read += result.size();
    } else if (errors::IsOutOfRange(s)) {
      total_read += result.size();
      break;
    } else {
      break;
    }
  }
  ASSERT_EQ(total_read, file_size);
  auto end = std::chrono::system_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << bench_name << " finish, threads: " << threads
            << ", time(ms): " << time << std::endl;
}

TEST(TestPrefetchedInputStream, DISABLED_Benchmark) {
  size_t file_size = 500 * _1_MB;
  string fname;
  TF_ASSERT_OK(
      PrepareDataFile(Env::Default(), file_size, /* eol */ false, &fname));

  BenchStream("warmup", fname, file_size, 1);

  BenchStream("prefetched", fname, file_size, 1);
  BenchStream("prefetched", fname, file_size, 2);
  BenchStream("prefetched", fname, file_size, 4);
  BenchStream("prefetched", fname, file_size, 8);

  BenchStream("buffered", fname, file_size, 1);
  BenchStream("buffered", fname, file_size, 2);
  BenchStream("buffered", fname, file_size, 4);
  BenchStream("buffered", fname, file_size, 8);
}

}  // namespace io
}  // namespace tensorflow