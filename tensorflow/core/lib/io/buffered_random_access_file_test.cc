#include "tensorflow/core/lib/io/buffered_random_access_file.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace io {

class TestBufferedRandomAccessFile : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    buf_size_ = GetParam();
    env_ = Env::Default();
    fname_ = testing::TmpDir() + "/buffered_random_access_file_test";
    StringPiece element = elements_.substr(0, 10);  // "0123456789"

    {
      // create a file containing as many as `file_len` repeating digits.
      env_->DeleteFile(fname_).IgnoreError();
      std::unique_ptr<WritableFile> writable;
      TF_ASSERT_OK(env_->NewAppendableFile(fname_, &writable));
      for (int i = 0; i < file_len_ / element.length(); i++) {
        TF_ASSERT_OK(writable->Append(element));
      }
      TF_ASSERT_OK(writable->Close());
    }
  }
  void TearDown() override {}

 protected:
  Env* env_;
  const size_t file_len_ = 1000;  // make sure file_len_ % 10 == 0
  const StringPiece elements_ =
      "01234567890123456789012345678901234567890123456789";  // len = 50
  size_t buf_size_;
  string fname_;
};

TEST_P(TestBufferedRandomAccessFile, SequencialRead) {
  std::unique_ptr<RandomAccessFile> raf;
  TF_ASSERT_OK(env_->NewRandomAccessFile(fname_, &raf));
  char scratch[1024];

  BufferedRandomAccessFile braf(std::move(raf), buf_size_);
  StringPiece result;
  uint64 offset = 0;
  while (offset < file_len_) {
    // may read more than a buffer size.
    size_t n = 1 + random::New64() % (buf_size_ * 2);
    // read exactly to file end.
    n = std::min<size_t>(n, file_len_ - offset);
    EXPECT_OK(braf.Read(offset, n, &result, scratch));
    EXPECT_EQ(result.length(), n);
    EXPECT_EQ(result, elements_.substr(offset % 10, n));
    offset += n;
  }
  Status status = braf.Read(file_len_ - 20, 25, &result, scratch);
  EXPECT_TRUE(errors::IsOutOfRange(status));
  EXPECT_EQ(result, elements_.substr(0, 20));
}

TEST_P(TestBufferedRandomAccessFile, RandomRead) {
  std::unique_ptr<RandomAccessFile> raf;
  TF_ASSERT_OK(env_->NewRandomAccessFile(fname_, &raf));
  char scratch[1024];

  BufferedRandomAccessFile braf(std::move(raf), buf_size_);
  StringPiece result;

  // Repeat random reading within a relatively small range.
  size_t range_start = 900;
  size_t range_size = 100;
  for (int i = 0; i < 100; i++) {
    size_t offset = range_start + random::New64() % range_size;
    size_t n = 1 + random::New64() % (buf_size_ * 2);
    Status status = braf.Read(offset, n, &result, scratch);
    if (offset + n > file_len_) {
      EXPECT_TRUE(errors::IsOutOfRange(status));
    } else {
      EXPECT_TRUE(status.ok());
    }
    size_t expected_len = std::min<size_t>(n, file_len_ - offset);
    EXPECT_EQ(result, elements_.substr(offset % 10, expected_len));
  }
}

INSTANTIATE_TEST_CASE_P(BufferedRandomAccessFileSuit,
                        TestBufferedRandomAccessFile,
                        ::testing::Range((size_t)2, (size_t)20, (size_t)2));

void BM_BufferedSmallSeqReads(const int iters, const int buf_size,
                              const int file_size) {
  testing::StopTiming();
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_random_access_file_bm_test";

  const string file_elem = "0123456789";
  {
    std::unique_ptr<WritableFile> write_file;
    TF_ASSERT_OK(env->NewWritableFile(fname, &write_file));
    for (int i = 0; i < file_size / file_elem.size(); ++i) {
      TF_ASSERT_OK(write_file->Append(file_elem));
    }
    TF_ASSERT_OK(write_file->Close());
  }

  StringPiece result;
  char scratch[1024];
  testing::StartTiming();

  for (int itr = 0; itr < iters; ++itr) {
    std::unique_ptr<RandomAccessFile> raf;
    TF_ASSERT_OK(env->NewRandomAccessFile(fname, &raf));
    BufferedRandomAccessFile braf(std::move(raf), buf_size);
    for (int64 i = 0; i < file_size; ++i) {
      TF_ASSERT_OK(braf.Read(i, 1, &result, scratch));
    }
  }
}

// To run benchmarks, you may need to add `timeout = "eternal"` to bazel config.
BENCHMARK(BM_BufferedSmallSeqReads)
    ->ArgPair(8, 10 * 1024 * 1024)
    ->ArgPair(32, 10 * 1024 * 1024)
    ->ArgPair(128, 10 * 1024 * 1024)
    ->ArgPair(512, 10 * 1024 * 1024);

}  // namespace io
}  // namespace tensorflow