/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/c/checkpoint_reader.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "tensorflow/c/tf_status.h"
#include "xla/tsl/platform/file_system.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace checkpoint {
namespace {

constexpr absl::string_view kSchemePrefix = "failread://";

class FailingAfterNReadsRandomAccessFile : public RandomAccessFile {
 public:
  FailingAfterNReadsRandomAccessFile(std::unique_ptr<RandomAccessFile> wrapped,
                                     int fail_after_reads,
                                     std::atomic<int>* read_count)
      : wrapped_(std::move(wrapped)),
        fail_after_reads_(fail_after_reads),
        read_count_(read_count) {}

  absl::Status Read(uint64_t offset, absl::string_view& result,
                    absl::Span<char> scratch) const override {
    int current_read = read_count_->fetch_add(1, std::memory_order_relaxed);
    if (fail_after_reads_ >= 0 && current_read >= fail_after_reads_) {
      return absl::DeadlineExceededError("Simulated block read timeout");
    }
    return wrapped_->Read(offset, result, scratch);
  }

 private:
  std::unique_ptr<RandomAccessFile> wrapped_;
  int fail_after_reads_;
  std::atomic<int>* read_count_;
};

class FailingBlockReadFileSystem : public NullFileSystem {
 public:
  void Reset(int fail_after_reads) {
    fail_after_reads_ = fail_after_reads;
    read_count_.store(0, std::memory_order_relaxed);
  }

  int read_count() const { return read_count_.load(std::memory_order_relaxed); }

  absl::Status FileExists(absl::string_view fname) override {
    return Env::Default()->FileExists(StripScheme(fname));
  }

  absl::Status GetFileSize(const std::string& fname,
                           uint64_t* file_size) override {
    return Env::Default()->GetFileSize(StripScheme(fname), file_size);
  }

  absl::Status NewRandomAccessFile(
      const std::string& fname,
      std::unique_ptr<RandomAccessFile>* result) override {
    std::unique_ptr<RandomAccessFile> underlying;
    absl::Status s =
        Env::Default()->NewRandomAccessFile(StripScheme(fname), &underlying);
    if (!s.ok()) {
      return s;
    }
    *result = std::make_unique<FailingAfterNReadsRandomAccessFile>(
        std::move(underlying), fail_after_reads_, &read_count_);
    return absl::OkStatus();
  }

 private:
  static std::string StripScheme(absl::string_view fname) {
    return std::string(absl::StripPrefix(fname, kSchemePrefix));
  }

  int fail_after_reads_ = -1;
  std::atomic<int> read_count_{0};
};

class CheckpointReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (fs_ == nullptr) {
      auto fs = std::make_unique<FailingBlockReadFileSystem>();
      fs_ = fs.get();
      ASSERT_TRUE(
          Env::Default()->RegisterFileSystem("failread", std::move(fs)).ok());
    }
  }

  static inline FailingBlockReadFileSystem* fs_ = nullptr;
};

TEST_F(CheckpointReaderTest, PropagatesBlockReadErrorInBuildV2VarMaps) {
  const std::string prefix = io::JoinPath(testing::TmpDir(), "v2_ckpt");
  // Exceed the default 256 KiB table block size so iteration spans multiple
  // data blocks and exercises Next() block reads in both passes.
  constexpr int kNumVars = 300;
  const std::string suffix(1024, 'a');
  {
    BundleWriter writer(Env::Default(), prefix);
    Tensor t(DT_FLOAT, TensorShape({2}));
    t.flat<float>()(0) = 1.0f;
    t.flat<float>()(1) = 2.0f;
    for (int i = 0; i < kNumVars; ++i) {
      ASSERT_TRUE(writer.Add(absl::StrCat("var_", i, "_", suffix), t).ok());
    }
    ASSERT_TRUE(writer.Finish().ok());
  }

  const std::string wrapped_prefix = absl::StrCat(kSchemePrefix, prefix);

  // Baseline read without injected failures to count total Read() calls.
  fs_->Reset(-1);
  {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    CheckpointReader reader(wrapped_prefix, status.get());
    ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    EXPECT_EQ(reader.GetVariableToShapeMap().size(), kNumVars);
  }
  const int total_reads = fs_->read_count();
  ASSERT_GE(total_reads, 6);

  // Inject a failure on every possible read offset, covering Table::Open,
  // BundleReader header Seek(), BuildV2VarMaps() pass 1 Next(), and
  // BuildV2VarMaps() pass 2 Seek(kHeaderEntryKey).
  for (int fail_after = 0; fail_after < total_reads; ++fail_after) {
    fs_->Reset(fail_after);
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    CheckpointReader reader(wrapped_prefix, status.get());
    EXPECT_EQ(TF_GetCode(status.get()), TF_DEADLINE_EXCEEDED)
        << "fail_after=" << fail_after << " msg=" << TF_Message(status.get());
  }
}

}  // namespace
}  // namespace checkpoint
}  // namespace tensorflow
