/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class RangeIterator : public TaskIterator {
 public:
  explicit RangeIterator(const int64_t range) : range_(range) {}

  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override {
    end_of_sequence = (next_ >= range_);
    if (end_of_sequence) {
      return OkStatus();
    }
    element = {Tensor{next_++}};
    return OkStatus();
  }

  int64_t Cardinality() const override { return range_; }

 private:
  const int64_t range_;
  int64_t next_ = 0;
};

StatusOr<std::unique_ptr<snapshot_util::Reader>> CreateSnapshotReader(
    const std::string& snapshot_path, int64_t num_elements, Env* env) {
  static constexpr int kTFRecordReader = 2;
  DataTypeVector dtypes(num_elements, DT_INT64);
  std::unique_ptr<snapshot_util::Reader> reader;
  TF_RETURN_IF_ERROR(snapshot_util::Reader::Create(
      env, snapshot_path, tsl::io::compression::kNone, kTFRecordReader, dtypes,
      &reader));
  return reader;
}

template <class T>
StatusOr<std::vector<T>> ReadSnapshot(const std::string& snapshot_path,
                                      int64_t num_elements) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<snapshot_util::Reader> reader,
      CreateSnapshotReader(snapshot_path, num_elements, Env::Default()));

  std::vector<Tensor> tensors;
  TF_RETURN_IF_ERROR(reader->ReadTensors(&tensors));

  std::vector<T> result;
  for (const Tensor& tensor : tensors) {
    result.push_back(tensor.unaligned_flat<T>().data()[0]);
  }
  return result;
}

TEST(SnapshotStreamWriterTest, WriteSnapshot) {
  const int64_t range = 10;
  std::string snapshot_path;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&snapshot_path));

  SnapshotStreamWriter snapshot_writer(std::make_unique<RangeIterator>(range),
                                       snapshot_path, Env::Default());
  TF_ASSERT_OK(snapshot_writer.Wait());

  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, range),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotStreamWriterTest, Cancel) {
  const int64_t range = 10000;
  std::string snapshot_path;
  EXPECT_TRUE(Env::Default()->LocalTempFilename(&snapshot_path));

  SnapshotStreamWriter snapshot_writer(std::make_unique<RangeIterator>(range),
                                       snapshot_path, Env::Default());
  snapshot_writer.Cancel();
  EXPECT_THAT(snapshot_writer.Wait(), StatusIs(error::CANCELLED));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
