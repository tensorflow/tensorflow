/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/data_transfer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

class TestDataTransferServer : public DataTransferServer {
 public:
  explicit TestDataTransferServer(bool* called) : called_(called) {}
  Status Start(const experimental::WorkerConfig& unused_config) override {
    *called_ = true;
    return absl::OkStatus();
  }
  int Port() const override { return 0; }

 private:
  bool* called_;
};

template <class T>
GetElementResult MakeElementResult(T value) {
  GetElementResult result;
  result.components.push_back(Tensor(std::move(value)));
  result.element_index = 0;
  result.end_of_sequence = false;
  return result;
}

TEST(DataTransferTest, RegisterDataTransferServerBuilder) {
  bool called = false;
  DataTransferServer::Register("test", [&called](auto ignore, auto* server) {
    *server = std::make_shared<TestDataTransferServer>(&called);
    return absl::OkStatus();
  });

  std::shared_ptr<DataTransferServer> server;
  TF_ASSERT_OK(DataTransferServer::Build("test", {}, &server));
  EXPECT_FALSE(called);

  TF_ASSERT_OK(server->Start(/*config=*/{}));
  EXPECT_TRUE(called);
}

TEST(DataTransferTest, EstimateMemoryUsageBytes) {
  GetElementResult empty;
  EXPECT_GT(empty.EstimatedMemoryUsageBytes(), 0);

  Tensor tensor(DT_INT64, TensorShape({10, 100}));
  GetElementResult int64_result = MakeElementResult(tensor);
  EXPECT_GT(int64_result.EstimatedMemoryUsageBytes(), 1000 * sizeof(int64_t));
  EXPECT_GT(int64_result.EstimatedMemoryUsageBytes(),
            int64_result.components[0].AllocatedBytes());
  EXPECT_GE(int64_result.EstimatedMemoryUsageBytes(), sizeof(int64_result));
}

TEST(DataTransferTest, EstimateVariantMemoryUsageBytes) {
  const size_t data_size = 1000;

  std::unique_ptr<CompressedElement> compressed{
      protobuf::Arena::Create<CompressedElement>(nullptr)};
  compressed->set_data(std::string(data_size, 'a'));

  Tensor tensor(DT_VARIANT, TensorShape({}));
  tensor.scalar<Variant>()() = *compressed;

  GetElementResult variant_result = MakeElementResult(tensor);
  EXPECT_GT(variant_result.EstimatedMemoryUsageBytes(), data_size);
  EXPECT_GT(variant_result.EstimatedMemoryUsageBytes(),
            compressed->ByteSizeLong());
  EXPECT_GT(variant_result.EstimatedMemoryUsageBytes(),
            compressed->SpaceUsedLong());
}

TEST(DataTransferTest, CopyGetElementResult) {
  std::string hello_world = "hello, world!";
  GetElementResult result = MakeElementResult(hello_world);
  ASSERT_EQ(result.components.size(), 1);
  EXPECT_GT(result.EstimatedMemoryUsageBytes(), hello_world.size());

  GetElementResult copy = result.Copy();
  ASSERT_EQ(copy.components.size(), 1);
  test::ExpectEqual(result.components[0], copy.components[0]);
  EXPECT_EQ(copy.EstimatedMemoryUsageBytes(),
            result.EstimatedMemoryUsageBytes());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
