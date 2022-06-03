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

#include "tensorflow/core/data/serialization_utils.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

class TestContext {
 public:
  static Status Create(std::unique_ptr<TestContext>* result) {
    *result = absl::WrapUnique<TestContext>(new TestContext());

    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 1});
    std::vector<std::unique_ptr<Device>> devices;
    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    (*result)->device_mgr_ =
        absl::make_unique<StaticDeviceMgr>(std::move(devices));

    FunctionDefLibrary proto;
    (*result)->lib_def_ = absl::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), proto);

    OptimizerOptions opts;
    (*result)->pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        (*result)->device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, (*result)->lib_def_.get(), opts);
    (*result)->runner_ = [](const std::function<void()>& fn) { fn(); };
    (*result)->params_.function_library =
        (*result)->pflr_->GetFLR("/device:CPU:0");
    (*result)->params_.device = (*result)->device_mgr_->ListDevices()[0];
    (*result)->params_.runner = &(*result)->runner_;
    (*result)->op_ctx_ =
        absl::make_unique<OpKernelContext>(&(*result)->params_, 0);
    (*result)->iter_ctx_ =
        absl::make_unique<IteratorContext>((*result)->op_ctx_.get());
    return OkStatus();
  }

  IteratorContext* iter_ctx() const { return iter_ctx_.get(); }

 private:
  TestContext() = default;

  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::function<void(std::function<void()>)> runner_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> op_ctx_;
  std::unique_ptr<IteratorContext> iter_ctx_;
};

string full_name(string key) { return FullName("Iterator:", key); }

TEST(SerializationUtilsTest, CheckpointElementsRoundTrip) {
  std::vector<std::vector<Tensor>> elements;
  elements.push_back(CreateTensors<int32>(TensorShape({3}), {{1, 2, 3}}));
  elements.push_back(CreateTensors<int32>(TensorShape({2}), {{4, 5}}));
  VariantTensorDataWriter writer;
  tstring test_prefix = full_name("test_prefix");
  TF_ASSERT_OK(WriteElementsToCheckpoint(&writer, test_prefix, elements));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  std::vector<std::vector<Tensor>> read_elements;

  std::unique_ptr<TestContext> ctx;
  TF_ASSERT_OK(TestContext::Create(&ctx));
  TF_ASSERT_OK(ReadElementsFromCheckpoint(ctx->iter_ctx(), &reader, test_prefix,
                                          &read_elements));
  ASSERT_EQ(elements.size(), read_elements.size());
  for (int i = 0; i < elements.size(); ++i) {
    std::vector<Tensor>& original = elements[i];
    std::vector<Tensor>& read = read_elements[i];

    ASSERT_EQ(original.size(), read.size());
    for (int j = 0; j < original.size(); ++j) {
      EXPECT_EQ(original[j].NumElements(), read[j].NumElements());
      EXPECT_EQ(original[j].flat<int32>()(0), read[j].flat<int32>()(0));
    }
  }
}

TEST(SerializationUtilsTest, VariantTensorDataRoundtrip) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar(full_name("Int64"), 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor(full_name("Tensor"), input_tensor));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  int64_t val_int64;
  TF_ASSERT_OK(reader.ReadScalar(full_name("Int64"), &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor(full_name("Tensor"), &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(SerializationUtilsTest, VariantTensorDataNonExistentKey) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  std::vector<const VariantTensorData*> reader_data;
  reader_data.push_back(&data);
  VariantTensorDataReader reader(reader_data);
  int64_t val_int64;
  tstring val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar(full_name("NonExistentKey"), &val_int64).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar(full_name("NonExistentKey"), &val_string).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadTensor(full_name("NonExistentKey"), &val_tensor).code());
}

TEST(SerializationUtilsTest, VariantTensorDataRoundtripIteratorName) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar("Iterator", "Int64", 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor("Iterator", "Tensor", input_tensor));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  int64_t val_int64;
  TF_ASSERT_OK(reader.ReadScalar("Iterator", "Int64", &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor("Iterator", "Tensor", &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(SerializationUtilsTest, VariantTensorDataNonExistentKeyIteratorName) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  std::vector<const VariantTensorData*> reader_data;
  reader_data.push_back(&data);
  VariantTensorDataReader reader(reader_data);
  int64_t val_int64;
  tstring val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("Iterator", "NonExistentKey", &val_int64).code());
  EXPECT_EQ(
      error::NOT_FOUND,
      reader.ReadScalar("Iterator", "NonExistentKey", &val_string).code());
  EXPECT_EQ(
      error::NOT_FOUND,
      reader.ReadTensor("Iterator", "NonExistentKey", &val_tensor).code());
}

TEST(SerializationUtilsTest, VariantTensorDataWriteAfterFlushing) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar(full_name("Int64"), 24));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  EXPECT_EQ(error::FAILED_PRECONDITION,
            writer.WriteTensor(full_name("Tensor"), input_tensor).code());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
