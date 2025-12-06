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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/test_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestContext> ctx,
                          TestContext::Create());
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
  absl::StrAppend(&data.metadata_, "key1", "@@");
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
  absl::StrAppend(&data.metadata_, "key1", "@@");
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

class ParameterizedIteratorStateVariantTest
    : public DatasetOpsTestBase,
      public ::testing::WithParamInterface<std::vector<Tensor>> {
 protected:
  VariantTensorData GetVariantTensorData() const {
    std::vector<Tensor> tensors = GetParam();
    VariantTensorData data;
    data.set_type_name(IteratorStateVariant::TypeName());
    for (Tensor& tensor : tensors) {
      *data.add_tensors() = std::move(tensor);
    }
    return data;
  }

  absl::StatusOr<VariantTensorData> EncodeAndDecode(
      const VariantTensorData& data) const {
    IteratorStateVariant encoder;
    TF_RETURN_IF_ERROR(encoder.InitializeFromVariantData(
        std::make_unique<VariantTensorData>(data)));
    VariantTensorData encoded_data;
    encoder.Encode(&encoded_data);

    IteratorStateVariant decoder;
    decoder.Decode(encoded_data);
    return *decoder.GetData();
  }

  absl::StatusOr<VariantTensorData> DecodeUncompressed(
      const VariantTensorData& data) const {
    IteratorStateVariant decoder;
    decoder.Decode(data);
    return *decoder.GetData();
  }
};

class ParemeterizedCheckpointIndicesTest
    : public DatasetOpsTestBase,
      public ::testing::WithParamInterface<absl::flat_hash_set<int64_t>> {
 protected:
  absl::flat_hash_set<int64_t> GetCheckpointIndices() const {
    absl::flat_hash_set<int64_t> checkpoint_indices = GetParam();
    return checkpoint_indices;
  }
};

std::vector<std::vector<Tensor>> TestCases() {
  return {
      CreateTensors<int64_t>(TensorShape{1}, {{1}}),           // int64
      CreateTensors<int64_t>(TensorShape{1}, {{1}, {2}}),      // multiple int64
      CreateTensors<tstring>(TensorShape{1}, {{"a"}, {"b"}}),  // tstring
      {CreateTensor<tstring>(TensorShape{1}, {"a"}),
       CreateTensor<int64_t>(TensorShape{1}, {1})},  // mixed tstring/int64
      {},                                            // empty
      {CreateTensor<int64_t>(TensorShape{128, 128}),
       CreateTensor<int64_t>(TensorShape{64, 2})},  // larger components
  };
}

std::vector<absl::flat_hash_set<int64_t>> CheckpointIndicesTestCases() {
  return {
      {/*checkpoint_indices*/},
      {/*checkpoint_indices*/ 0},
      {/*checkpoint_indices*/ 0, 1},
      {/*checkpoint_indices*/ 0, 1, 2},
      {/*checkpoint_indices*/ 1, 3, 4},
      {/*checkpoint_indices*/ 1, 2, 3, 4},
      {/*checkpoint_indices*/ 0, 1, 2, 3, 4},
  };
}

TEST_P(ParameterizedIteratorStateVariantTest, EncodeAndDecode) {
  VariantTensorData data = GetVariantTensorData();
  TF_ASSERT_OK_AND_ASSIGN(VariantTensorData result, EncodeAndDecode(data));

  EXPECT_EQ(result.type_name(), data.type_name());
  for (int i = 0; i < result.tensors_size(); ++i) {
    test::ExpectEqual(result.tensors(i), data.tensors(i));
  }
}

TEST_P(ParameterizedIteratorStateVariantTest, DecodeUncompressed) {
  VariantTensorData data = GetVariantTensorData();
  TF_ASSERT_OK_AND_ASSIGN(VariantTensorData result, DecodeUncompressed(data));

  EXPECT_EQ(result.type_name(), data.type_name());
  for (int i = 0; i < result.tensors_size(); ++i) {
    test::ExpectEqual(result.tensors(i), data.tensors(i));
  }
}

TEST_P(ParemeterizedCheckpointIndicesTest,
       CheckpointElementsRoundTripUsingIndices) {
  std::vector<std::vector<Tensor>> elements;
  elements.push_back(CreateTensors<int32>(TensorShape({3}), {{1, 2, 3}}));
  elements.push_back(CreateTensors<int32>(TensorShape({2}), {{4, 5}}));
  elements.push_back(
      CreateTensors<int32>(TensorShape({5}), {{6, 7, 8, 9, 10}}));
  elements.push_back(
      CreateTensors<int32>(TensorShape({4}), {{11, 12, 13, 14}}));
  elements.push_back(CreateTensors<int32>(TensorShape({2}), {{15, 16}}));
  VariantTensorDataWriter writer;
  tstring test_prefix = full_name("test_prefix");
  // Generate checkpoint for entire buffer
  absl::flat_hash_set<int64_t> checkpoint_indices_write = {0, 1, 2, 3, 4};
  TF_ASSERT_OK(WriteElementsToCheckpoint(&writer, test_prefix, elements));
  // Update the elements at checkpoint indices
  for (auto index : GetCheckpointIndices()) {
    elements.at(index) = CreateTensors<int32>(TensorShape({1}), {{1}});
  }
  TF_ASSERT_OK(UpdateCheckpointElements(&writer, test_prefix, elements,
                                        GetCheckpointIndices()));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  std::vector<std::vector<Tensor>> read_elements;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestContext> ctx,
                          TestContext::Create());
  TF_ASSERT_OK(ReadElementsFromCheckpoint(ctx->iter_ctx(), &reader, test_prefix,
                                          &read_elements));

  ASSERT_EQ(elements.size(), read_elements.size());
  // Check if checkpoint state of entire buffer is as expected
  for (int index = 0; index < elements.size(); ++index) {
    std::vector<Tensor>& original = elements[index];
    std::vector<Tensor>& read = read_elements[index];

    ASSERT_EQ(original.size(), read.size());
    for (int j = 0; j < original.size(); ++j) {
      EXPECT_EQ(original[j].NumElements(), read[j].NumElements());
      EXPECT_EQ(original[j].flat<int32>()(0), read[j].flat<int32>()(0));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Instantiation, ParameterizedIteratorStateVariantTest,
                         ::testing::ValuesIn(TestCases()));

INSTANTIATE_TEST_SUITE_P(Instantiation, ParemeterizedCheckpointIndicesTest,
                         ::testing::ValuesIn(CheckpointIndicesTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
