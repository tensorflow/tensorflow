/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/tensor_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tensor_dataset";

class TensorDatasetParams : public DatasetParams {
 public:
  TensorDatasetParams(std::vector<Tensor> components, string node_name)
      : DatasetParams(TensorDtypes(components), TensorShapes(components),
                      std::move(node_name)),
        components_(std::move(components)) {}

  std::vector<Tensor> GetInputTensors() const override { return components_; }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->reserve(components_.size());
    for (int i = 0; i < components_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(TensorDatasetOp::kComponents, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{TensorDatasetOp::kToutput_types, output_dtypes_},
                    {TensorDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  string dataset_type() const override { return TensorDatasetOp::kDatasetType; }

 private:
  DataTypeVector TensorDtypes(const std::vector<Tensor>& input_components) {
    DataTypeVector dtypes;
    for (const auto& component : input_components) {
      dtypes.emplace_back(component.dtype());
    }
    return dtypes;
  }

  std::vector<PartialTensorShape> TensorShapes(
      const std::vector<Tensor>& input_components) {
    std::vector<PartialTensorShape> shapes;
    for (const auto& component : input_components) {
      shapes.emplace_back(component.shape());
    }
    return shapes;
  }

 public:
  std::vector<Tensor> components_;
};

class TensorDatasetOpTest : public DatasetOpsTestBase {};

std::vector<Tensor> PlainTensors() {
  return {CreateTensor<int64>(TensorShape({}), {1}),
          CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3}),
          CreateTensor<double>(TensorShape({}), {37.0}),
          CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})};
}

// Test case 1: test a dataset that represents a single tuple of plain tensors.
TensorDatasetParams PlainTensorDatasetParams() {
  return {/*components=*/PlainTensors(),
          /*node_name=*/kNodeName};
}

// Test case 2: test a dataset that represents a tuple of nested tensors.
TensorDatasetParams NestedTensorDatasetParams() {
  return {/*components=*/
          {CreateTensor<Variant>(TensorShape({}),
                                 {CreateTensor<double>(TensorShape({2, 2}),
                                                       {1.0, 2.0, 3.0, 4.0})}),
           CreateTensor<Variant>(
               TensorShape({}),
               {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
           CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3})},
          /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<TensorDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PlainTensorDatasetParams(),
           /*expected_outputs=*/PlainTensors()},
          {/*dataset_params=*/NestedTensorDatasetParams(),
           /*expected_outputs=*/
           {CreateTensor<Variant>(TensorShape({}),
                                  {CreateTensor<double>(TensorShape({2, 2}),
                                                        {1.0, 2.0, 3.0, 4.0})}),
            CreateTensor<Variant>(
                TensorShape({}),
                {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
            CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3})}}};
}

class ParameterizedGetNextTest : public TensorDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<TensorDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
  }
  EXPECT_EQ(out_tensors.size(), test_case.expected_outputs.size());
  for (int i = 0; i < out_tensors.size(); ++i) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor* output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor* expected_output =
          test_case.expected_outputs[i].scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], test_case.expected_outputs[i]));
    }
  }
}

INSTANTIATE_TEST_CASE_P(TensorDatasetOpTest, ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

TEST_F(TensorDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(TensorDatasetOp::kDatasetType)));
}

TEST_F(TensorDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TensorDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(TensorDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

TEST_F(TensorDatasetOpTest, Cardinality) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(1));
}

TEST_F(TensorDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(TensorDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(TensorDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      "FromTensor", dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<TensorDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PlainTensorDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           PlainTensors()},
          {/*dataset_params=*/NestedTensorDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {CreateTensor<Variant>(TensorShape({}),
                                  {CreateTensor<double>(TensorShape({2, 2}),
                                                        {1.0, 2.0, 3.0, 4.0})}),
            CreateTensor<Variant>(
                TensorShape({}),
                {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
            CreateTensor<int64>(TensorShape({1, 3}), {1, 2, 3})}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public TensorDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<TensorDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  int cardinality = 1;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }

    if (breakpoint >= cardinality) {
      EXPECT_TRUE(end_of_sequence);
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }

  EXPECT_EQ(out_tensors.size(), test_case.expected_outputs.size());
  for (int i = 0; i < out_tensors.size(); ++i) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor* output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor* expected_output =
          test_case.expected_outputs[i].scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], test_case.expected_outputs[i]));
    }
  }
}

INSTANTIATE_TEST_CASE_P(TensorDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(TensorDatasetOpTest, Splitting) {
  auto params = PlainTensorDatasetParams();
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, /*expected_outputs=*/PlainTensors()));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/2,
      /*expected_outputs=*/CreateTensors<int64>(TensorShape({}), {})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/0,
      /*expected_outputs=*/PlainTensors()));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
