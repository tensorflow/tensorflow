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
#include "tensorflow/core/kernels/data/experimental/list_dataset_op.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tensor_list_dataset";

class ListDatasetOpTest : public DatasetOpsTestBase {};

class ListDatasetParams : public DatasetParams {
 public:
  ListDatasetParams(std::vector<std::vector<Tensor>> elements, string node_name)
      : DatasetParams(ListOutputTypes(elements), ListOutputShapes(elements),
                      std::move(node_name)) {
    input_types_.reserve(elements.size() * elements.front().size());
    tensors_.reserve(elements.size() * elements.front().size());
    for (const auto& element : elements) {
      for (const auto& tensor : element) {
        input_types_.push_back(tensor.dtype());
        tensors_.emplace_back(std::move(tensor));
      }
    }
  }

  std::vector<Tensor> GetInputTensors() const override { return tensors_; }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->reserve(tensors_.size());
    for (int i = 0; i < tensors_.size(); ++i) {
      input_names->emplace_back(absl::StrCat("tensors_", i));
    }
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{"Tinput_types", input_types_},
                    {"output_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"metadata", ""}};
    return OkStatus();
  }

  string dataset_type() const override { return "List"; }

  int64_t num_elements() const {
    return tensors_.size() / num_tensors_per_element();
  }

  size_t num_tensors_per_element() const { return output_shapes_.size(); }

 private:
  DataTypeVector ListInputTypes(
      const std::vector<std::vector<Tensor>>& input_elements) {
    DataTypeVector input_types;
    for (const auto& element : input_elements) {
      for (const auto& tensor : element) {
        input_types.emplace_back(tensor.dtype());
      }
    }
    return input_types;
  }

  DataTypeVector ListOutputTypes(
      const std::vector<std::vector<Tensor>>& input_elements) {
    DataTypeVector output_types;
    for (const auto& tensor : input_elements.front()) {
      output_types.emplace_back(tensor.dtype());
    }
    return output_types;
  }

  std::vector<PartialTensorShape> ListOutputShapes(
      const std::vector<std::vector<Tensor>>& input_elements) {
    std::vector<PartialTensorShape> output_shapes;
    for (const auto& tensor : input_elements.front()) {
      gtl::InlinedVector<int64_t, 4> partial_dim_sizes;
      partial_dim_sizes.reserve(tensor.dims());
      for (int i = 0; i < tensor.dims(); ++i) {
        partial_dim_sizes.push_back(tensor.dim_size(i));
      }
      output_shapes.emplace_back(std::move(partial_dim_sizes));
    }
    return output_shapes;
  }

 public:
  std::vector<Tensor> tensors_;
  DataTypeVector input_types_;
};

ListDatasetParams PlainListDatasetParams() {
  std::vector<std::vector<Tensor>> elements = {
      {CreateTensor<int64_t>(TensorShape({}), {1}),
       CreateTensor<int64_t>(TensorShape({2}), {1, 2}),
       CreateTensor<uint32>(TensorShape({}), {2}),
       CreateTensor<uint32>(TensorShape({2}), {2, 3}),
       CreateTensor<uint64>(TensorShape({}), {3}),
       CreateTensor<uint64>(TensorShape({2}), {3, 4}),
       CreateTensor<double>(TensorShape({1}), {37.0}),
       CreateTensor<tstring>(TensorShape({1}), {"a"})},
      {CreateTensor<int64_t>(TensorShape({}), {2}),
       CreateTensor<int64_t>(TensorShape({2}), {3, 4}),
       CreateTensor<uint32>(TensorShape({}), {3}),
       CreateTensor<uint32>(TensorShape({2}), {4, 5}),
       CreateTensor<uint64>(TensorShape({}), {4}),
       CreateTensor<uint64>(TensorShape({2}), {5, 6}),
       CreateTensor<double>(TensorShape({1}), {38.0}),
       CreateTensor<tstring>(TensorShape({1}), {"b"})}};

  return {std::move(elements), kNodeName};
}

ListDatasetParams NestedListDatasetParams() {
  std::vector<std::vector<Tensor>> elements = {
      {CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
       CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
       CreateTensor<int64_t>(TensorShape({3}), {1, 2, 3})},
      {CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
       CreateTensor<Variant>(
           TensorShape({1}),
           {CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
       CreateTensor<int64_t>(TensorShape({3}), {4, 5, 6})}};

  return {std::move(elements), kNodeName};
}

std::vector<GetNextTestCase<ListDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/PlainListDatasetParams(),
       /*expected_outputs=*/{CreateTensor<int64_t>(TensorShape({}), {1}),
                             CreateTensor<int64_t>(TensorShape({2}), {1, 2}),
                             CreateTensor<uint32>(TensorShape({}), {2}),
                             CreateTensor<uint32>(TensorShape({2}), {2, 3}),
                             CreateTensor<uint64>(TensorShape({}), {3}),
                             CreateTensor<uint64>(TensorShape({2}), {3, 4}),
                             CreateTensor<double>(TensorShape({1}), {37.0}),
                             CreateTensor<tstring>(TensorShape({1}), {"a"}),
                             CreateTensor<int64_t>(TensorShape({}), {2}),
                             CreateTensor<int64_t>(TensorShape({2}), {3, 4}),
                             CreateTensor<uint32>(TensorShape({}), {3}),
                             CreateTensor<uint32>(TensorShape({2}), {4, 5}),
                             CreateTensor<uint64>(TensorShape({}), {4}),
                             CreateTensor<uint64>(TensorShape({2}), {5, 6}),
                             CreateTensor<double>(TensorShape({1}), {38.0}),
                             CreateTensor<tstring>(TensorShape({1}), {"b"})}},
      {/*dataset_params=*/NestedListDatasetParams(),
       /*expected_outputs=*/
       {CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
        CreateTensor<int64_t>(TensorShape({3}), {1, 2, 3}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
        CreateTensor<int64_t>(TensorShape({3}), {4, 5, 6})}}};
}

class ParameterizedGetNextTest
    : public ListDatasetOpTest,
      public ::testing::WithParamInterface<GetNextTestCase<ListDatasetParams>> {
};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  size_t num_tensors_per_element =
      test_case.dataset_params.num_tensors_per_element();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_element = 0;

  while (true) {
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
    if (end_of_sequence) {
      EXPECT_TRUE(out_tensors.empty());
      break;
    }
    for (int i = 0; i < out_tensors.size(); ++i) {
      EXPECT_LT(i + num_tensors_per_element * cur_element,
                test_case.expected_outputs.size());
      if (out_tensors[i].dtype() == DT_VARIANT) {
        // Currently `ExpectEqual()` does not support the variant tensor
        // yet, so we manually cast the variant to numeric/string tensor.
        const Tensor* output = out_tensors[i].scalar<Variant>()().get<Tensor>();
        const Tensor* expected_output =
            test_case
                .expected_outputs[i + num_tensors_per_element * cur_element]
                .scalar<Variant>()()
                .get<Tensor>();
        TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
      } else {
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[i],
            test_case
                .expected_outputs[i + num_tensors_per_element * cur_element]));
      }
    }
    cur_element++;
  }
}

INSTANTIATE_TEST_SUITE_P(ListDatasetOpTest, ParameterizedGetNextTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(ListDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PlainListDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ListDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PlainListDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(ListDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<ListDatasetParams>>
DatasetOutputTypesTestCases() {
  return {
      {PlainListDatasetParams(), PlainListDatasetParams().output_dtypes()},
      {NestedListDatasetParams(), NestedListDatasetParams().output_dtypes()}};
}

DATASET_OUTPUT_DTYPES_TEST_P(ListDatasetOpTest, ListDatasetParams,
                             DatasetOutputTypesTestCases())

std::vector<DatasetOutputShapesTestCase<ListDatasetParams>>
DatasetOutputShapesTestCases() {
  return {
      {PlainListDatasetParams(), PlainListDatasetParams().output_shapes()},
      {NestedListDatasetParams(), NestedListDatasetParams().output_shapes()}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ListDatasetOpTest, ListDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ListDatasetParams>>
DatasetCardinalityTestCases() {
  return {{PlainListDatasetParams(), /*expected_cardinality=*/2},
          {NestedListDatasetParams(), /*expected_cardinality=*/2}};
}

DATASET_CARDINALITY_TEST_P(ListDatasetOpTest, ListDatasetParams,
                           DatasetCardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<ListDatasetParams>>
IteratorOutputTypesTestCases() {
  return {
      {PlainListDatasetParams(), PlainListDatasetParams().output_dtypes()},
      {NestedListDatasetParams(), NestedListDatasetParams().output_dtypes()}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(ListDatasetOpTest, ListDatasetParams,
                              IteratorOutputTypesTestCases())

std::vector<IteratorOutputShapesTestCase<ListDatasetParams>>
IteratorOutputShapesTestCases() {
  return {
      {PlainListDatasetParams(), PlainListDatasetParams().output_shapes()},
      {NestedListDatasetParams(), NestedListDatasetParams().output_shapes()}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ListDatasetOpTest, ListDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ListDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = PlainListDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ListDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ListDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/PlainListDatasetParams(),
       /*breakpoints=*/{0, 1, 2},
       /*expected_outputs=*/
       {CreateTensor<int64_t>(TensorShape({}), {1}),
        CreateTensor<int64_t>(TensorShape({2}), {1, 2}),
        CreateTensor<uint32>(TensorShape({}), {2}),
        CreateTensor<uint32>(TensorShape({2}), {2, 3}),
        CreateTensor<uint64>(TensorShape({}), {3}),
        CreateTensor<uint64>(TensorShape({2}), {3, 4}),
        CreateTensor<double>(TensorShape({1}), {37.0}),
        CreateTensor<tstring>(TensorShape({1}), {"a"}),
        CreateTensor<int64_t>(TensorShape({}), {2}),
        CreateTensor<int64_t>(TensorShape({2}), {3, 4}),
        CreateTensor<uint32>(TensorShape({}), {3}),
        CreateTensor<uint32>(TensorShape({2}), {4, 5}),
        CreateTensor<uint64>(TensorShape({}), {4}),
        CreateTensor<uint64>(TensorShape({2}), {5, 6}),
        CreateTensor<double>(TensorShape({1}), {38.0}),
        CreateTensor<tstring>(TensorShape({1}), {"b"})}},
      {/*dataset_params=*/NestedListDatasetParams(),
       /*breakpoints=*/{0, 1, 2},
       /*expected_outputs=*/
       {CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
        CreateTensor<int64_t>(TensorShape({3}), {1, 2, 3}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
        CreateTensor<int64_t>(TensorShape({3}), {4, 5, 6})}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public ListDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<ListDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));

  int cur_iteration = 0;
  bool end_of_sequence = false;

  auto params = static_cast<ListDatasetParams&>(test_case.dataset_params);
  int64_t num_elements = params.num_elements();
  size_t num_tensors_per_element = params.num_tensors_per_element();
  std::vector<Tensor> out_tensors;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      cur_iteration++;
    }

    if (breakpoint == 0) {
      EXPECT_FALSE(end_of_sequence);
    } else if (breakpoint <= num_elements) {
      for (int i = 0; i < out_tensors.size(); ++i) {
        if (out_tensors[i].dtype() == DT_VARIANT) {
          const Tensor* output =
              out_tensors[i].scalar<Variant>()().get<Tensor>();
          const Tensor* expected_output =
              test_case
                  .expected_outputs[i + num_tensors_per_element *
                                            (cur_iteration - 1)]
                  .scalar<Variant>()()
                  .get<Tensor>();
          TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
        } else {
          TF_EXPECT_OK(ExpectEqual(
              out_tensors[i],
              test_case.expected_outputs[i + num_tensors_per_element *
                                                 (cur_iteration - 1)]));
        }
      }
    } else {
      EXPECT_TRUE(end_of_sequence);
    }

    VariantTensorDataWriter writer;
    TF_ASSERT_OK(iterator_->Save(serialization_context.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader, "Iterator",
                                 *dataset_, &iterator_));
  }
}
INSTANTIATE_TEST_SUITE_P(
    ListDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(ListDatasetOpTest, SplitProvider) {
  auto params =
      ListDatasetParams({{CreateTensor<int64_t>(TensorShape({}), {6})},
                         {CreateTensor<int64_t>(TensorShape({}), {2})},
                         {CreateTensor<int64_t>(TensorShape({}), {3})},
                         {CreateTensor<int64_t>(TensorShape({}), {8})},
                         {CreateTensor<int64_t>(TensorShape({}), {7})},
                         {CreateTensor<int64_t>(TensorShape({}), {0})},
                         {CreateTensor<int64_t>(TensorShape({}), {10})}},
                        kNodeName);
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}),
                                     {{6}, {2}, {3}, {8}, {7}, {0}, {10}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/1,
      CreateTensors<int64_t>(TensorShape({}), {{2}, {7}})));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
