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
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tensor_slice_dataset";

class TensorSliceDatasetParams : public DatasetParams {
 public:
  TensorSliceDatasetParams(std::vector<Tensor> components, string node_name)
      : DatasetParams(TensorSliceDtypes(components),
                      TensorSliceShapes(components), std::move(node_name),
                      DatasetParamsType::TensorSlice),
        components_(std::move(components)) {}

  Status GetInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    for (auto& component : components_) {
      inputs->emplace_back(TensorValue(&component));
    }
    return Status::OK();
  }

  Status GetInputPlaceholder(
      std::vector<string>* input_placeholder) const override {
    input_placeholder->reserve(components_.size());
    for (int i = 0; i < components_.size(); ++i) {
      input_placeholder->emplace_back(
          absl::StrCat(TensorSliceDatasetOp::kComponents, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{TensorSliceDatasetOp::kToutputTypes, output_dtypes_},
                    {TensorSliceDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

 private:
  DataTypeVector TensorSliceDtypes(
      const std::vector<Tensor>& input_components) {
    DataTypeVector dtypes;
    for (const auto& component : input_components) {
      dtypes.emplace_back(component.dtype());
    }
    return dtypes;
  }

  std::vector<PartialTensorShape> TensorSliceShapes(
      const std::vector<Tensor>& input_components) {
    std::vector<PartialTensorShape> shapes;
    for (const auto& component : input_components) {
      gtl::InlinedVector<int64, 4> partial_dim_sizes;
      for (int i = 1; i < component.dims(); ++i) {
        partial_dim_sizes.push_back(component.dim_size(i));
      }
      shapes.emplace_back(std::move(partial_dim_sizes));
    }
    return shapes;
  }

 public:
  std::vector<Tensor> components_;
};

class TensorSliceDatasetOpTest : public DatasetOpsTestBaseV2 {};

TensorSliceDatasetParams PlainTensorSliceDatasetParams() {
  std::vector<Tensor> components = {
      CreateTensor<int64>(TensorShape({2}), {1, 2}),
      CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4}),
      CreateTensor<uint32>(TensorShape({2}), {2, 3}),
      CreateTensor<uint32>(TensorShape({2, 2}), {2, 3, 4, 5}),
      CreateTensor<uint64>(TensorShape({2}), {3, 4}),
      CreateTensor<uint64>(TensorShape({2, 2}), {3, 4, 5, 6}),
      CreateTensor<double>(TensorShape({2, 1}), {37.0, 38.0}),
      CreateTensor<tstring>(TensorShape({2, 1}), {"a", "b"})};

  return {std::move(components), kNodeName};
}

TensorSliceDatasetParams NestedTensorSliceDatasetParams() {
  std::vector<Tensor> components = {
      CreateTensor<Variant>(
          TensorShape({2, 1}),
          {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0}),
           CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
      CreateTensor<Variant>(
          TensorShape({2, 1}),
          {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"}),
           CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
      CreateTensor<int64>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6})};

  return {std::move(components), kNodeName};
}

std::vector<GetNextTestCase<TensorSliceDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/PlainTensorSliceDatasetParams(),
       /*expected_outputs=*/{CreateTensor<int64>(TensorShape({}), {1}),
                             CreateTensor<int64>(TensorShape({2}), {1, 2}),
                             CreateTensor<uint32>(TensorShape({}), {2}),
                             CreateTensor<uint32>(TensorShape({2}), {2, 3}),
                             CreateTensor<uint64>(TensorShape({}), {3}),
                             CreateTensor<uint64>(TensorShape({2}), {3, 4}),
                             CreateTensor<double>(TensorShape({1}), {37.0}),
                             CreateTensor<tstring>(TensorShape({1}), {"a"}),
                             CreateTensor<int64>(TensorShape({}), {2}),
                             CreateTensor<int64>(TensorShape({2}), {3, 4}),
                             CreateTensor<uint32>(TensorShape({}), {3}),
                             CreateTensor<uint32>(TensorShape({2}), {4, 5}),
                             CreateTensor<uint64>(TensorShape({}), {4}),
                             CreateTensor<uint64>(TensorShape({2}), {5, 6}),
                             CreateTensor<double>(TensorShape({1}), {38.0}),
                             CreateTensor<tstring>(TensorShape({1}), {"b"})}},
      {/*dataset_params=*/NestedTensorSliceDatasetParams(),
       /*expected_outputs=*/
       {CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
        CreateTensor<int64>(TensorShape({3}), {1, 2, 3}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
        CreateTensor<int64>(TensorShape({3}), {4, 5, 6})}}};
}

class ParameterizedGetNextTest
    : public TensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<TensorSliceDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::vector<string> input_placeholder;
  TF_ASSERT_OK(
      test_case.dataset_params.GetInputPlaceholder(&input_placeholder));
  size_t num_tensors_per_slice = input_placeholder.size();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_slice = 0;

  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
    for (int i = 0; i < out_tensors.size(); ++i) {
      EXPECT_LT(i + num_tensors_per_slice * cur_slice,
                test_case.expected_outputs.size());
      if (out_tensors[i].dtype() == DT_VARIANT) {
        // Currently `ExpectEqual()` does not support the variant tensor
        // yet, so we manually cast the variant to numeric/string tensor.
        const Tensor* output = out_tensors[i].scalar<Variant>()().get<Tensor>();
        const Tensor* expected_output =
            test_case.expected_outputs[i + num_tensors_per_slice * cur_slice]
                .scalar<Variant>()()
                .get<Tensor>();
        TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
      } else {
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[i],
            test_case.expected_outputs[i + num_tensors_per_slice * cur_slice]));
      }
    }
    out_tensors.clear();
    cur_slice++;
  }
}

INSTANTIATE_TEST_SUITE_P(TensorSliceDatasetOpTest, ParameterizedGetNextTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(TensorSliceDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PlainTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TensorSliceDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PlainTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(TensorSliceDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<TensorSliceDatasetParams>>
DatasetOutputTypesTestCases() {
  return {{PlainTensorSliceDatasetParams(),
           PlainTensorSliceDatasetParams().output_dtypes()},
          {NestedTensorSliceDatasetParams(),
           NestedTensorSliceDatasetParams().output_dtypes()}};
}

DATASET_OUTPUT_DTYPES_TEST_P(TensorSliceDatasetOpTest, TensorSliceDatasetParams,
                             DatasetOutputTypesTestCases())

std::vector<DatasetOutputShapesTestCase<TensorSliceDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{PlainTensorSliceDatasetParams(),
           PlainTensorSliceDatasetParams().output_shapes()},
          {NestedTensorSliceDatasetParams(),
           NestedTensorSliceDatasetParams().output_shapes()}};
}

DATASET_OUTPUT_SHAPES_TEST_P(TensorSliceDatasetOpTest, TensorSliceDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<TensorSliceDatasetParams>>
DatasetCardinalityTestCases() {
  return {{PlainTensorSliceDatasetParams(), /*expected_cardinality=*/2},
          {NestedTensorSliceDatasetParams(), /*expected_cardinality=*/2}};
}

DATASET_CARDINALITY_TEST_P(TensorSliceDatasetOpTest, TensorSliceDatasetParams,
                           DatasetCardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<TensorSliceDatasetParams>>
IteratorOutputTypesTestCases() {
  return {{PlainTensorSliceDatasetParams(),
           PlainTensorSliceDatasetParams().output_dtypes()},
          {NestedTensorSliceDatasetParams(),
           NestedTensorSliceDatasetParams().output_dtypes()}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(TensorSliceDatasetOpTest,
                              TensorSliceDatasetParams,
                              IteratorOutputTypesTestCases())

std::vector<IteratorOutputShapesTestCase<TensorSliceDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{PlainTensorSliceDatasetParams(),
           PlainTensorSliceDatasetParams().output_shapes()},
          {NestedTensorSliceDatasetParams(),
           NestedTensorSliceDatasetParams().output_shapes()}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(TensorSliceDatasetOpTest,
                              TensorSliceDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(TensorSliceDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = PlainTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      TensorSliceDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<TensorSliceDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/PlainTensorSliceDatasetParams(),
       /*breakpoints=*/{0, 1, 2},
       /*expected_outputs=*/
       {CreateTensor<int64>(TensorShape({}), {1}),
        CreateTensor<int64>(TensorShape({2}), {1, 2}),
        CreateTensor<uint32>(TensorShape({}), {2}),
        CreateTensor<uint32>(TensorShape({2}), {2, 3}),
        CreateTensor<uint64>(TensorShape({}), {3}),
        CreateTensor<uint64>(TensorShape({2}), {3, 4}),
        CreateTensor<double>(TensorShape({1}), {37.0}),
        CreateTensor<tstring>(TensorShape({1}), {"a"}),
        CreateTensor<int64>(TensorShape({}), {2}),
        CreateTensor<int64>(TensorShape({2}), {3, 4}),
        CreateTensor<uint32>(TensorShape({}), {3}),
        CreateTensor<uint32>(TensorShape({2}), {4, 5}),
        CreateTensor<uint64>(TensorShape({}), {4}),
        CreateTensor<uint64>(TensorShape({2}), {5, 6}),
        CreateTensor<double>(TensorShape({1}), {38.0}),
        CreateTensor<tstring>(TensorShape({1}), {"b"})}},
      {/*dataset_params=*/NestedTensorSliceDatasetParams(),
       /*breakpoints=*/{0, 1, 2},
       /*expected_outputs=*/
       {CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
        CreateTensor<int64>(TensorShape({3}), {1, 2, 3}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<double>(TensorShape({2, 2}), {5.0, 6.0, 7.0, 8.0})}),
        CreateTensor<Variant>(
            TensorShape({1}),
            {CreateTensor<tstring>(TensorShape({1, 2}), {"c", "d"})}),
        CreateTensor<int64>(TensorShape({3}), {4, 5, 6})}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public TensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<TensorSliceDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));

  int cur_iteration = 0;
  bool end_of_sequence = false;

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.GetInputs(&inputs));
  int64 num_slices = inputs[0].tensor->dim_size(0);
  size_t num_tensors_per_slice = inputs.size();
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
    } else if (breakpoint <= num_slices) {
      for (int i = 0; i < out_tensors.size(); ++i) {
        if (out_tensors[i].dtype() == DT_VARIANT) {
          const Tensor* output =
              out_tensors[i].scalar<Variant>()().get<Tensor>();
          const Tensor* expected_output =
              test_case
                  .expected_outputs[i +
                                    num_tensors_per_slice * (cur_iteration - 1)]
                  .scalar<Variant>()()
                  .get<Tensor>();
          TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
        } else {
          TF_EXPECT_OK(ExpectEqual(
              out_tensors[i],
              test_case.expected_outputs[i + num_tensors_per_slice *
                                                 (cur_iteration - 1)]));
        }
      }
    } else {
      EXPECT_TRUE(end_of_sequence);
    }

    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_ASSERT_OK(iterator_->Save(serialization_context.get(), &writer));
    TF_ASSERT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader, "Iterator",
                                 *dataset_, &iterator_));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TensorSliceDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
