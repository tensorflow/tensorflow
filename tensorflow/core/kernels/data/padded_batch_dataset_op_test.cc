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
#include "tensorflow/core/kernels/data/padded_batch_dataset_op.h"

#include "tensorflow/core/kernels/data/concatenate_dataset_op.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "padded_batch_dataset";
constexpr int kOpVersion = 2;

class PaddedBatchDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates `ConcatenateDataset` variant tensor from the input vector of
  // tensor vectors.
  Status CreateConcatenateDatasetTensor(
      const std::vector<std::vector<Tensor>> &tensor_vectors,
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      Tensor *concatenate_dataset_tensor) {
    // Create two `TensorSliceDataset` tensors as the inputs for
    // `ConcatenateDataset`.
    std::vector<Tensor> tensor_slice_dataset_tensors;
    for (int i = 0; i < tensor_vectors.size(); ++i) {
      std::vector<Tensor> tensors = tensor_vectors[i];
      DatasetBase *tensor_slice_dataset;
      TF_RETURN_IF_ERROR(
          CreateTensorSliceDataset(strings::StrCat("tensor_slice_node_", i),
                                   &tensors, &tensor_slice_dataset));
      Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
      TF_RETURN_IF_ERROR(
          StoreDatasetInVariantTensor(tensor_slice_dataset, &dataset_tensor));
      tensor_slice_dataset_tensors.emplace_back(std::move(dataset_tensor));
    }

    // Create a `ConcatenateDataset` dataset.
    std::unique_ptr<OpKernel> concatenate_dataset_op_kernel;
    NodeDef concatenate_node_def = test::function::NDef(
        "concatenate_dataset",
        name_utils::OpName(ConcatenateDatasetOp::kDatasetType),
        {ConcatenateDatasetOp::kInputDataset,
         ConcatenateDatasetOp::kAnotherDataset},
        {{ConcatenateDatasetOp::kOutputTypes, {output_types}},
         {ConcatenateDatasetOp::kOutputShapes, {output_shapes}}});
    TF_RETURN_IF_ERROR(
        CreateOpKernel(concatenate_node_def, &concatenate_dataset_op_kernel));

    gtl::InlinedVector<TensorValue, 4> concatenate_dataset_inputs;
    for (auto &tensor : tensor_slice_dataset_tensors) {
      concatenate_dataset_inputs.emplace_back(&tensor);
    }

    std::unique_ptr<OpKernelContext> concatenate_dataset_op_context;
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*concatenate_dataset_op_kernel,
                                          concatenate_dataset_inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(
        concatenate_dataset_op_kernel.get(), &concatenate_dataset_inputs,
        &concatenate_dataset_op_context));
    DatasetBase *concatenate_dataset;
    TF_RETURN_IF_ERROR(CreateDataset(concatenate_dataset_op_kernel.get(),
                                     concatenate_dataset_op_context.get(),
                                     &concatenate_dataset));

    // Store the `ConcatenateDataset` dataset in a tensor.
    TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(concatenate_dataset,
                                                   concatenate_dataset_tensor));
    return Status::OK();
  }

  // Creates a new `PaddedBatchDataset` op kernel
  Status CreatePaddedBatchDatasetKernel(
      bool parallel_copy, int n, const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    std::vector<string> inputs({PaddedBatchDatasetOp::kInputDataset,
                                PaddedBatchDatasetOp::kBatchSize});
    // Create the placeholder names for the input padded_shapes.
    for (int i = 0; i < n; ++i) {
      inputs.emplace_back(
          strings::StrCat(PaddedBatchDatasetOp::kPaddedShapes, "_", i));
    }
    // Create the placeholder names for the input padding_values.
    for (int j = 0; j < output_types.size(); ++j) {
      inputs.emplace_back(
          strings::StrCat(PaddedBatchDatasetOp::kPaddingValues, "_", j));
    }
    inputs.push_back(PaddedBatchDatasetOp::kDropRemainder);

    name_utils::OpNameParams params;
    params.op_version = kOpVersion;
    NodeDef node_def = test::function::NDef(
        kNodeName,
        name_utils::OpName(PaddedBatchDatasetOp::kDatasetType, params), inputs,
        {{PaddedBatchDatasetOp::kParallelCopy, parallel_copy},
         {PaddedBatchDatasetOp::kToutputTypes, output_types},
         {PaddedBatchDatasetOp::kOutputShapes, output_shapes},
         {PaddedBatchDatasetOp::kNumPaddedShapes, n}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `PaddedBatchDataset` op kernel context.
  Status CreatePaddedBatchDatasetContext(
      OpKernel *const op_kernel,
      gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  // Used for creating two `TensorSliceDataset` datasets, which will be the
  // input datasets for `ConcatenateDataset`. Then the `ConcatenateDataset`
  // dataset will be the input for `PaddedBatchDataset`.
  std::vector<std::vector<Tensor>> input_tensors;
  DataTypeVector concatenate_output_dtypes;
  std::vector<PartialTensorShape> concatenate_output_shapes;
  Tensor batch_size;
  std::vector<Tensor> padded_shapes;
  std::vector<Tensor> padding_values;
  Tensor drop_remainder;
  bool parallel_copy;
  int64 n;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

template <typename T>
std::vector<Tensor> ConvertToTensorVec(std::vector<T> values) {
  std::vector<Tensor> tensors;
  tensors.reserve(values.size());
  for (auto &value : values) {
    tensors.emplace_back(
        DatasetOpsTestBase::CreateTensor<T>(TensorShape({1}), {value}));
  }
  return tensors;
}

// Test case 1: input elements with same shapes.
TestCase TestCase1() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(
               TensorShape{4, 2}, {6, 7, 8, 9, 10, 11, 12, 13})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {true}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                   {0, 1, 1, 2, 3, 1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                   {4, 5, 1, 6, 7, 1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                   {8, 9, 1, 10, 11, 1})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({2, 3})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 2: input elements with different shapes.
TestCase TestCase2() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{4, 1},
                                                    {6, 7, 8, 9})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({-1})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {true}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                   {0, 1, 1, 2, 3, 1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                   {4, 5, 1, 6, 1, 1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                   {7, 1, 1, 8, 1, 1})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({2, 3})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 3: similar with the test case 2 but drop_remainder = false.
TestCase TestCase3() {
  return {
      /*input_tensors*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                {0, 1, 2, 3, 4, 5})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{4, 1},
                                                {6, 7, 8, 9})}},
      /*concatenate_output_dtypes*/ {DT_INT64},
      /*concatenate_output_shapes*/ {PartialTensorShape({-1})},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
      /*padded_shapes*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
      /*padding_values*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
      /*parallel_copy*/ false,
      /*n*/ 1,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {0, 1, 1, 2, 3, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {4, 5, 1, 6, 1, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {7, 1, 1, 8, 1, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1, 3}, {9, 1, 1})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({-1, 3})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 2, 5}};
}

// Test case 4: similar with the test case 3 but the input elements can be
// divided by the batch size evenly. As drop_remainder = false, the output
// shape is still {-1, 3} instead of {2, 3}.
TestCase TestCase4() {
  return {
      /*input_tensors*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                {0, 1, 2, 3, 4, 5})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 1}, {6, 7, 8})}},
      /*concatenate_output_dtypes*/ {DT_INT64},
      /*concatenate_output_shapes*/ {PartialTensorShape({-1})},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
      /*padded_shapes*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
      /*padding_values*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
      /*parallel_copy*/ false,
      /*n*/ 1,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {0, 1, 1, 2, 3, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {4, 5, 1, 6, 1, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {7, 1, 1, 8, 1, 1})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({-1, 3})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 2, 5}};
}

// Test case 5: similar with the test case 3 but padded_shapes = {-1}.
TestCase TestCase5() {
  return {
      /*input_tensors*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                {0, 1, 2, 3, 4, 5})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{4, 1},
                                                {6, 7, 8, 9})}},
      /*concatenate_output_dtypes*/ {DT_INT64},
      /*concatenate_output_shapes*/ {PartialTensorShape({-1})},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
      /*padded_shapes*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {-1})},
      /*padding_values*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
      /*parallel_copy*/ false,
      /*n*/ 1,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {0, 1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {4, 5, 6, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 1}, {7, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1, 1}, {9})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 2, 5}};
}

// Test case 6: similar with the test case 5 but parallel_copy = true.
TestCase TestCase6() {
  return {
      /*input_tensors*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                {0, 1, 2, 3, 4, 5})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{4, 1},
                                                {6, 7, 8, 9})}},
      /*concatenate_output_dtypes*/ {DT_INT64},
      /*concatenate_output_shapes*/ {PartialTensorShape({-1})},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
      /*padded_shapes*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {-1})},
      /*padding_values*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
      /*parallel_copy*/ true,
      /*n*/ 1,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {0, 1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {4, 5, 6, 1}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 1}, {7, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1, 1}, {9})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 2, 5}};
}

// Test case 7: empty input elements.
TestCase TestCase7() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{0}, {})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{0}, {})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({-1})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {-1})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase ShortPaddingTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase InvalidPaddingShapesTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {1, 2})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase InvalidBatchSizeTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {-1}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase InvalidPaddedShapesSizeTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 2,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase InvalidPaddedValuesSizeTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64, DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase InvalidPaddedValuesDTypeTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<string>(TensorShape{}, {"a"})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase InvalidPaddedValuesShapeTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {0, 1, 2, 3, 4, 5})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                    {6, 7, 8, 9, 10, 11})}},
          /*concatenate_output_dtypes*/ {DT_INT64},
          /*concatenate_output_shapes*/ {PartialTensorShape({2})},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape{}, {2}),
          /*padded_shapes*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*padding_values*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1})},
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape{}, {false}),
          /*parallel_copy*/ true,
          /*n*/ 1,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1, -1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5}};
}

class ParameterizedPaddedBatchDatasetOpTest
    : public PaddedBatchDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedPaddedBatchDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

TEST_F(PaddedBatchDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  EXPECT_EQ(padded_batch_dataset->node_name(), kNodeName);
}

TEST_F(PaddedBatchDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  EXPECT_EQ(padded_batch_dataset->type_string(),
            name_utils::OpName(PaddedBatchDatasetOp::kDatasetType, params));
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(padded_batch_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(padded_batch_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  EXPECT_EQ(padded_batch_dataset->Cardinality(),
            test_case.expected_cardinality);
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(padded_batch_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(PaddedBatchDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));
  name_utils::IteratorPrefixParams params;
  params.op_version = kOpVersion;
  params.prefix = "Iterator";
  EXPECT_EQ(
      iterator->prefix(),
      name_utils::IteratorPrefix(PaddedBatchDatasetOp::kDatasetType, params));
}

TEST_P(ParameterizedPaddedBatchDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int> &breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, "Iterator",
                                 *padded_batch_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

INSTANTIATE_TEST_SUITE_P(PaddedBatchDatasetOpTest,
                         ParameterizedPaddedBatchDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6(),
                              TestCase7()})));

TEST_F(PaddedBatchDatasetOpTest, ShortPadding) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = ShortPaddingTestCase();
  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::DATA_LOSS);
}

TEST_F(PaddedBatchDatasetOpTest, InvalidPaddedShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = InvalidPaddingShapesTestCase();
  std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
  TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
      test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &padded_batch_dataset_kernel));

  Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(CreateConcatenateDatasetTensor(
      test_case.input_tensors, test_case.concatenate_output_dtypes,
      test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
  Tensor batch_size = test_case.batch_size;
  std::vector<Tensor> padded_shapes = test_case.padded_shapes;
  std::vector<Tensor> padding_values = test_case.padding_values;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
  for (auto &padded_shape : padded_shapes) {
    inputs.emplace_back(&padded_shape);
  }
  for (auto &padding_value : padding_values) {
    inputs.emplace_back(&padding_value);
  }
  inputs.emplace_back(&drop_remainder);

  std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
  TF_ASSERT_OK(
      CreatePaddedBatchDatasetContext(padded_batch_dataset_kernel.get(),
                                      &inputs, &padded_batch_dataset_context));
  DatasetBase *padded_batch_dataset;
  TF_ASSERT_OK(CreateDataset(padded_batch_dataset_kernel.get(),
                             padded_batch_dataset_context.get(),
                             &padded_batch_dataset));
  core::ScopedUnref scoped_unref(padded_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(padded_batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(padded_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(PaddedBatchDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<TestCase> test_cases = {
      InvalidBatchSizeTestCase(), InvalidPaddedShapesSizeTestCase(),
      InvalidPaddedValuesSizeTestCase(), InvalidPaddedValuesDTypeTestCase(),
      InvalidPaddedValuesShapeTestCase()};
  for (const TestCase &test_case : test_cases) {
    std::unique_ptr<OpKernel> padded_batch_dataset_kernel;
    TF_ASSERT_OK(CreatePaddedBatchDatasetKernel(
        test_case.parallel_copy, test_case.n, test_case.expected_output_dtypes,
        test_case.expected_output_shapes, &padded_batch_dataset_kernel));

    Tensor concatenate_dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_ASSERT_OK(CreateConcatenateDatasetTensor(
        test_case.input_tensors, test_case.concatenate_output_dtypes,
        test_case.concatenate_output_shapes, &concatenate_dataset_tensor));
    Tensor batch_size = test_case.batch_size;
    std::vector<Tensor> padded_shapes = test_case.padded_shapes;
    std::vector<Tensor> padding_values = test_case.padding_values;
    Tensor drop_remainder = test_case.drop_remainder;
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&concatenate_dataset_tensor), TensorValue(&batch_size)});
    for (auto &padded_shape : padded_shapes) {
      inputs.emplace_back(&padded_shape);
    }
    for (auto &padding_value : padding_values) {
      inputs.emplace_back(&padding_value);
    }
    inputs.emplace_back(&drop_remainder);

    std::unique_ptr<OpKernelContext> padded_batch_dataset_context;
    TF_ASSERT_OK(CreatePaddedBatchDatasetContext(
        padded_batch_dataset_kernel.get(), &inputs,
        &padded_batch_dataset_context));
    DatasetBase *padded_batch_dataset;
    EXPECT_EQ(
        CreateDataset(padded_batch_dataset_kernel.get(),
                      padded_batch_dataset_context.get(), &padded_batch_dataset)
            .code(),
        tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
