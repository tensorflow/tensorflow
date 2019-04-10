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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "interleave_dataset";
constexpr char kOpName[] = "InterleaveDataset";

class InterleaveDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates `TensorSliceDataset` variant tensor from the input vector of
  // tensors.
  Status CreateTensorSliceDatasetTensor(
      std::vector<Tensor> *const tensor_vector, Tensor *dataset_tensor) {
    DatasetBase *tensor_slice_dataset;
    TF_RETURN_IF_ERROR(CreateTensorSliceDataset(
        "tensor_slice_node", tensor_vector, &tensor_slice_dataset));
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(tensor_slice_dataset, dataset_tensor));
    return Status::OK();
  }

  // Creates a new `InterleaveDataset` op kernel
  Status CreateInterleaveDatasetKernel(
      const FunctionDefHelper::AttrValueWrapper &func,
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "cycle_length", "block_length"},
        {{"f", func},
         {"Targuments", {}},
         {"output_types", output_types},
         {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `InterleaveDataset` op kernel context.
  Status CreateInterleaveDatasetContext(
      OpKernel *const op_kernel,
      gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<Tensor> input_tensors;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  Tensor cycle_length;
  Tensor block_length;
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

FunctionDefHelper::AttrValueWrapper MakeTensorSliceDatasetFunc(
    const DataTypeVector &output_types,
    const std::vector<PartialTensorShape> &output_shapes) {
  return FunctionDefHelper::FunctionRef(
      /*name*/ "MakeTensorSliceDataset",
      /*attrs*/ {{"Toutput_types", output_types},
                 {"output_shapes", output_shapes}});
}

// test case 1: cycle_length = 1, block_length = 1.
TestCase TestCase1() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*cycle_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*block_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// test case 2: cycle_length = 2, block_length = 1.
TestCase TestCase2() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*cycle_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
          /*block_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 3, 1, 4, 2, 5, 6, 7, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// test case 3: cycle_length = 3, block_length = 1.
TestCase TestCase3() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*cycle_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
          /*block_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 3, 6, 1, 4, 7, 2, 5, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// test case 4: cycle_length = 5, block_length = 1.
TestCase TestCase4() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*cycle_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
          /*block_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 3, 6, 1, 4, 7, 2, 5, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// test case 5: cycle_length = 2, block_length = 2.
TestCase TestCase5() {
  return {
      /*input_tensors*/
      {DatasetOpsTestBase::CreateTensor<string>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*func*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib*/ {test::function::MakeTensorSliceDataset()},
      /*cycle_length*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*block_length*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*expected_outputs*/
      ConvertToTensorVec<string>({"a", "b", "d", "e", "c", "f", "g", "h", "i"}),
      /*expected_output_dtypes*/ {DT_STRING},
      /*expected_output_shapes*/ {PartialTensorShape({1})},
      /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
      /*breakpoints*/ {0, 4, 11}};
}

// test case 6: cycle_length = 2, block_length = 3.
TestCase TestCase6() {
  return {
      /*input_tensors*/
      {DatasetOpsTestBase::CreateTensor<string>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*func*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib*/ {test::function::MakeTensorSliceDataset()},
      /*cycle_length*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*block_length*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*expected_outputs*/
      ConvertToTensorVec<string>({"a", "b", "c", "d", "e", "f", "g", "h", "i"}),
      /*expected_output_dtypes*/ {DT_STRING},
      /*expected_output_shapes*/ {PartialTensorShape({1})},
      /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
      /*breakpoints*/ {0, 4, 11}};
}

// test case 7: cycle_length = 2, block_length = 5.
TestCase TestCase7() {
  return {
      /*input_tensors*/
      {DatasetOpsTestBase::CreateTensor<string>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*func*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib*/ {test::function::MakeTensorSliceDataset()},
      /*cycle_length*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*block_length*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {5}),
      /*expected_outputs*/
      ConvertToTensorVec<string>({"a", "b", "c", "d", "e", "f", "g", "h", "i"}),
      /*expected_output_dtypes*/ {DT_STRING},
      /*expected_output_shapes*/ {PartialTensorShape({1})},
      /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
      /*breakpoints*/ {0, 4, 11}};
}

// test case 8: cycle_length = 0, block_length = 5.
TestCase InvalidCycleLengthTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*cycle_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
          /*block_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {5}),
          /*expected_outputs*/ ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {}};
}

// test case 9: cycle_length = 1, block_length = -1.
TestCase InvalidBlockLengthTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*cycle_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*block_length*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-1}),
          /*expected_outputs*/ ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {}};
}

class ParameterizedInterleaveDatasetOpTest
    : public InterleaveDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedInterleaveDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(interleave_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(interleave_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                &iterator));
  auto expected_outputs_it = test_case.expected_outputs.begin();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      for (const auto &tensor : out_tensors) {
        EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

TEST_F(InterleaveDatasetOpTest, InvalidCycleLength) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = InvalidCycleLengthTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  EXPECT_EQ(CreateDataset(interleave_dataset_kernel.get(),
                          interleave_dataset_context.get(), &interleave_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(InterleaveDatasetOpTest, InvalidBlockLength) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = InvalidBlockLengthTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  EXPECT_EQ(CreateDataset(interleave_dataset_kernel.get(),
                          interleave_dataset_context.get(), &interleave_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(InterleaveDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  EXPECT_EQ(interleave_dataset->node_name(), kNodeName);
}

TEST_F(InterleaveDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  EXPECT_EQ(interleave_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedInterleaveDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(interleave_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedInterleaveDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(interleave_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedInterleaveDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  EXPECT_EQ(interleave_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedInterleaveDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(interleave_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedInterleaveDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(interleave_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(interleave_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedInterleaveDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(interleave_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(interleave_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(InterleaveDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(interleave_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(interleave_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Interleave");
}

TEST_P(ParameterizedInterleaveDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> interleave_dataset_kernel;
  TF_ASSERT_OK(CreateInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor cycle_length = test_case.cycle_length;
  Tensor block_length = test_case.block_length;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&tensor_slice_dataset_tensor, &cycle_length, &block_length});
  std::unique_ptr<OpKernelContext> interleave_dataset_context;
  TF_ASSERT_OK(CreateInterleaveDatasetContext(
      interleave_dataset_kernel.get(), &inputs, &interleave_dataset_context));
  DatasetBase *interleave_dataset;
  TF_ASSERT_OK(CreateDataset(interleave_dataset_kernel.get(),
                             interleave_dataset_context.get(),
                             &interleave_dataset));
  core::ScopedUnref scoped_unref(interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(interleave_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(interleave_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  const std::vector<int> &breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator->Restore(iterator_ctx.get(), &reader));

    while (cur_iteration <= breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        for (auto &tensor : out_tensors) {
          EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
          TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
          expected_outputs_it++;
        }
      }
      cur_iteration++;
    }

    if (breakpoint >= test_case.expected_outputs.size()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(InterleaveDatasetOpTest,
                         ParameterizedInterleaveDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6(),
                              TestCase7()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
