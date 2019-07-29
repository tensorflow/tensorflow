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
#include "tensorflow/core/kernels/data/experimental/parallel_interleave_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "parallel_interleave_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

class ParallelInterleaveDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates `TensorSliceDataset` variant tensor from the input vector of
  // tensors.
  Status CreateTensorSliceDatasetTensor(
      std::vector<Tensor>* const tensor_vector, Tensor* dataset_tensor) {
    DatasetBase* tensor_slice_dataset;
    TF_RETURN_IF_ERROR(CreateTensorSliceDataset(
        "tensor_slice_node", tensor_vector, &tensor_slice_dataset));
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(tensor_slice_dataset, dataset_tensor));
    return Status::OK();
  }

  // Creates a new `ParallelInterleaveDataset` op kernel
  Status CreateParallelInterleaveDatasetKernel(
      const FunctionDefHelper::AttrValueWrapper& func,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName,
        name_utils::OpName(ParallelInterleaveDatasetOp::kDatasetType),
        {ParallelInterleaveDatasetOp::kInputDataset,
         ParallelInterleaveDatasetOp::kCycleLength,
         ParallelInterleaveDatasetOp::kBlockLength,
         ParallelInterleaveDatasetOp::kSloppy,
         ParallelInterleaveDatasetOp::kBufferOutputElements,
         ParallelInterleaveDatasetOp::kPrefetchInputElements},
        {{ParallelInterleaveDatasetOp::kFunc, func},
         {ParallelInterleaveDatasetOp::kTarguments, {}},
         {ParallelInterleaveDatasetOp::kOutputTypes, output_types},
         {ParallelInterleaveDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `ParallelInterleaveDataset` op kernel context.
  Status CreateParallelInterleaveDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  TestCase(std::vector<Tensor> input_tensors, int64 cycle_length,
           int64 block_length, bool sloppy, int64 buffer_output_elements,
           int64 prefetch_input_elements,
           FunctionDefHelper::AttrValueWrapper func,
           std::vector<FunctionDef> func_lib,
           std::vector<Tensor> expected_outputs,
           DataTypeVector expected_output_dtypes,
           std::vector<PartialTensorShape> expected_output_shapes,
           int64 expected_cardinality, std::vector<int> breakpoints)
      : input_tensors(std::move(input_tensors)),
        cycle_length(DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}),
                                                             {cycle_length})),
        block_length(DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}),
                                                             {block_length})),
        sloppy(
            DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {sloppy})),
        buffer_output_elements(DatasetOpsTestBase::CreateTensor<int64>(
            TensorShape({}), {buffer_output_elements})),
        prefetch_input_elements(DatasetOpsTestBase::CreateTensor<int64>(
            TensorShape({}), {prefetch_input_elements})),
        func(std::move(func)),
        func_lib(std::move(func_lib)),
        expected_outputs(std::move(expected_outputs)),
        expected_output_dtypes(std::move(expected_output_dtypes)),
        expected_output_shapes(std::move(expected_output_shapes)),
        expected_cardinality(expected_cardinality),
        breakpoints(std::move(breakpoints)) {}

  std::vector<Tensor> input_tensors;
  Tensor cycle_length;
  Tensor block_length;
  Tensor sloppy;
  Tensor buffer_output_elements;
  Tensor prefetch_input_elements;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
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
  for (auto& value : values) {
    tensors.emplace_back(
        DatasetOpsTestBase::CreateTensor<T>(TensorShape({1}), {value}));
  }
  return tensors;
}

FunctionDefHelper::AttrValueWrapper MakeTensorSliceDatasetFunc(
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes) {
  return FunctionDefHelper::FunctionRef(
      /*name*/ "MakeTensorSliceDataset",
      /*attrs*/ {{TensorSliceDatasetOp::kToutputTypes, output_types},
                 {TensorSliceDatasetOp::kOutputShapes, output_shapes}});
}

// Test case 1: cycle_length = 1, block_length = 1, sloppy = false,
// buffer_output_elements = 1, prefetch_input_elements = 1
TestCase TestCase1() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/1,
          /*block_length=*/1,
          /*sloppy=*/false,
          /*buffer_output_elements=*/1,
          /*prefetch_input_elements=*/1,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// Test case 2: cycle_length = 2, block_length = 1, sloppy = false,
// buffer_output_elements = 1, prefetch_input_elements = 0
TestCase TestCase2() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/2,
          /*block_length=*/1,
          /*sloppy=*/false,
          /*buffer_output_elements=*/1,
          /*prefetch_input_elements=*/0,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 3, 1, 4, 2, 5, 6, 7, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// Test case 3: cycle_length = 3, block_length = 1, sloppy = true,
// buffer_output_elements = 3, prefetch_input_elements = 2
TestCase TestCase3() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/3,
          /*block_length=*/1,
          /*sloppy=*/true,
          /*buffer_output_elements=*/3,
          /*prefetch_input_elements=*/2,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 3, 6, 1, 4, 7, 2, 5, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// Test case 4: cycle_length = 5, block_length = 1, sloppy = true
// buffer_output_elements = 1, prefetch_input_elements = 2
TestCase TestCase4() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/5,
          /*block_length=*/1,
          /*sloppy=*/true,
          /*buffer_output_elements=*/1,
          /*prefetch_input_elements=*/2,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 3, 6, 1, 4, 7, 2, 5, 8}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// Test case 5: cycle_length = 2, block_length = 2, sloppy = false
// buffer_output_elements = 2, prefetch_input_elements = 2
TestCase TestCase5() {
  return {
      /*input_tensors=*/
      {DatasetOpsTestBase::CreateTensor<string>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*cycle_length=*/2,
      /*block_length=*/2,
      /*sloppy=*/false,
      /*buffer_output_elements=*/2,
      /*prefetch_input_elements=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*expected_outputs*/
      ConvertToTensorVec<string>({"a", "b", "d", "e", "c", "f", "g", "h", "i"}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({1})},
      /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
      /*breakpoints*/ {0, 4, 11}};
}

TestCase InvalidCycleLengthTestCase() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/0,
          /*block_length=*/1,
          /*sloppy=*/false,
          /*buffer_output_elements=*/1,
          /*prefetch_input_elements=*/1,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

TestCase InvalidBlockLengthTestCase() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/1,
          /*block_length=*/-1,
          /*sloppy=*/false,
          /*buffer_output_elements=*/1,
          /*prefetch_input_elements=*/1,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

TestCase InvalidBufferOutputElementsTestCase() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/1,
          /*block_length=*/1,
          /*sloppy=*/false,
          /*buffer_output_elements=*/0,
          /*prefetch_input_elements=*/1,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

TestCase InvalidPrefetchInputElementsTestCase() {
  return {/*input_tensors=*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*cycle_length=*/1,
          /*block_length=*/1,
          /*sloppy=*/false,
          /*buffer_output_elements=*/1,
          /*prefetch_input_elements=*/-1,
          /*func=*/
          MakeTensorSliceDatasetFunc(
              DataTypeVector({DT_INT64}),
              std::vector<PartialTensorShape>({PartialTensorShape({1})})),
          /*func_lib=*/{test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

class ParameterizedParallelInterleaveDatasetOpTest
    : public ParallelInterleaveDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(parallel_interleave_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_interleave_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(
      ExpectEqual(out_tensors, test_case.expected_outputs,
                  /*compare_order=*/!test_case.sloppy.scalar<bool>()()));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  EXPECT_EQ(parallel_interleave_dataset->node_name(), kNodeName);
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  EXPECT_EQ(parallel_interleave_dataset->type_string(),
            name_utils::OpName(ParallelInterleaveDatasetOp::kDatasetType));
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(parallel_interleave_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  TF_EXPECT_OK(
      VerifyShapesCompatible(parallel_interleave_dataset->output_shapes(),
                             test_case.expected_output_shapes));
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  EXPECT_EQ(parallel_interleave_dataset->Cardinality(),
            test_case.expected_cardinality);
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(
      parallel_interleave_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(parallel_interleave_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_interleave_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(parallel_interleave_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_interleave_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(ParallelInterleaveDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(parallel_interleave_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_interleave_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));
  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(
                ParallelInterleaveDatasetOp::kDatasetType, kIteratorPrefix));
}

TEST_P(ParameterizedParallelInterleaveDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&test_case.cycle_length),
       TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
       TensorValue(&test_case.buffer_output_elements),
       TensorValue(&test_case.prefetch_input_elements)});
  std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
  TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
      parallel_interleave_dataset_kernel.get(), &inputs,
      &parallel_interleave_dataset_context));
  DatasetBase* parallel_interleave_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_interleave_dataset_kernel.get(),
                             parallel_interleave_dataset_context.get(),
                             &parallel_interleave_dataset));
  core::ScopedUnref scoped_unref_dataset(parallel_interleave_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(parallel_interleave_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_interleave_dataset->MakeIterator(
      iterator_ctx.get(), kIteratorPrefix, &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, kIteratorPrefix,
                                 *parallel_interleave_dataset, &iterator));
    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(
      ExpectEqual(out_tensors, test_case.expected_outputs,
                  /*compare_order*/ !test_case.sloppy.scalar<bool>()()));
}

INSTANTIATE_TEST_SUITE_P(ParallelInterleaveDatasetOpTest,
                         ParameterizedParallelInterleaveDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5()})));

TEST_F(ParallelInterleaveDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));

  std::vector<TestCase> test_cases({InvalidCycleLengthTestCase(),
                                    InvalidBlockLengthTestCase(),
                                    InvalidBufferOutputElementsTestCase(),
                                    InvalidPrefetchInputElementsTestCase()});
  for (auto test_case : test_cases) {
    TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));
    std::unique_ptr<OpKernel> parallel_interleave_dataset_kernel;
    TF_ASSERT_OK(CreateParallelInterleaveDatasetKernel(
        test_case.func, test_case.expected_output_dtypes,
        test_case.expected_output_shapes, &parallel_interleave_dataset_kernel));

    Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
    std::vector<Tensor> inputs_for_tensor_slice_dataset =
        test_case.input_tensors;
    TF_ASSERT_OK(CreateTensorSliceDatasetTensor(
        &inputs_for_tensor_slice_dataset, &tensor_slice_dataset_tensor));
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&tensor_slice_dataset_tensor),
         TensorValue(&test_case.cycle_length),
         TensorValue(&test_case.block_length), TensorValue(&test_case.sloppy),
         TensorValue(&test_case.buffer_output_elements),
         TensorValue(&test_case.prefetch_input_elements)});
    std::unique_ptr<OpKernelContext> parallel_interleave_dataset_context;
    TF_ASSERT_OK(CreateParallelInterleaveDatasetContext(
        parallel_interleave_dataset_kernel.get(), &inputs,
        &parallel_interleave_dataset_context));
    DatasetBase* parallel_interleave_dataset;
    EXPECT_EQ(CreateDataset(parallel_interleave_dataset_kernel.get(),
                            parallel_interleave_dataset_context.get(),
                            &parallel_interleave_dataset)
                  .code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
