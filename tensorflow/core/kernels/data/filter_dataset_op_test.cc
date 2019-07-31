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
#include "tensorflow/core/kernels/data/filter_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "filter_dataset";

class FilterDatasetOpTest : public DatasetOpsTestBase {
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

  // Creates a new `FilterDataset` op kernel
  Status CreateFilterDatasetKernel(
      const FunctionDefHelper::AttrValueWrapper &func,
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(FilterDatasetOp::kDatasetType),
        {FilterDatasetOp::kInputDataset},
        {{FilterDatasetOp::kPredicate, func},
         {FilterDatasetOp::kTarguments, {}},
         {FilterDatasetOp::kOutputTypes, output_types},
         {FilterDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `FilterDataset` op kernel context.
  Status CreateFilterDatasetContext(
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
    tensors.emplace_back(CreateTensor<T>(TensorShape({1}), {value}));
  }
  return tensors;
}

// Test case 1: norm case.
TestCase TestCase1() {
  return {/*input_tensors*/
          {CreateTensor<int64>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
          /*func*/ FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::IsZero()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 0, 0}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 2: the input dataset has no outputs.
TestCase TestCase2() {
  return {/*input_tensors*/
          {CreateTensor<int64>(TensorShape{0}, {})},
          /*func*/ FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::IsZero()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 3: the filter function returns two outputs.
TestCase InvalidFuncTestCase1() {
  return {/*input_tensors*/
          {CreateTensor<int64>(TensorShape{3, 3}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
          /*func*/
          FunctionDefHelper::FunctionRef(
              "GetUnique", {{"T", DT_INT64}, {"out_idx", DT_INT32}}),
          /*func_lib*/ {test::function::Unique()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({3, 1})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {}};
}

// Test case 4: the filter function returns a 1-D bool tensor.
TestCase InvalidFuncTestCase2() {
  return {
      /*input_tensors*/
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*func*/ FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*expected_outputs*/
      ConvertToTensorVec<int64>({}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({3, 1})},
      /*expected_cardinality*/ kUnknownCardinality,
      /*breakpoints*/ {}};
}

// Test case 5: the filter function returns a scalar int64 tensor.
TestCase InvalidFuncTestCase3() {
  return {/*input_tensors*/
          {CreateTensor<int64>(TensorShape{9}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
          /*func*/ FunctionDefHelper::FunctionRef("NonZero", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::NonZero()},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({}),
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {}};
}

class ParameterizedFilterDatasetOpTest
    : public FilterDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedFilterDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(filter_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      filter_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
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

TEST_F(FilterDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  EXPECT_EQ(filter_dataset->node_name(), kNodeName);
}

TEST_F(FilterDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  EXPECT_EQ(filter_dataset->type_string(),
            name_utils::OpName(FilterDatasetOp::kDatasetType));
}

TEST_P(ParameterizedFilterDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(filter_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFilterDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(filter_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedFilterDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  EXPECT_EQ(filter_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedFilterDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(filter_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedFilterDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(filter_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      filter_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFilterDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(filter_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      filter_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(ParameterizedFilterDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(filter_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      filter_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), name_utils::IteratorPrefix(
                                    FilterDatasetOp::kDatasetType, "Iterator"));
}

TEST_P(ParameterizedFilterDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> filter_dataset_kernel;
  TF_ASSERT_OK(CreateFilterDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &filter_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&tensor_slice_dataset_tensor)});
  std::unique_ptr<OpKernelContext> filter_dataset_context;
  TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(), &inputs,
                                          &filter_dataset_context));
  DatasetBase *filter_dataset;
  TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                             filter_dataset_context.get(), &filter_dataset));
  core::ScopedUnref scoped_unref(filter_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(filter_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      filter_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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
                                 *filter_dataset, &iterator));

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

INSTANTIATE_TEST_SUITE_P(
    FilterDatasetOpTest, ParameterizedFilterDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestCase>({TestCase1(), TestCase2()})));

TEST_F(ParameterizedFilterDatasetOpTest, InvalidFuncs) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(
      {test::function::IsZero(), test::function::Unique(),
       test::function::NonZero()},
      cpu_num));

  std::vector<TestCase> test_cases(
      {InvalidFuncTestCase1(), InvalidFuncTestCase2(), InvalidFuncTestCase3()});
  for (const auto &test_case : test_cases) {
    std::unique_ptr<OpKernel> filter_dataset_kernel;
    TF_ASSERT_OK(CreateFilterDatasetKernel(
        test_case.func, test_case.expected_output_dtypes,
        test_case.expected_output_shapes, &filter_dataset_kernel));
    Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
    std::vector<Tensor> inputs_for_tensor_slice_dataset =
        test_case.input_tensors;
    TF_ASSERT_OK(CreateTensorSliceDatasetTensor(
        &inputs_for_tensor_slice_dataset, &tensor_slice_dataset_tensor));
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&tensor_slice_dataset_tensor)});
    std::unique_ptr<OpKernelContext> filter_dataset_context;
    TF_ASSERT_OK(CreateFilterDatasetContext(filter_dataset_kernel.get(),
                                            &inputs, &filter_dataset_context));
    DatasetBase *filter_dataset;
    TF_ASSERT_OK(CreateDataset(filter_dataset_kernel.get(),
                               filter_dataset_context.get(), &filter_dataset));
    core::ScopedUnref scoped_unref(filter_dataset);

    std::unique_ptr<IteratorContext> iterator_ctx;
    TF_ASSERT_OK(
        CreateIteratorContext(filter_dataset_context.get(), &iterator_ctx));
    std::unique_ptr<IteratorBase> iterator;
    TF_ASSERT_OK(filter_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                              &iterator));

    bool end_of_sequence = false;
    std::vector<Tensor> out_tensors;
    EXPECT_EQ(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence)
            .code(),
        tensorflow::error::INVALID_ARGUMENT);
    EXPECT_TRUE(out_tensors.empty());
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
