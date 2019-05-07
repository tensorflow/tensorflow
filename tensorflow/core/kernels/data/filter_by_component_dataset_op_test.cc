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

constexpr char kNodeName[] = "filter_by_last_component_dataset";
constexpr char kOpName[] = "FilterByLastComponentDataset";

class FilterByLastComponentDatasetOpTest : public DatasetOpsTestBase {
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

  // Creates a new `FilterByLastComponentDataset` op kernel.
  Status CreateFilterByLastComponentDatasetKernel(
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset"},
        {{"output_types", output_types}, {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `FilterByLastComponentDataset` op kernel context.
  Status CreateFilterByLastComponentDatasetContext(
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
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1: simple case.
TestCase TestCase1() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                   {0, 1, 2, 3, 4, 5}),
           DatasetOpsTestBase::CreateTensor<bool>(TensorShape{3, 1},
                                                  {true, false, true})},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {0, 1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {4, 5})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({2})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 1, 5}};
}

// Test case 2: the output of input dataset is empty.
TestCase TestCase2() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{0}, {})},
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0}};
}

// Test case 3: the output of input dataset has only one component.
TestCase TestCase3() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<bool>(TensorShape{3, 1},
                                                  {true, false, true})},
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_BOOL},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 1, 5}};
}

// Test case 4: the last component has more than one element.
TestCase InvalidLastComponentShape() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                   {0, 1, 2, 3, 4, 5}),
           DatasetOpsTestBase::CreateTensor<bool>(
               TensorShape{3, 2}, {true, false, true, true, false, true})},
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({2})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {}};
}

// Test case 5: the data type of last component is not DT_BOOL.
TestCase InvalidLastComponentDType() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3, 2},
                                                   {0, 1, 2, 3, 4, 5}),
           DatasetOpsTestBase::CreateTensor<int>(TensorShape{3}, {1, 1, 0})},
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({2})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {}};
}

class ParameterizedFilterByLastComponentDatasetOpTest
    : public FilterByLastComponentDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(
      filter_by_last_component_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(filter_by_last_component_dataset->MakeIterator(
      iterator_ctx.get(), "Iterator", &iterator));
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

TEST_F(FilterByLastComponentDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = TestCase1();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  EXPECT_EQ(filter_by_last_component_dataset->node_name(), kNodeName);
}

TEST_F(FilterByLastComponentDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = TestCase1();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  EXPECT_EQ(filter_by_last_component_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  TF_EXPECT_OK(
      VerifyTypesMatch(filter_by_last_component_dataset->output_dtypes(),
                       test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  TF_EXPECT_OK(
      VerifyShapesCompatible(filter_by_last_component_dataset->output_shapes(),
                             test_case.expected_output_shapes));
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  EXPECT_EQ(filter_by_last_component_dataset->Cardinality(),
            test_case.expected_cardinality);
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(
      filter_by_last_component_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(
      filter_by_last_component_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(filter_by_last_component_dataset->MakeIterator(
      iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(
      filter_by_last_component_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(filter_by_last_component_dataset->MakeIterator(
      iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(FilterByLastComponentDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = TestCase1();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(
      filter_by_last_component_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(filter_by_last_component_dataset->MakeIterator(
      iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::FilterByLastComponent");
}

TEST_P(ParameterizedFilterByLastComponentDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();

  std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
      test_case.expected_output_dtypes, test_case.expected_output_shapes,
      &filter_by_last_component_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
  TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
      filter_by_last_component_dataset_kernel.get(), &inputs,
      &filter_by_last_component_dataset_context));
  DatasetBase *filter_by_last_component_dataset;
  TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                             filter_by_last_component_dataset_context.get(),
                             &filter_by_last_component_dataset));
  core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(
      filter_by_last_component_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(filter_by_last_component_dataset->MakeIterator(
      iterator_ctx.get(), "Iterator", &iterator));
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
                                 *filter_by_last_component_dataset, &iterator));

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

INSTANTIATE_TEST_SUITE_P(FilterByLastComponentDatasetOpTest,
                         ParameterizedFilterByLastComponentDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

TEST_F(FilterByLastComponentDatasetOpTest, InvalidLastComponent) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  std::vector<TestCase> test_cases = {InvalidLastComponentShape(),
                                      InvalidLastComponentDType()};
  for (const TestCase &test_case : test_cases) {
    std::unique_ptr<OpKernel> filter_by_last_component_dataset_kernel;
    TF_ASSERT_OK(CreateFilterByLastComponentDatasetKernel(
        test_case.expected_output_dtypes, test_case.expected_output_shapes,
        &filter_by_last_component_dataset_kernel));

    Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
    std::vector<Tensor> inputs_for_tensor_slice_dataset =
        test_case.input_tensors;
    TF_ASSERT_OK(CreateTensorSliceDatasetTensor(
        &inputs_for_tensor_slice_dataset, &tensor_slice_dataset_tensor));
    gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
    std::unique_ptr<OpKernelContext> filter_by_last_component_dataset_context;
    TF_ASSERT_OK(CreateFilterByLastComponentDatasetContext(
        filter_by_last_component_dataset_kernel.get(), &inputs,
        &filter_by_last_component_dataset_context));
    DatasetBase *filter_by_last_component_dataset;
    TF_ASSERT_OK(CreateDataset(filter_by_last_component_dataset_kernel.get(),
                               filter_by_last_component_dataset_context.get(),
                               &filter_by_last_component_dataset));
    core::ScopedUnref scoped_unref(filter_by_last_component_dataset);

    std::unique_ptr<IteratorContext> iterator_ctx;
    TF_ASSERT_OK(CreateIteratorContext(
        filter_by_last_component_dataset_context.get(), &iterator_ctx));
    std::unique_ptr<IteratorBase> iterator;
    TF_ASSERT_OK(filter_by_last_component_dataset->MakeIterator(
        iterator_ctx.get(), "Iterator", &iterator));

    std::vector<Tensor> next;
    bool end_of_sequence = false;
    EXPECT_EQ(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence).code(),
        tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
