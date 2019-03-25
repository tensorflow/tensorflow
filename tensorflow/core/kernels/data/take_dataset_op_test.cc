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

constexpr char kNodeName[] = "take_dataset";
constexpr char kOpName[] = "TakeDataset";

class TakeDatasetOpTest : public DatasetOpsTestBase {
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

  // Create a new `TakeDataset` op kernel.
  Status CreateTakeDatasetKernel(
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "count"},
        {{"output_types", output_types}, {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Create a new `TakeDataset` op kernel context.
  Status CreateTakeDatasetContext(
      OpKernel *op_kernel, gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<Tensor> input_tensors;
  int64 count;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1: take fewer than input size.
TestCase TakeLessTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{10, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
          /*count*/ 4,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 2: take more than input size.
TestCase TakeMoreTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{10, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
          /*count*/ 25,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {5}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {7}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {8}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {9})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ 10,
          /*breakpoints*/ {0, 2, 5, 11}};
}

// Test case 3: take all of input.
TestCase TakeAllTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{10, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
          /*count*/ -1,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {5}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {7}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {8}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {9})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ -1,
          /*breakpoints*/ {0, 2, 5, 11}};
}

// Test case 4: take nothing.
TestCase TakeNothingTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{10, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
          /*count*/ 0,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 2, 5, 11}};
}

class ParameterizedTakeDatasetOpTest
    : public TakeDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedTakeDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(take_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      take_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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

TEST_F(TakeDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = TakeLessTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  EXPECT_EQ(take_dataset->node_name(), kNodeName);
}

TEST_F(TakeDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = TakeLessTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  EXPECT_EQ(take_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedTakeDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(take_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedTakeDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(take_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedTakeDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  EXPECT_EQ(take_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_F(TakeDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = TakeLessTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(take_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedTakeDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(take_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      take_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedTakeDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(take_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      take_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedTakeDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(take_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      take_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  if (test_case.count == 0) {
    EXPECT_EQ(iterator->prefix(), "Iterator::EmptyTake");
  } else {
    EXPECT_EQ(iterator->prefix(), "Iterator::FiniteTake");
  }
}

TEST_P(ParameterizedTakeDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  auto expected_outputs_it = test_case.expected_outputs.begin();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_take_dataset;
  inputs_for_take_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_take_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> take_dataset_kernel;
  TF_ASSERT_OK(CreateTakeDatasetKernel(test_case.expected_output_dtypes,
                                       test_case.expected_output_shapes,
                                       &take_dataset_kernel));
  std::unique_ptr<OpKernelContext> take_dataset_context;
  TF_ASSERT_OK(CreateTakeDatasetContext(take_dataset_kernel.get(),
                                        &inputs_for_take_dataset,
                                        &take_dataset_context));
  DatasetBase *take_dataset;
  TF_ASSERT_OK(CreateDataset(take_dataset_kernel.get(),
                             take_dataset_context.get(), &take_dataset));
  core::ScopedUnref scoped_unref(take_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(take_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      take_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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

INSTANTIATE_TEST_SUITE_P(TakeDatasetOpTest, ParameterizedTakeDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TakeLessTestCase(), TakeMoreTestCase(),
                              TakeAllTestCase(), TakeNothingTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
