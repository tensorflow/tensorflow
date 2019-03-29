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

constexpr char kNodeName[] = "concatenate_dataset";
constexpr char kOpName[] = "ConcatenateDataset";

class ConcatenateDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates `TensorSliceDataset` variant tensors from the input vector of
  // tensor vectors.
  Status CreateTensorSliceDatasetTensors(
      const std::vector<std::vector<Tensor>> &tensor_vectors,
      std::vector<Tensor> *const dataset_tensors) {
    for (int i = 0; i < tensor_vectors.size(); ++i) {
      std::vector<Tensor> tensors = tensor_vectors[i];
      DatasetBase *tensor_slice_dataset;
      TF_RETURN_IF_ERROR(
          CreateTensorSliceDataset(strings::StrCat("tensor_slice_node_", i),
                                   &tensors, &tensor_slice_dataset));
      Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
      TF_RETURN_IF_ERROR(
          StoreDatasetInVariantTensor(tensor_slice_dataset, &dataset_tensor));
      dataset_tensors->emplace_back(std::move(dataset_tensor));
    }
    return Status::OK();
  }

  // Creates a new ConcatenateDataset op kernel.
  Status CreateConcatenateDatasetKernel(
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "another_dataset"},
        {{"output_types", output_types}, {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new ConcatenateDataset op kernel context.
  Status CreateConcatenateDatasetContext(
      OpKernel *const op_kernel,
      gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<std::vector<Tensor>> input_tensors;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1: same shape.
TestCase SameShapeTestCase() {
  return {/*input_tensors*/
          {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                                    {1, 2, 3, 4}),
            DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                                    {5, 6, 7, 8})},
           {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                                    {11, 12, 13, 14}),
            DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                                    {15, 16, 17, 18})}},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {1, 2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {5, 6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {3, 4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {7, 8}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {11, 12}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {15, 16}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {13, 14}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {17, 18})},
          /*expected_output_dtypes*/ {DT_INT64, DT_INT64},
          /*expected_output_shapes*/
          {PartialTensorShape({2}), PartialTensorShape({2})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 2: different shape.
TestCase DifferentShapeTestCase() {
  return {
      /*input_tensors*/
      {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                                {1, 2, 3, 4, 5, 6}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                                {7, 8, 9, 10})},
       {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                                {11, 12, 13, 14}),
        DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 1}, {15, 16})}},
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3}, {1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {7, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3}, {4, 5, 6}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {9, 10}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {11, 12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {15}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {13, 14}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {16})},
      /*expected_output_dtypes*/ {DT_INT64, DT_INT64},
      /*expected_output_shapes*/
      {PartialTensorShape({-1}), PartialTensorShape({-1})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 2, 5}};
}

// Test case 3: different dtypes
TestCase DifferentDtypeTestCase() {
  return {/*input_tensors*/ {{DatasetOpsTestBase::CreateTensor<int64>(
                                 TensorShape({2, 2}), {1, 2, 3, 4})},
                             {DatasetOpsTestBase::CreateTensor<double>(
                                 TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}},
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({2})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

class ParameterizedConcatenateDatasetOpTest
    : public ConcatenateDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedConcatenateDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(concatenate_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
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

TEST_F(ConcatenateDatasetOpTest, DifferentDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = DifferentDtypeTestCase();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  EXPECT_EQ(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                          &concatenate_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(ConcatenateDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = SameShapeTestCase();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);

  EXPECT_EQ(concatenate_dataset->node_name(), kNodeName);
}

TEST_F(ConcatenateDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = SameShapeTestCase();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);

  EXPECT_EQ(concatenate_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedConcatenateDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);
  TF_EXPECT_OK(VerifyTypesMatch(concatenate_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedConcatenateDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(concatenate_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedConcatenateDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);

  EXPECT_EQ(concatenate_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_F(ConcatenateDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = SameShapeTestCase();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(concatenate_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedConcatenateDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(concatenate_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                 &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedConcatenateDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(concatenate_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                 &iterator));
  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(ConcatenateDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = SameShapeTestCase();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(concatenate_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                 &iterator));
  EXPECT_EQ(iterator->prefix(), "Iterator::Concatenate");
}

TEST_P(ParameterizedConcatenateDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(test_case.input_tensors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *concatenate_dataset;
  TF_ASSERT_OK(CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(),
                             &concatenate_dataset));
  core::ScopedUnref scoped_unref(concatenate_dataset);
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(concatenate_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                                 &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  std::vector<int> breakpoints = GetParam().breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator->Restore(iterator_ctx.get(), &reader));

    while (cur_iteration < breakpoint) {
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

    if (breakpoint >= concatenate_dataset->Cardinality()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ConcatenateDatasetOpTest,
                         ParameterizedConcatenateDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {SameShapeTestCase(), DifferentShapeTestCase()})));
}  // namespace
}  // namespace data
}  // namespace tensorflow
