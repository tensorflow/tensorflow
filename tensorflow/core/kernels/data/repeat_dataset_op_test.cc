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

constexpr char kNodeName[] = "repeat_dataset";
constexpr char kOpName[] = "RepeatDataset";

class RepeatDatasetOpTest : public DatasetOpsTestBase {
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

  // Creates a new `RepeatDataset` op kernel.
  Status CreateRepeatDatasetKernel(
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "count"},
        {{"output_types", output_types}, {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Create a new `RepeatDataset` op kernel context.
  Status CreateRepeatDatasetContext(
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

TestCase FiniteRepeatTestCase() {
  return {
      /*input_tensors*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {1, 2, 3, 4}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape{2, 1}, {"a", "b"})},
      /*count*/ 2,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {1, 2}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape{1}, {"a"}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {3, 4}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape{1}, {"b"}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {1, 2}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape{1}, {"a"}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {3, 4}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape{1}, {"b"})},
      /*expected_output_dtypes*/ {DT_INT64, DT_STRING},
      /*expected_output_shapes*/
      {PartialTensorShape({2}), PartialTensorShape({1})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 3}};
}

TestCase EmptyRepeatTestCase() {
  return {
      /*input_tensors*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {1, 2, 3, 4}),
       DatasetOpsTestBase::CreateTensor<string>(TensorShape{2, 1}, {"a", "b"})},
      /*count*/ 0,
      /*expected_outputs*/
      {},
      /*expected_output_dtypes*/ {DT_INT64, DT_STRING},
      /*expected_output_shapes*/
      {PartialTensorShape({2}), PartialTensorShape({1})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 3}};
}

TestCase ForeverRepeatTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 1}, {1, 2})},
          /*count*/ -1,
          /*expected_outputs*/
          // Use the first group of the repeated tensors to represent the
          // infinite outputs.
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {2})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ -1,
          /*breakpoints*/ {0, 1, 3}};
}

class ParameterizedDatasetOpTest
    : public RepeatDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(repeat_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      repeat_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  auto expected_outputs_it = test_case.expected_outputs.begin();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;

  if (test_case.count < 0) {
    // We test only a finite number of steps of the infinite sequence.
    for (int i = 0; i < 100; ++i) {
      out_tensors.clear();
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      for (const auto &tensor : out_tensors) {
        TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
        expected_outputs_it++;
        // In the forever-repeat test case, the first group of the repeated
        // tensors is used to represent the expected outputs, so the iterator
        // of the expected outputs needs to be reset once it reaches the end.
        if (expected_outputs_it == test_case.expected_outputs.end()) {
          expected_outputs_it = test_case.expected_outputs.begin();
        }
      }
    }
    EXPECT_FALSE(end_of_sequence);
  } else {
    while (!end_of_sequence) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
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
}

TEST_F(RepeatDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = FiniteRepeatTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  EXPECT_EQ(repeat_dataset->node_name(), kNodeName);
}

TEST_F(RepeatDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = FiniteRepeatTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  EXPECT_EQ(repeat_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);
  TF_EXPECT_OK(VerifyTypesMatch(repeat_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);
  TF_EXPECT_OK(VerifyShapesCompatible(repeat_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  EXPECT_EQ(repeat_dataset->Cardinality(), GetParam().expected_cardinality);
}

TEST_F(RepeatDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = FiniteRepeatTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(repeat_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(repeat_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      repeat_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(repeat_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      repeat_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  const TestCase &test_case = GetParam();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));
  Tensor count = CreateTensor<int64>(TensorShape{}, {test_case.count});
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(repeat_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      repeat_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  if (test_case.count < 0) {
    EXPECT_EQ(iterator->prefix(), "Iterator::ForeverRepeat");
  } else if (test_case.count == 0) {
    EXPECT_EQ(iterator->prefix(), "Iterator::EmptyRepeat");
  } else {
    EXPECT_EQ(iterator->prefix(), "Iterator::FiniteRepeat");
  }
}

TEST_P(ParameterizedDatasetOpTest, Roundtrip) {
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
  gtl::InlinedVector<TensorValue, 4> inputs_for_repeat_dataset;
  inputs_for_repeat_dataset.emplace_back(&tensor_slice_dataset_tensor);
  inputs_for_repeat_dataset.emplace_back(&count);

  std::unique_ptr<OpKernel> repeat_dataset_kernel;
  TF_ASSERT_OK(CreateRepeatDatasetKernel(test_case.expected_output_dtypes,
                                         test_case.expected_output_shapes,
                                         &repeat_dataset_kernel));
  std::unique_ptr<OpKernelContext> repeat_dataset_context;
  TF_ASSERT_OK(CreateRepeatDatasetContext(repeat_dataset_kernel.get(),
                                          &inputs_for_repeat_dataset,
                                          &repeat_dataset_context));
  DatasetBase *repeat_dataset;
  TF_ASSERT_OK(CreateDataset(repeat_dataset_kernel.get(),
                             repeat_dataset_context.get(), &repeat_dataset));
  core::ScopedUnref scoped_unref(repeat_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(repeat_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      repeat_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = repeat_dataset->Cardinality() == 0;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  std::vector<int> breakpoints = GetParam().breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator->Restore(iterator_ctx.get(), &reader));

    while (cur_iteration < breakpoint) {
      out_tensors.clear();
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
      if (test_case.count < 0 &&
          expected_outputs_it == test_case.expected_outputs.end()) {
        expected_outputs_it = test_case.expected_outputs.begin();
      }
    }

    if (breakpoint >= repeat_dataset->Cardinality()) {
      if (test_case.count < 0) {
        EXPECT_FALSE(end_of_sequence);
      } else {
        EXPECT_TRUE(end_of_sequence);
        EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
      }
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RepeatDatasetOpTest, ParameterizedDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {FiniteRepeatTestCase(), EmptyRepeatTestCase(),
                              ForeverRepeatTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
