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
  // Creates TensorSliceDatasets and save them as the variant tensors.
  Status CreateTensorSliceDatasetTensors(
      std::vector<std::vector<Tensor>> tensor_vectors,
      std::vector<Tensor> *dataset_tensors) {
    for (int i = 0; i < tensor_vectors.size(); ++i) {
      std::vector<Tensor> tensors = tensor_vectors[i];
      DatasetBase *tensor_slice_dataset;
      TF_RETURN_IF_ERROR(
          CreateTensorSliceDataset(strings::StrCat("tensor_slice_node_", i),
                                   tensors, &tensor_slice_dataset));
      Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
      TF_RETURN_IF_ERROR(
          StoreDatasetInVariantTensor(tensor_slice_dataset, &dataset_tensor));
      dataset_tensors->emplace_back(std::move(dataset_tensor));
    }
    return Status::OK();
  }

  // Creates a new ConcatenateDataset op kernel.
  Status CreateConcatenateDatasetKernel(
      DataTypeVector output_tyeps,
      std::vector<PartialTensorShape> output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    node_def_ = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "another_dataset"},
        {{"output_types", output_tyeps}, {"output_shapes", output_shapes}});
    TF_CHECK_OK(CreateOpKernel(node_def_, op_kernel));
    return Status::OK();
  }

  // Creates a new ConcatenateDataset op kernel context.
  Status CreateConcatenateDatasetContext(
      OpKernel *const op_kernel, gtl::InlinedVector<TensorValue, 4> *inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }

 private:
  NodeDef node_def_;
};

struct TestParam {
  std::vector<std::vector<Tensor>> input_tensors;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
} TestCases[] = {
    // Same shape.
    {{{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2}, {1, 2, 3, 4}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                               {5, 6, 7, 8})},
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                               {11, 12, 13, 14}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                               {15, 16, 17, 18})}},
     {{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {1, 2}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {5, 6}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {3, 4}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {7, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {11, 12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {15, 16}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {13, 14}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {17, 18})}},
     {DT_INT64, DT_INT64},
     {PartialTensorShape({2}), PartialTensorShape({2})},
     4,
     {0, 2, 5}},
    // Different shape.
    {{{DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 3},
                                               {1, 2, 3, 4, 5, 6}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                               {7, 8, 9, 10})},
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 2},
                                               {11, 12, 13, 14}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2, 1}, {15, 16})}},
     {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3}, {1, 2, 3}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {7, 8}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{3}, {4, 5, 6}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {9, 10}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {11, 12}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {15}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{2}, {13, 14}),
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {16})},
     {DT_INT64, DT_INT64},
     {PartialTensorShape({-1}), PartialTensorShape({-1})},
     4,
     {0, 2, 5}}};

struct DatasetGetNextTest : ConcatenateDatasetOpTest,
                            ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  auto expected_outputs_it = expected_outputs.begin();
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      for (auto &tensor : out_tensors) {
        EXPECT_NE(expected_outputs_it, expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, DatasetGetNextTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(ConcatenateDatasetOpTest, DifferentDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = {
      {CreateTensor<int64>(TensorShape({2, 2}), {1, 2, 3, 4})},
      {CreateTensor<double>(TensorShape({2, 2}), {1.0, 2.0, 3.0, 4.0})}};
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::vector<PartialTensorShape> output_shapes =
      TestCases[0].expected_output_shapes;
  DataTypeVector output_dtypes = TestCases[0].expected_output_dtypes;
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(output_dtypes, output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  EXPECT_EQ(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(ConcatenateDatasetOpTest, DatasetName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = TestCases[0].input_tensors;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::vector<PartialTensorShape> output_shapes =
      TestCases[0].expected_output_shapes;
  DataTypeVector output_dtypes = TestCases[0].expected_output_dtypes;
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(output_dtypes, output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->name(), kOpName);
}

struct DatasetOutputDtypesTest : ConcatenateDatasetOpTest,
                                 ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetOutputDtypesTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_OK(VerifyTypesMatch(dataset->output_dtypes(), expected_output_dtypes));
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, DatasetOutputDtypesTest,
                        ::testing::ValuesIn(TestCases));

struct DatasetOutputShapesTest : ConcatenateDatasetOpTest,
                                 ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetOutputShapesTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_OK(
      VerifyShapesCompatible(dataset->output_shapes(), expected_output_shapes));
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, DatasetOutputShapesTest,
                        ::testing::ValuesIn(TestCases));

struct DatasetCardinalityTest : ConcatenateDatasetOpTest,
                                ::testing::WithParamInterface<TestParam> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  int64 expected_cardinality = GetParam().expected_cardinality;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->Cardinality(), expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, DatasetCardinalityTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(ConcatenateDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = TestCases[0].input_tensors;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::vector<PartialTensorShape> output_shapes =
      TestCases[0].expected_output_shapes;
  DataTypeVector output_dtypes = TestCases[0].expected_output_dtypes;
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(output_dtypes, output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

struct IteratorOutputDtypesTest : ConcatenateDatasetOpTest,
                                  ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorOutputDtypesTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(
      VerifyTypesMatch(iterator->output_dtypes(), expected_output_dtypes));
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, IteratorOutputDtypesTest,
                        ::testing::ValuesIn(TestCases));

struct IteratorOutputShapesTest : ConcatenateDatasetOpTest,
                                  ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorOutputShapesTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      expected_output_shapes));
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, IteratorOutputShapesTest,
                        ::testing::ValuesIn(TestCases));

TEST_F(ConcatenateDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = TestCases[0].input_tensors;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }
  std::vector<PartialTensorShape> output_shapes =
      TestCases[0].expected_output_shapes;
  DataTypeVector output_dtypes = TestCases[0].expected_output_dtypes;
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(output_dtypes, output_shapes,
                                              &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  EXPECT_EQ(iterator->prefix(), "Iterator::Concatenate");
}

struct IteratorRoundtripTest : ConcatenateDatasetOpTest,
                               ::testing::WithParamInterface<TestParam> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<std::vector<Tensor>> tensor_vectors = GetParam().input_tensors;
  std::vector<Tensor> expected_outputs = GetParam().expected_outputs;
  auto expected_outputs_it = expected_outputs.begin();
  std::vector<PartialTensorShape> expected_output_shapes =
      GetParam().expected_output_shapes;
  DataTypeVector expected_output_dtypes = GetParam().expected_output_dtypes;
  std::vector<Tensor> tensor_slice_dataset_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensors(tensor_vectors,
                                               &tensor_slice_dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto &tensor : tensor_slice_dataset_tensors) {
    inputs.emplace_back(&tensor);
  }

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateConcatenateDatasetKernel(
      expected_output_dtypes, expected_output_shapes, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateConcatenateDatasetContext(dataset_kernel.get(), &inputs,
                                               &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
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
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        for (auto &tensor : out_tensors) {
          EXPECT_NE(expected_outputs_it, expected_outputs.end());
          TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
          expected_outputs_it++;
        }
      }
      cur_iteration++;
    }

    if (breakpoint >= dataset->Cardinality()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_CASE_P(ConcatenateDatasetOpTest, IteratorRoundtripTest,
                        ::testing::ValuesIn(TestCases));
}  // namespace
}  // namespace data
}  // namespace tensorflow
