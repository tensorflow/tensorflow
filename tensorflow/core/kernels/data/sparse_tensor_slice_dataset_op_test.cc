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

constexpr char kNodeName[] = "sparse_tensor_slice_dataset";
constexpr char kOpName[] = "SparseTensorSliceDataset";

class SparseTensorSliceDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new SparseTensorSliceDataset op kernel.
  Status CreateSparseTensorSliceDatasetKernel(
      DataType tvalues, std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"indices", "values", "dense_shape"},
        {{"Tvalues", tvalues}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new SparseTensorSliceDataset op kernel context.
  Status CreateSparseTensorSliceDatasetContext(
      OpKernel *const op_kernel, gtl::InlinedVector<TensorValue, 4> *inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct SparseTensorParam {
  Tensor indices;
  Tensor values;
  Tensor dense_shape;
};

struct TestCase {
  SparseTensorParam input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs;
  std::vector<int> breakpoints;
};

TestCase TwoDimsTestCase() {
  return {
      /*input_sparse_tensor*/
      {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({2, 2},
                                                           {0, 0, 1, 1}),
       /*values*/ DatasetOpsTestBase::CreateTensor<int32>({2}, {888, 999}),
       /*dense_shape*/ DatasetOpsTestBase::CreateTensor<int64>({2}, {2, 2})},
      /*expected_outputs*/
      {{/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 1}, {0}),
        /*values*/ DatasetOpsTestBase::CreateTensor<int32>({1}, {888}),
        /*dense_shape*/ DatasetOpsTestBase::CreateTensor<int64>({1}, {2})},
       {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 1}, {1}),
        /*values*/ DatasetOpsTestBase::CreateTensor<int32>({1}, {999}),
        /*dense_shape*/ DatasetOpsTestBase::CreateTensor<int64>({1}, {2})}},
      /*breakpoints*/ {0, 1, 2}};
}

TestCase ThreeDimsTestCase() {
  return {
      /*input_sparse_tensor*/
      {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({2, 3},
                                                           {0, 0, 0, 1, 1, 1}),
       /*values*/ DatasetOpsTestBase::CreateTensor<double>({2}, {888.0, 999.0}),
       /*dense_shape*/ DatasetOpsTestBase::CreateTensor<int64>({3}, {2, 2, 2})},
      /*expected_outputs*/
      {{/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 2}, {0, 0}),
        /*values*/ DatasetOpsTestBase::CreateTensor<double>({1}, {888.0}),
        /*dense_shape*/ DatasetOpsTestBase::CreateTensor<int64>({2}, {2, 2})},
       {{/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 2}, {1, 1})},
        {/*values*/ DatasetOpsTestBase::CreateTensor<double>({1}, {999.0})},
        {/*dense_shape*/ DatasetOpsTestBase::CreateTensor<int64>({2},
                                                                 {2, 2})}}},
      /*breakpoints*/ {0, 1, 2}};
}

TestCase FourDimsTestCase() {
  return {
      /*input_sparse_tensor*/
      {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>(
           {2, 4}, {0, 0, 0, 0, 1, 1, 1, 1}),
       /*values*/ DatasetOpsTestBase::CreateTensor<string>({2}, {"a", "b"}),
       /*dense_shape*/
       DatasetOpsTestBase::CreateTensor<int64>({4}, {3, 2, 2, 2})},
      /*expected_outputs*/
      {{/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 3}, {0, 0, 0}),
        /*values*/ DatasetOpsTestBase::CreateTensor<string>({1}, {"a"}),
        /*dense_shape*/
        DatasetOpsTestBase::CreateTensor<int64>({3}, {2, 2, 2})},
       {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 3}, {1, 1, 1}),
        /*values*/ DatasetOpsTestBase::CreateTensor<string>({1}, {"b"}),
        /*dense_shape*/
        DatasetOpsTestBase::CreateTensor<int64>({3}, {2, 2, 2})},
       {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({0, 3}, {}),
        /*values*/ DatasetOpsTestBase::CreateTensor<string>({0}, {}),
        /*dense_shape*/
        DatasetOpsTestBase::CreateTensor<int64>({3}, {2, 2, 2})}},
      /*breakpoints*/ {0, 1, 3}};
}

TestCase FiveDimsTestCase() {
  return {/*input_sparse_tensor*/
          {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>(
               {2, 5}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}),
           /*values*/ DatasetOpsTestBase::CreateTensor<int32>({2}, {888, 999}),
           /*dense_shape*/
           DatasetOpsTestBase::CreateTensor<int64>({5}, {3, 2, 2, 2, 2})},
          /*expected_outputs*/
          {{/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 4},
                                                                {0, 0, 0, 0}),
            /*values*/ DatasetOpsTestBase::CreateTensor<int32>({1}, {888}),
            /*dense_shape*/
            DatasetOpsTestBase::CreateTensor<int64>({4}, {2, 2, 2, 2})},
           {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({1, 4},
                                                                {1, 1, 1, 1}),
            /*values*/ DatasetOpsTestBase::CreateTensor<int32>({1}, {999}),
            /*dense_shape*/
            DatasetOpsTestBase::CreateTensor<int64>({4}, {2, 2, 2, 2})},
           {/*indices*/ DatasetOpsTestBase::CreateTensor<int64>({0, 4}, {}),
            /*values*/ DatasetOpsTestBase::CreateTensor<int32>({0}, {}),
            /*dense_shape*/
            DatasetOpsTestBase::CreateTensor<int64>({4}, {2, 2, 2, 2})}},
          /*breakpoints*/ {0, 1, 3}};
}

class ParameterizedSparseTensorSliceDatasetOpTest
    : public SparseTensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
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
  auto expected_outputs_it = expected_outputs.begin();
  while (!end_of_sequence) {
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      TF_EXPECT_OK(ExpectEqual(out_tensors[0], expected_outputs_it->indices));
      TF_EXPECT_OK(ExpectEqual(out_tensors[1], expected_outputs_it->values));
      TF_EXPECT_OK(
          ExpectEqual(out_tensors[2], expected_outputs_it->dense_shape));
      expected_outputs_it++;
    }
  }
  EXPECT_EQ(expected_outputs_it, expected_outputs.end());
}

TEST_F(SparseTensorSliceDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = TwoDimsTestCase();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->node_name(), kNodeName);
}

TEST_F(SparseTensorSliceDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = TwoDimsTestCase();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->type_string(), kOpName);
}

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  DataTypeVector expected_output_dtypes = {
      expected_outputs[0].indices.dtype(), expected_outputs[0].values.dtype(),
      expected_outputs[0].dense_shape.dtype()};
  TF_EXPECT_OK(
      VerifyTypesMatch(dataset->output_dtypes(), expected_output_dtypes));
}

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::vector<PartialTensorShape> expected_output_shapes = {
      expected_outputs[0].indices.shape(), expected_outputs[0].values.shape(),
      expected_outputs[0].dense_shape.shape()};
  TF_EXPECT_OK(
      VerifyShapesCompatible(dataset->output_shapes(), expected_output_shapes));
}

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = TwoDimsTestCase();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  EXPECT_EQ(dataset->Cardinality(), expected_outputs.size());
}

TEST_F(SparseTensorSliceDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = TwoDimsTestCase();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
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

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  DataTypeVector expected_output_dtypes = {
      expected_outputs[0].indices.dtype(), expected_outputs[0].values.dtype(),
      expected_outputs[0].dense_shape.dtype()};
  TF_EXPECT_OK(
      VerifyTypesMatch(iterator->output_dtypes(), expected_output_dtypes));
}

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  std::vector<PartialTensorShape> expected_output_shapes = {
      expected_outputs[0].indices.shape(), expected_outputs[0].values.shape(),
      expected_outputs[0].dense_shape.shape()};
  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      expected_output_shapes));
}

TEST_F(SparseTensorSliceDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = TwoDimsTestCase();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
  DatasetBase *dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_kernel_ctx.get(), &dataset));
  core::ScopedUnref scoped_unref(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_kernel_ctx.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  EXPECT_EQ(iterator->prefix(), strings::StrCat("Iterator::SparseTensorSlice"));
}

TEST_P(ParameterizedSparseTensorSliceDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = GetParam();
  SparseTensorParam input_sparse_tensor = test_case.input_sparse_tensor;
  std::vector<SparseTensorParam> expected_outputs = test_case.expected_outputs;
  std::vector<int> breakpoints = test_case.breakpoints;
  DataType tvalues = input_sparse_tensor.values.dtype();
  gtl::InlinedVector<TensorValue, 4> inputs = {
      &input_sparse_tensor.indices, &input_sparse_tensor.values,
      &input_sparse_tensor.dense_shape};

  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetKernel(tvalues, &dataset_kernel));
  std::unique_ptr<OpKernelContext> dataset_kernel_ctx;
  TF_ASSERT_OK(CreateSparseTensorSliceDatasetContext(
      dataset_kernel.get(), &inputs, &dataset_kernel_ctx));
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

  int cur_iteration = 0;
  bool end_of_sequence = false;
  int64 num_slices = input_sparse_tensor.dense_shape.dim_size(0);
  std::vector<Tensor> out_tensors;

  for (int breakpoint : breakpoints) {
    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      cur_iteration++;
    }

    if (breakpoint == 0) {
      EXPECT_FALSE(end_of_sequence);
    } else if (breakpoint <= num_slices) {
      for (int i = 0; i < out_tensors.size(); ++i) {
        TF_EXPECT_OK(ExpectEqual(out_tensors[0],
                                 expected_outputs[cur_iteration - 1].indices));
        TF_EXPECT_OK(ExpectEqual(out_tensors[1],
                                 expected_outputs[cur_iteration - 1].values));
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[2], expected_outputs[cur_iteration - 1].dense_shape));
      }
    } else {
      EXPECT_TRUE(end_of_sequence);
    }

    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_ASSERT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_ASSERT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_ASSERT_OK(iterator->Restore(iterator_ctx.get(), &reader));
  }
}

INSTANTIATE_TEST_SUITE_P(SparseTensorSliceDatasetOpTest,
                         ParameterizedSparseTensorSliceDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TwoDimsTestCase(), ThreeDimsTestCase(),
                              FourDimsTestCase(), FiveDimsTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
