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

constexpr char kNodeName[] = "flat_map_dataset";
constexpr char kOpName[] = "FlatMapDataset";

class FlatMapDatasetOpTest : public DatasetOpsTestBase {
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

  // Creates a new `FlatMapDataset` op kernel
  Status CreateFlatMapDatasetKernel(
      const FunctionDefHelper::AttrValueWrapper &func,
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel) {
    NodeDef node_def =
        test::function::NDef(kNodeName, kOpName, {"input_dataset"},
                             {{"f", func},
                              {"Targuments", {}},
                              {"output_types", output_types},
                              {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Creates a new `FlatMapDataset` op kernel context.
  Status CreateFlatMapDatasetContext(
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

TestCase MakeTensorSliceDatasetFuncTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          FunctionDefHelper::FunctionRef(
              /*name*/ "MakeTensorSliceDataset",
              /*attrs*/ {{"Toutput_types", DataTypeVector({DT_INT64})},
                         {"output_shapes", std::vector<PartialTensorShape>(
                                               {PartialTensorShape({1})})}}),
          /*func_lib*/ {test::function::MakeTensorSliceDataset()},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {3}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {5}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {7}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape{1}, {8})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ tensorflow::data::kUnknownCardinality,
          /*breakpoints*/ {0, 4, 11}};
}

// Test case 2: test the case if the function does not return a single scalar
// of dtype DT_VARIANT.
TestCase InvalidFuncTestCase() {
  return {/*input_tensors*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*func*/
          FunctionDefHelper::FunctionRef(/*name*/ "NonZero",
                                         /*attrs*/ {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::NonZero()},
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({1})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

class ParameterizedFlatMapDatasetOpTest
    : public FlatMapDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedFlatMapDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(flat_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(flat_map_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
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

TEST_F(FlatMapDatasetOpTest, InvalidFunc) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = InvalidFuncTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(flat_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(flat_map_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                              &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(FlatMapDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = MakeTensorSliceDatasetFuncTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  EXPECT_EQ(flat_map_dataset->node_name(), kNodeName);
}

TEST_F(FlatMapDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = MakeTensorSliceDatasetFuncTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  EXPECT_EQ(flat_map_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedFlatMapDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(flat_map_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFlatMapDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(flat_map_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedFlatMapDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  EXPECT_EQ(flat_map_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_F(FlatMapDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = MakeTensorSliceDatasetFuncTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(flat_map_dataset->Save(serialization_ctx.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedFlatMapDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(flat_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(flat_map_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                              &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedFlatMapDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(flat_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(flat_map_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                              &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(FlatMapDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = MakeTensorSliceDatasetFuncTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(flat_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(flat_map_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
                                              &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::FlatMap");
}

TEST_P(ParameterizedFlatMapDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  const TestCase &test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> flat_map_dataset_kernel;
  TF_ASSERT_OK(CreateFlatMapDatasetKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &flat_map_dataset_kernel));

  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensor(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs({&tensor_slice_dataset_tensor});
  std::unique_ptr<OpKernelContext> flat_map_dataset_context;
  TF_ASSERT_OK(CreateFlatMapDatasetContext(flat_map_dataset_kernel.get(),
                                           &inputs, &flat_map_dataset_context));
  DatasetBase *flat_map_dataset;
  TF_ASSERT_OK(CreateDataset(flat_map_dataset_kernel.get(),
                             flat_map_dataset_context.get(),
                             &flat_map_dataset));
  core::ScopedUnref scoped_unref(flat_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(flat_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(flat_map_dataset->MakeIterator(iterator_ctx.get(), "Iterator",
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

INSTANTIATE_TEST_SUITE_P(FlatMapDatasetOpTest,
                         ParameterizedFlatMapDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {MakeTensorSliceDatasetFuncTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
