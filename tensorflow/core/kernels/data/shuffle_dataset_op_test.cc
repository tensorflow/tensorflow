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

constexpr char kNodeName[] = "shuffle_dataset";
constexpr char kOpName[] = "ShuffleDataset";

class ShuffleDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `ShuffleDataset` op kernel
  Status CreateShuffleDatasetOpKernel(
      bool reshuffle_each_iteration, const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* shuffle_dataset_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "buffer_size", "seed", "seed2"},
        {{"reshuffle_each_iteration", reshuffle_each_iteration},
         {"output_types", output_types},
         {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, shuffle_dataset_kernel));
    return Status::OK();
  }

  // Creates a new `ShuffleDataset` op kernel context.
  Status CreateShuffleDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct RangeDatasetParam {
  int64 start;
  int64 end;
  int64 step;
};

struct TestCase {
  RangeDatasetParam range_data_param;
  Tensor buffer_size;
  Tensor seed;
  Tensor seed2;
  bool reshuffle_each_iteration;
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
        DatasetOpsTestBase::CreateTensor<T>(TensorShape({}), {value}));
  }
  return tensors;
}

// Test case 1: normal case with reshuffle_each_iteration = false
TestCase TestCase1() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*reshuffle_each_iteration*/ false,
      /*expected_outputs*/
      ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 2: normal case with reshuffle_each_iteration = true
TestCase TestCase2() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*reshuffle_each_iteration*/ true,
      /*expected_outputs*/
      ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 3: special case with buffer_size = 1 &
// reshuffle_each_iteration = true
TestCase TestCase3() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*reshuffle_each_iteration*/ true,
      /*expected_outputs*/
      ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

TestCase InvalidBufferSizeTestCase() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-1}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*reshuffle_each_iteration*/ true,
      /*expected_outputs*/ ConvertToTensorVec<int64>({}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

class ParameterizedShuffleDatasetOpTest
    : public ShuffleDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedShuffleDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shuffle_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shuffle_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  // When `buffer_size = 1`, the output sequence of `ShuffleDataset` will be in
  // order, so we need to consider the element sequence when evaluating the
  // result for this case.
  bool expect_items_equal = test_case.buffer_size.flat<int64>()(0) > 1;
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*expect_items_equal*/ expect_items_equal));
}

TEST_F(ShuffleDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  EXPECT_EQ(shuffle_dataset->node_name(), kNodeName);
}

TEST_F(ShuffleDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  EXPECT_EQ(shuffle_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(shuffle_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(shuffle_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(shuffle_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(shuffle_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedShuffleDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shuffle_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shuffle_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shuffle_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shuffle_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(ShuffleDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shuffle_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shuffle_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Shuffle");
}

TEST_P(ParameterizedShuffleDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  TF_ASSERT_OK(CreateDataset(shuffle_dataset_kernel.get(),
                             shuffle_dataset_context.get(), &shuffle_dataset));
  core::ScopedUnref scoped_unref_shuffle_dataset(shuffle_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shuffle_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shuffle_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, "Iterator",
                                 *shuffle_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  // When `buffer_size = 1`, the output sequence of `ShuffleDataset` will be in
  // order, so we need to consider the element sequence when evaluating the
  // result for this case.
  bool expect_items_equal = test_case.buffer_size.flat<int64>()(0) > 1;
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*expect_items_equal*/ expect_items_equal));
}

INSTANTIATE_TEST_SUITE_P(ShuffleDatasetOpTest,
                         ParameterizedShuffleDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

TEST_F(ShuffleDatasetOpTest, InvalidBufferSize) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = InvalidBufferSizeTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shuffle_dataset_kernel;
  TF_ASSERT_OK(CreateShuffleDatasetOpKernel(
      test_case.reshuffle_each_iteration, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shuffle_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor buffer_size = test_case.buffer_size;
  Tensor seed = test_case.seed;
  Tensor seed2 = test_case.seed2;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&range_dataset_tensor, &buffer_size, &seed, &seed2});

  std::unique_ptr<OpKernelContext> shuffle_dataset_context;
  TF_ASSERT_OK(CreateShuffleDatasetContext(shuffle_dataset_kernel.get(),
                                           &inputs, &shuffle_dataset_context));
  DatasetBase* shuffle_dataset;
  EXPECT_EQ(CreateDataset(shuffle_dataset_kernel.get(),
                          shuffle_dataset_context.get(), &shuffle_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
