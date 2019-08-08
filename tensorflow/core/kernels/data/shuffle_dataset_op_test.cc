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

constexpr char kShuffleNodeName[] = "shuffle_dataset";
constexpr char kShuffleOpName[] = "ShuffleDataset";
constexpr char kShuffleAndRepeatNodeName[] = "shuffle_and_repeat_dataset";
constexpr char kShuffleAndRepeatOpName[] = "ShuffleAndRepeatDataset";

class ShuffleDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `ShuffleDataset`/`ShuffleAndRepeatDataset` op kernel
  Status CreateDatasetOpKernel(
      int64 count, bool reshuffle_each_iteration,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* shuffle_dataset_kernel) {
    NodeDef node_def;
    if (count == 1) {
      node_def = test::function::NDef(
          kShuffleNodeName, kShuffleOpName,
          {"input_dataset", "buffer_size", "seed", "seed2"},
          {{"reshuffle_each_iteration", reshuffle_each_iteration},
           {"output_types", output_types},
           {"output_shapes", output_shapes}});
    } else {
      node_def = test::function::NDef(
          kShuffleAndRepeatNodeName, kShuffleAndRepeatOpName,
          {"input_dataset", "buffer_size", "seed", "seed2", "count"},
          {{"output_types", output_types}, {"output_shapes", output_shapes}});
    }
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, shuffle_dataset_kernel));
    return Status::OK();
  }

  // Creates a new `ShuffleDataset`/`ShuffleAndRepeatDataset` op kernel context.
  Status CreateDatasetContext(OpKernel* const op_kernel,
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
  Tensor count;
  bool reshuffle_each_iteration;
  std::vector<Tensor> expected_shuffle_outputs;
  std::vector<Tensor> expected_reshuffle_outputs;
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

// Test case 1: test shuffle_dataset with reshuffle_each_iteration = false.
TestCase TestCase1() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ false,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({2, 3, 0, 5, 6, 4, 7, 8, 9, 1}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({2, 3, 0, 5, 6, 4, 7, 8, 9, 1}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 2: test shuffle_dataset with reshuffle_each_iteration = true.
TestCase TestCase2() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({2, 6, 1, 3, 9, 5, 0, 8, 7, 4}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({1, 6, 0, 5, 2, 7, 4, 3, 9, 8}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 3: similar with the test case 2 but a smaller buffer size than
// the input dataset.
TestCase TestCase3() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({0, 2, 1, 3, 5, 6, 4, 7, 8, 9}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({1, 0, 2, 3, 4, 5, 6, 7, 9, 8}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 4: similar with the test case 2 but has different seeds.
TestCase TestCase4() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({3, 0, 8, 1, 5, 4, 7, 2, 6, 9}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({4, 6, 9, 0, 1, 8, 2, 7, 3, 5}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 5: test shuffle_dataset with buffer_size = 1 &
// reshuffle_each_iteration = true.
TestCase TestCase5() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 10,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 6: test shuffle_dataset with an empty input dataset.
TestCase TestCase6() {
  return {
      /*range_data_param*/ {0, 0, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

// Test case 7: test shuffle_and_repeat_dataset with buffer_size = 10 &
// count = 2.
TestCase TestCase7() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*reshuffle_each_iteration*/ false,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>(
          {9, 0, 8, 6, 1, 3, 7, 2, 4, 5, 4, 3, 0, 5, 8, 2, 6, 9, 7, 1}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>(
          {9, 0, 8, 6, 1, 3, 7, 2, 4, 5, 4, 3, 0, 5, 8, 2, 6, 9, 7, 1}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 20,
      /*breakpoints*/ {0, 5, 22}};
}

// Test case 8: test shuffle_and_repeat_dataset with buffer_size = 10 &
// count = -1
TestCase TestCase8() {
  return {
      /*range_data_param*/ {0, 3, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-1}),
      /*reshuffle_each_iteration*/ false,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>(
          {2, 0, 1, 2, 0, 1, 1, 2, 0, 1, 0, 2, 2, 0, 1, 1, 0, 2, 2, 1, 0}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>(
          {2, 0, 1, 2, 0, 1, 1, 2, 0, 1, 0, 2, 2, 0, 1, 1, 0, 2, 2, 1, 0}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ kInfiniteCardinality,
      /*breakpoints*/ {0, 5, 20}};
}

TestCase InvalidBufferSizeTestCaseForShuffleDataset() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-1}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/ ConvertToTensorVec<int64>({}),
      /*expected_reshuffle_outputs*/ ConvertToTensorVec<int64>({}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

TestCase InvalidBufferSizeTestCaseForShuffleAndRepeatDataset() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-1}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*reshuffle_each_iteration*/ true,
      /*expected_shuffle_outputs*/ ConvertToTensorVec<int64>({}),
      /*expected_reshuffle_outputs*/ ConvertToTensorVec<int64>({}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 1, 9}};
}

TestCase InvalidCountTestCaseForShuffleAndRepeatDataset() {
  return {
      /*range_data_param*/ {0, 3, 1},
      /*buffer_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
      /*seed*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
      /*seed2*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*count*/ DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
      /*reshuffle_each_iteration*/ false,
      /*expected_shuffle_outputs*/
      ConvertToTensorVec<int64>({}),
      /*expected_reshuffle_outputs*/
      ConvertToTensorVec<int64>({}),
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 0,
      /*breakpoints*/ {0, 5, 20}};
}

class ParameterizedShuffleDatasetOpTest
    : public ShuffleDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedShuffleDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> shuffled_out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    shuffled_out_tensors.insert(shuffled_out_tensors.end(), next.begin(),
                                next.end());
    // For the forever-repeat case, we test only a finite number of steps of
    // the infinite sequence.
    if (count_value == -1 && shuffled_out_tensors.size() ==
                                 test_case.expected_shuffle_outputs.size()) {
      break;
    }
  }

  // Reshuffle the dataset.
  end_of_sequence = false;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));
  std::vector<Tensor> reshuffled_out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    reshuffled_out_tensors.insert(reshuffled_out_tensors.end(), next.begin(),
                                  next.end());
    // For the forever-repeat case, we test only a finite number of steps of
    // the infinite sequence.
    if (count_value == -1 && reshuffled_out_tensors.size() ==
                                 test_case.expected_shuffle_outputs.size()) {
      break;
    }
  }

  TF_EXPECT_OK(ExpectEqual(shuffled_out_tensors,
                           test_case.expected_shuffle_outputs,
                           /*compare_order*/ true));
  TF_EXPECT_OK(ExpectEqual(reshuffled_out_tensors,
                           test_case.expected_reshuffle_outputs,
                           /*compare_order*/ true));
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  if (count_value == 1) {
    EXPECT_EQ(dataset->node_name(), kShuffleNodeName);
  } else {
    EXPECT_EQ(dataset->node_name(), kShuffleAndRepeatNodeName);
  }
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  if (count_value == 1) {
    EXPECT_EQ(dataset->type_string(), kShuffleOpName);
  } else {
    EXPECT_EQ(dataset->type_string(), kShuffleAndRepeatOpName);
  }
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  TF_EXPECT_OK(VerifyTypesMatch(dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  EXPECT_EQ(dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedShuffleDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedShuffleDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedShuffleDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  if (count_value == 1) {
    EXPECT_EQ(iterator->prefix(), "Iterator::Shuffle");
  } else {
    EXPECT_EQ(iterator->prefix(), "Iterator::ShuffleAndRepeat");
  }
}

TEST_P(ParameterizedShuffleDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor count = test_case.count;
  int64 count_value = count.flat<int64>()(0);
  std::unique_ptr<OpKernel> dataset_kernel;
  TF_ASSERT_OK(
      CreateDatasetOpKernel(count_value, test_case.reshuffle_each_iteration,
                            test_case.expected_output_dtypes,
                            test_case.expected_output_shapes, &dataset_kernel));

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
      {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
       TensorValue(&seed), TensorValue(&seed2)});
  if (count_value != 1) inputs.push_back(TensorValue(&count));

  std::unique_ptr<OpKernelContext> dataset_context;
  TF_ASSERT_OK(
      CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
  DatasetBase* dataset;
  TF_ASSERT_OK(
      CreateDataset(dataset_kernel.get(), dataset_context.get(), &dataset));
  core::ScopedUnref scoped_unref_dataset(dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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
                                 *dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_shuffle_outputs,
                           /*compare_order*/ true));
}

INSTANTIATE_TEST_SUITE_P(ShuffleDatasetOpTest,
                         ParameterizedShuffleDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6(),
                              TestCase7(), TestCase8()})));

TEST_F(ShuffleDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  std::vector<TestCase> test_cases = {
      InvalidBufferSizeTestCaseForShuffleDataset(),
      InvalidBufferSizeTestCaseForShuffleAndRepeatDataset(),
      InvalidCountTestCaseForShuffleAndRepeatDataset()};
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  for (const auto& test_case : test_cases) {
    Tensor count = test_case.count;
    int64 count_value = count.flat<int64>()(0);
    std::unique_ptr<OpKernel> dataset_kernel;
    TF_ASSERT_OK(CreateDatasetOpKernel(
        count_value, test_case.reshuffle_each_iteration,
        test_case.expected_output_dtypes, test_case.expected_output_shapes,
        &dataset_kernel));

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
        {TensorValue(&range_dataset_tensor), TensorValue(&buffer_size),
         TensorValue(&seed), TensorValue(&seed2)});
    if (count_value != 1) inputs.push_back(TensorValue(&count));

    std::unique_ptr<OpKernelContext> dataset_context;
    TF_ASSERT_OK(
        CreateDatasetContext(dataset_kernel.get(), &inputs, &dataset_context));
    DatasetBase* shuffle_dataset;
    EXPECT_EQ(CreateDataset(dataset_kernel.get(), dataset_context.get(),
                            &shuffle_dataset)
                  .code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
