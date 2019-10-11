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
#include "tensorflow/core/kernels/data/shard_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "shard_dataset";

class ShardDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `ShardDataset` op kernel.
  Status CreateShardDatasetOpKernel(
      bool require_non_empty, const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(ShardDatasetOp::kDatasetType),
        {ShardDatasetOp::kInputDataset, ShardDatasetOp::kNumShards,
         ShardDatasetOp::kIndex},
        {{ShardDatasetOp::kRequireNonEmpty, require_non_empty},
         {ShardDatasetOp::kOutputTypes, output_types},
         {ShardDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Create a new `ShardDataset` op kernel context
  Status CreateShardDatasetContext(
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
  RangeDatasetParam range_dataset_param;
  Tensor num_shards;
  Tensor index;
  bool require_non_empty;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test Case 1: simple case.
TestCase TestCase1() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {2}),
          /*require_non_empty*/ true,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {2}),
           CreateTensor<int64>(TensorShape({}), {7})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 2,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 2: zero offset.
TestCase TestCase2() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {0}),
          /*require_non_empty*/ true,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {0}),
           CreateTensor<int64>(TensorShape({}), {5})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 2,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 3: iterator ends before first element.
TestCase TestCase3() {
  return {/*range_data_param*/ {0, 1, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {2}),
          /*require_non_empty*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 1}};
}

// Test Case 4: larger num_shards.
TestCase TestCase4() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {7}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*require_non_empty*/ true,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {5})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 1,
          /*breakpoints*/ {0, 5}};
}

// Test Case 5: index == num_shards.
TestCase TestCase5() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {4}),
          /*require_non_empty*/ true,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {4}),
           CreateTensor<int64>(TensorShape({}), {9})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 2,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 6: similar with test_case_5 but the number of outputs could not be
// divided evenly by num_shards.
TestCase TestCase6() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {4}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {3}),
          /*require_non_empty*/ true,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {3}),
           CreateTensor<int64>(TensorShape({}), {7})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 2,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 7: num_shard is larger than the cardinality of input dataset;
// require_non_empty = false.
TestCase TestCase7() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {20}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*require_non_empty*/ false,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {5})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 1,
          /*breakpoints*/ {0, 5}};
}

// Test Case 8: similar with test_case_7 but require_non_empty = true.
TestCase NoElemForEachShardTestCase() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {20}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*require_non_empty*/ true,
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {5})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 1,
          /*breakpoints*/ {0, 5}};
}

TestCase IndexGreaterNumShardsCase() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {7}),
          /*require_non_empty*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

TestCase NegativeIndexTestCase() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {5}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {-3}),
          /*require_non_empty*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

TestCase NegativeNumShardsTestCase() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {-3}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {1}),
          /*require_non_empty*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

TestCase ZeroNumShardsTestCase() {
  return {/*range_data_param*/ {0, 10, 1},
          /*num_shards*/
          CreateTensor<int64>(TensorShape({}), {0}),
          /*index*/
          CreateTensor<int64>(TensorShape({}), {1}),
          /*require_non_empty*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

class ParameterizedShardDatasetOpTest
    : public ShardDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedShardDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shard_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence));
    if (!end_of_sequence) {
      EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
      TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
      expected_outputs_it++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

TEST_F(ShardDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  EXPECT_EQ(shard_dataset->node_name(), kNodeName);
}

TEST_F(ShardDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  EXPECT_EQ(shard_dataset->type_string(),
            name_utils::OpName(ShardDatasetOp::kDatasetType));
}

TEST_P(ParameterizedShardDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(shard_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedShardDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(shard_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedShardDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  EXPECT_EQ(shard_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedShardDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shard_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedShardDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shard_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(ShardDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shard_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), name_utils::IteratorPrefix(
                                    ShardDatasetOp::kDatasetType, "Iterator"));
}

TEST_P(ParameterizedShardDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shard_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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
                                 *shard_dataset, &iterator));

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

INSTANTIATE_TEST_SUITE_P(ShardDatasetOpTest, ParameterizedShardDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6(),
                              TestCase7()})));

TEST_F(ShardDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<TestCase> test_cases = {
      IndexGreaterNumShardsCase(), NegativeIndexTestCase(),
      NegativeNumShardsTestCase(), ZeroNumShardsTestCase()};
  for (const auto& test_case : test_cases) {
    std::unique_ptr<OpKernel> shard_dataset_kernel;
    TF_ASSERT_OK(CreateShardDatasetOpKernel(
        test_case.require_non_empty, test_case.expected_output_dtypes,
        test_case.expected_output_shapes, &shard_dataset_kernel));

    DatasetBase* range_dataset;
    TF_ASSERT_OK(CreateRangeDataset<int64>(
        test_case.range_dataset_param.start, test_case.range_dataset_param.end,
        test_case.range_dataset_param.step, "range", &range_dataset));
    Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_ASSERT_OK(
        StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

    Tensor num_shards = test_case.num_shards;
    Tensor index = test_case.index;
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&range_dataset_tensor), TensorValue(&num_shards),
         TensorValue(&index)});
    std::unique_ptr<OpKernelContext> shard_dataset_context;
    TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                           &shard_dataset_context));

    DatasetBase* shard_dataset;
    EXPECT_EQ(CreateDataset(shard_dataset_kernel.get(),
                            shard_dataset_context.get(), &shard_dataset)
                  .code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

TEST_F(ShardDatasetOpTest, NoElemForEachShard) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TestCase test_case = NoElemForEachShardTestCase();

  std::unique_ptr<OpKernel> shard_dataset_kernel;
  TF_ASSERT_OK(CreateShardDatasetOpKernel(
      test_case.require_non_empty, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &shard_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor num_shards = test_case.num_shards;
  Tensor index = test_case.index;
  gtl::InlinedVector<TensorValue, 4> inputs({TensorValue(&range_dataset_tensor),
                                             TensorValue(&num_shards),
                                             TensorValue(&index)});
  std::unique_ptr<OpKernelContext> shard_dataset_context;
  TF_ASSERT_OK(CreateShardDatasetContext(shard_dataset_kernel.get(), &inputs,
                                         &shard_dataset_context));

  DatasetBase* shard_dataset;
  TF_ASSERT_OK(CreateDataset(shard_dataset_kernel.get(),
                             shard_dataset_context.get(), &shard_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      shard_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;

  EXPECT_EQ(
      iterator->GetNext(iterator_ctx.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
