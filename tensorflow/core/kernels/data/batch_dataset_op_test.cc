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

constexpr char kNodeName[] = "batch_dataset_v2";
constexpr char kOpName[] = "BatchDatasetV2";

class BatchDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `BatchDataset` op kernel.
  Status CreateBatchDatasetOpKernel(
      bool parallel_copy, const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* batch_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "batch_size", "drop_remainder"},
        {{"parallel_copy", parallel_copy},
         {"output_types", output_types},
         {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, batch_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `BatchDataset` op kernel context
  Status CreateBatchDatasetContext(
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
  Tensor batch_size;
  Tensor drop_remainder;
  bool parallel_copy;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test Case 1: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can evenly split the input dataset.
TestCase TestCase1() {
  return {
      /*range_data_param*/ {0, 12, 1},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
      /*parallel_copy*/ true,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({4}), {0, 1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({4}), {4, 5, 6, 7}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({4}),
                                               {8, 9, 10, 11})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({4})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 1, 5}};
}

// Test Case 2: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can evenly split the input dataset.
TestCase TestCase2() {
  return {
      /*range_data_param*/ {0, 12, 1},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*parallel_copy*/ false,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({4}), {0, 1, 2, 3}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({4}), {4, 5, 6, 7}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({4}),
                                               {8, 9, 10, 11})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({4})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 1, 5}};
}

// Test Case 3: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can not evenly split the input dataset.
TestCase TestCase3() {
  return {/*range_data_param*/ {0, 10, 1},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
          /*parallel_copy*/ false,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1}), {9})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 4: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can not evenly split the input dataset.
TestCase TestCase4() {
  return {
      /*range_data_param*/ {0, 10, 1},
      /*batch_size*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*drop_remainder*/
      DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
      /*parallel_copy*/ true,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({3}), {6, 7, 8})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({3})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 1, 5}};
}

// Test Case 5: test BatchDatasetV2 with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
TestCase TestCase5() {
  return {/*range_data_param*/ {0, 10, 1},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {true}),
          /*parallel_copy*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({12})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 6: test BatchDatasetV2 with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
TestCase TestCase6() {
  return {/*range_data_param*/ {0, 10, 1},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
          /*parallel_copy*/ true,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(
              TensorShape({10}), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({-1})},
          /*expected_cardinality*/ 1,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 7: test BatchDatasetV2 with `drop_remainder` = false and
// the output of the input dataset is empty.
TestCase TestCase7() {
  return {/*range_data_param*/ {0, 0, 1},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
          /*parallel_copy*/ false,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({4})},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {0, 1, 5}};
}

// Test Case 8: test BatchDatasetV2 with an invalid batch size
TestCase InvalidBatchSizeTestCase() {
  return {/*range_data_param*/ {0, 10, 1},
          /*batch_size*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-1}),
          /*drop_remainder*/
          DatasetOpsTestBase::CreateTensor<bool>(TensorShape({}), {false}),
          /*parallel_copy*/ false,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({3})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 1, 5}};
}

class ParameterizedBatchDatasetOpTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedBatchDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      batch_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

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

TEST_P(ParameterizedBatchDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  EXPECT_EQ(batch_dataset->node_name(), kNodeName);
}

TEST_P(ParameterizedBatchDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  EXPECT_EQ(batch_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedBatchDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(batch_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedBatchDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(batch_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedBatchDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  EXPECT_EQ(batch_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedBatchDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(batch_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedBatchDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      batch_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedBatchDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      batch_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedBatchDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      batch_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Batch");
}

TEST_P(ParameterizedBatchDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref_batch_dataset(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      batch_dataset->MakeIterator(iterator_ctx.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, "Iterator",
                                 *batch_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_ctx.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
        expected_outputs_it++;
      }
      cur_iteration++;
    }

    if (breakpoint >= test_case.expected_cardinality) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(BatchDatasetOpTest, ParameterizedBatchDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6(),
                              TestCase7()})));

TEST_F(BatchDatasetOpTest, InvalidBatchSize) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TestCase test_case = InvalidBatchSizeTestCase();
  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.parallel_copy, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &batch_dataset_kernel));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_dataset_param.start, test_case.range_dataset_param.end,
      test_case.range_dataset_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));

  Tensor batch_size = test_case.batch_size;
  Tensor drop_remainder = test_case.drop_remainder;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&range_dataset_tensor),
                                            TensorValue(&batch_size),
                                            TensorValue(&drop_remainder)};
  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  EXPECT_EQ(CreateDataset(batch_dataset_kernel.get(),
                          batch_dataset_context.get(), &batch_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
