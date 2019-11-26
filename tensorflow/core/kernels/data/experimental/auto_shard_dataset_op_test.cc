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
#include "tensorflow/core/kernels/data/experimental/auto_shard_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/shard_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "auto_shard_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

class AutoShardDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `AutoShardDataset` op kernel.
  Status CreateAutoShardDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(AutoShardDatasetOp::kDatasetType),
        {AutoShardDatasetOp::kInputDataset, AutoShardDatasetOp::kNumWorkers,
         AutoShardDatasetOp::kIndex},
        {{AutoShardDatasetOp::kAutoShardPolicy, 0},  // AutoShardPolicy == AUTO
         {AutoShardDatasetOp::kOutputTypes, output_types},
         {AutoShardDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Create a new `AutoShardDataset` op kernel context
  Status CreateAutoShardDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  TestCase(int64 start, int64 stop, int64 step, int64 num_workers, int64 index,
           std::vector<Tensor> expected_outputs,
           DataTypeVector expected_output_dtypes,
           std::vector<PartialTensorShape> expected_output_shapes,
           int64 expected_cardinality, std::vector<int> breakpoints)
      : start(CreateTensor<int64>(TensorShape({}), {start})),
        stop(CreateTensor<int64>(TensorShape({}), {stop})),
        step(CreateTensor<int64>(TensorShape({}), {step})),
        num_workers(CreateTensor<int64>(TensorShape({}), {num_workers})),
        index(CreateTensor<int64>(TensorShape({}), {index})),
        expected_outputs(std::move(expected_outputs)),
        expected_output_dtypes(std::move(expected_output_dtypes)),
        expected_output_shapes(std::move(expected_output_shapes)),
        expected_cardinality(expected_cardinality),
        breakpoints(std::move(breakpoints)) {}

  Tensor start;
  Tensor stop;
  Tensor step;
  Tensor num_workers;
  Tensor index;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test Case 1: simple case.
TestCase SimpleCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*num_workers=*/5,
          /*index=*/2,
          /*expected_outputs=*/
          {CreateTensor<int64>(TensorShape({}), {2}),
           CreateTensor<int64>(TensorShape({}), {7})},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/2,
          /*breakpoints=*/{0, 1, 5}};
}

// Test Case 2: the index is larger than the available elements.
TestCase IndexLargerThanAvailableElementsCase() {
  return {/*start=*/0,
          /*stop=*/1,
          /*step=*/1,
          /*num_workers=*/5,
          /*index=*/2,
          /*expected_outputs=*/{},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/2,
          /*breakpoints=*/{0, 1}};
}

// Test Case 3: the number of outputs could not be evenly divided by
// num_workers.
TestCase ElementsUnequallyDividedCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*num_workers=*/4,
          /*index=*/3,
          /*expected_outputs=*/
          {CreateTensor<int64>(TensorShape({}), {3}),
           CreateTensor<int64>(TensorShape({}), {7})},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/2,
          /*breakpoints=*/{0, 1, 5}};
}

// TODO(feihugis): add more test cases that have ReaderDatasets (e.g. a
// CSVDataset or a TFRecordDataset) in the pipeline.

TestCase IndexGreaterNumWorkersCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*num_workers=*/5,
          /*index=*/7,
          /*expected_outputs=*/{},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/0,
          /*breakpoints=*/{}};
}

TestCase NegativeIndexTestCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*num_workers=*/5,
          /*index=*/-3,
          /*expected_outputs=*/{},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/0,
          /*breakpoints=*/{}};
}

TestCase NegativeNumWorkersTestCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*num_workers=*/-3,
          /*index=*/1,
          /*expected_outputs=*/{},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/0,
          /*breakpoints=*/{}};
}

TestCase ZeroNumWorkersTestCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*num_workers=*/0,
          /*index=*/1,
          /*expected_outputs=*/{},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({})},
          /*expected_cardinality=*/0,
          /*breakpoints=*/{}};
}

class ParameterizedAutoShardDatasetOpTest
    : public AutoShardDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedAutoShardDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> auto_shard_dataset_kernel;
  TF_ASSERT_OK(CreateAutoShardDatasetOpKernel(test_case.expected_output_dtypes,
                                              test_case.expected_output_shapes,
                                              &auto_shard_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.num_workers),
       TensorValue(&test_case.index)});
  std::unique_ptr<OpKernelContext> auto_shard_dataset_context;
  TF_ASSERT_OK(CreateAutoShardDatasetContext(
      auto_shard_dataset_kernel.get(), &inputs, &auto_shard_dataset_context));

  DatasetBase* auto_shard_dataset;
  TF_ASSERT_OK(CreateDataset(auto_shard_dataset_kernel.get(),
                             auto_shard_dataset_context.get(),
                             &auto_shard_dataset));
  core::ScopedUnref scoped_unref_auto_shard_dataset(auto_shard_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(auto_shard_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(auto_shard_dataset->MakeIterator(iterator_ctx.get(),
                                                kIteratorPrefix, &iterator));

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

INSTANTIATE_TEST_SUITE_P(AutoShardDatasetOpTest,
                         ParameterizedAutoShardDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {SimpleCase(),
                              IndexLargerThanAvailableElementsCase(),
                              ElementsUnequallyDividedCase()})));

TEST_F(AutoShardDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<TestCase> test_cases = {
      IndexGreaterNumWorkersCase(), NegativeIndexTestCase(),
      NegativeNumWorkersTestCase(), ZeroNumWorkersTestCase()};
  for (auto& test_case : test_cases) {
    std::unique_ptr<OpKernel> auto_shard_dataset_kernel;
    TF_ASSERT_OK(CreateAutoShardDatasetOpKernel(
        test_case.expected_output_dtypes, test_case.expected_output_shapes,
        &auto_shard_dataset_kernel));

    Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop,
                                  test_case.step, {DT_INT64}, {TensorShape({})},
                                  &range_dataset_tensor));
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&range_dataset_tensor),
         TensorValue(&test_case.num_workers), TensorValue(&test_case.index)});
    std::unique_ptr<OpKernelContext> auto_shard_dataset_context;
    TF_ASSERT_OK(CreateAutoShardDatasetContext(
        auto_shard_dataset_kernel.get(), &inputs, &auto_shard_dataset_context));

    DatasetBase* auto_shard_dataset;
    EXPECT_EQ(
        CreateDataset(auto_shard_dataset_kernel.get(),
                      auto_shard_dataset_context.get(), &auto_shard_dataset)
            .code(),
        tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
