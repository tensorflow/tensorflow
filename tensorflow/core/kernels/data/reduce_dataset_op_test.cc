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

constexpr char kNodeName[] = "reduce_dataset";
constexpr char kOpName[] = "ReduceDataset";

class ReduceDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Create a new `ReduceDataset` op kernel.
  Status CreateReduceDatasetOpKernel(
      const FunctionDefHelper::AttrValueWrapper &func,
      const DataTypeVector &t_state, const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      bool use_inter_op_parallelism,
      std::unique_ptr<OpKernel> *reduce_dataset_op_kernel) {
    std::vector<string> components;
    components.reserve(1 + t_state.size());
    components.emplace_back("input_dataset");
    for (int i = 0; i < t_state.size(); ++i) {
      components.emplace_back(strings::StrCat("initial_state_", i));
    }
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, components,
        {{"f", func},
         {"Tstate", t_state},
         {"Targuments", {}},
         {"output_types", output_types},
         {"output_shapes", output_shapes},
         {"use_inter_op_parallelism", use_inter_op_parallelism}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, reduce_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `ReduceDataset` op kernel context
  Status CreateReduceDatasetContext(
      OpKernel *const op_kernel,
      gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
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
  std::vector<Tensor> initial_state;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  DataTypeVector t_state;
  bool use_inter_op_parallelism;
  std::vector<Tensor> expected_outputs;
  DataTypeVector output_dtypes;
  std::vector<PartialTensorShape> output_shapes;
};

// Test case 1: the input function has one output.
TestCase TestCase1() {
  return {/*range_data_param*/ {0, 10, 1},
          /*initial_state*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0})},
          /*func*/
          FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::XAddY()},
          /*t_state*/ {DT_INT64},
          /*use_inter_op_parallelism*/ true,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {45})},
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})}};
}

// Test case 2: the input function has two outputs.
TestCase TestCase2() {
  return {/*range_data_param*/ {0, 10, 1},
          /*initial_state*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0})},
          /*func*/
          FunctionDefHelper::FunctionRef("XPlusOneXTimesY", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::XPlusOneXTimesY()},
          /*t_state*/ {DT_INT64},
          /*use_inter_op_parallelism*/ true,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0})},
          /*output_dtypes*/ {DT_INT64, DT_INT64},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})}};
}

// Test case 3: the input dataset has no outputs, so the reduce dataset just
// returns the initial state.
TestCase TestCase3() {
  return {/*range_data_param*/ {0, 0, 1},
          /*initial_state*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3})},
          /*func*/
          FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::XAddY()},
          /*t_state*/ {DT_INT64, DT_INT64},
          /*use_inter_op_parallelism*/ true,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3})},
          /*output_dtypes*/ {DT_INT64, DT_INT64},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})}};
}

class ParameterizedReduceDatasetOpTest
    : public ReduceDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

// TODO(feighuis): Re-enable this test.
TEST_P(ParameterizedReduceDatasetOpTest, DISABLED_Compute) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> reduce_dataset_kernel;
  TF_ASSERT_OK(CreateReduceDatasetOpKernel(
      test_case.func, test_case.t_state, test_case.output_dtypes,
      test_case.output_shapes, test_case.use_inter_op_parallelism,
      &reduce_dataset_kernel));

  DatasetBase *range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  std::vector<Tensor> initial_state = test_case.initial_state;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_dataset_tensor)});
  for (auto &t : initial_state) {
    inputs.emplace_back(&t);
  }

  std::unique_ptr<OpKernelContext> reduce_dataset_context;
  TF_ASSERT_OK(CreateReduceDatasetContext(reduce_dataset_kernel.get(), &inputs,
                                          &reduce_dataset_context));
  TF_ASSERT_OK(
      RunOpKernel(reduce_dataset_kernel.get(), reduce_dataset_context.get()));

  int num_outputs = reduce_dataset_context->num_outputs();
  EXPECT_EQ(num_outputs, test_case.expected_outputs.size());
  for (int i = 0; i < num_outputs; i++) {
    // output will be released by the op kernel context.
    Tensor *output = reduce_dataset_context->mutable_output(i);
    TF_EXPECT_OK(ExpectEqual(test_case.expected_outputs[i], *output));
  }
}

INSTANTIATE_TEST_SUITE_P(ReduceDatasetOpTest, ParameterizedReduceDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
