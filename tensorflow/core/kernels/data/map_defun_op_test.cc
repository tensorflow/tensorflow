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
#include "tensorflow/core/kernels/data/map_defun_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "map_defun";
constexpr char kOpName[] = "MapDefun";

class MapDefunOpParams : public DatasetParams {
 public:
  MapDefunOpParams(std::vector<Tensor> arguments,
                   std::vector<Tensor> captured_inputs,
                   DataTypeVector type_arguments, DataTypeVector type_captured,
                   DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   int max_intra_op_parallelism, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        arguments_(std::move(arguments)),
        captured_inputs_(std::move(captured_inputs)),
        type_arguments_(std::move(type_arguments)),
        type_captured_(std::move(type_captured)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        max_intra_op_parallelism_(max_intra_op_parallelism) {}

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = arguments_;
    input_tensors.insert(input_tensors.end(), captured_inputs_.begin(),
                         captured_inputs_.end());
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();

    input_names->reserve(arguments_.size() + captured_inputs_.size());
    for (int i = 0; i < arguments_.size(); ++i) {
      input_names->emplace_back(
          strings::StrCat(MapDefunOp::kArguments, "_", i));
    }
    for (int i = 0; i < captured_inputs_.size(); ++i) {
      input_names->emplace_back(
          strings::StrCat(MapDefunOp::kCapturedInputs, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {
        {MapDefunOp::kTarguments, type_arguments_},
        {MapDefunOp::kTcaptured, type_captured_},
        {MapDefunOp::kOutputShapes, output_shapes_},
        {MapDefunOp::kOutputTypes, output_dtypes_},
        {MapDefunOp::kFunc, func_},
        {MapDefunOp::kMaxIntraOpParallelism, max_intra_op_parallelism_}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override { return "MapDef"; }

 private:
  std::vector<Tensor> arguments_;
  std::vector<Tensor> captured_inputs_;
  DataTypeVector type_arguments_;
  DataTypeVector type_captured_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  int max_intra_op_parallelism_;
};

class MapDefunOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `MapDefun` op kernel
  Status CreateMapDefunOpKernel(const MapDefunOpParams& params,
                                std::unique_ptr<OpKernel>* map_defun_kernel) {
    std::vector<string> input_namess;
    TF_RETURN_IF_ERROR(params.GetInputNames(&input_namess));
    AttributeVector attributes;
    TF_RETURN_IF_ERROR(params.GetAttributes(&attributes));

    NodeDef node_def =
        test::function::NDef(kNodeName, kOpName, input_namess, attributes);
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, map_defun_kernel));
    return Status::OK();
  }

  // Creates a new `MapDefun` op kernel context.
  Status CreateMapDefunContext(OpKernel* const op_kernel,
                               gtl::InlinedVector<TensorValue, 4>* const inputs,
                               std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  MapDefunOpParams map_defun_op_params;
  std::vector<Tensor> expected_outputs;
};

// Test case 1: one input for the map function with no captured inputs.
TestCase TestCase1() {
  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/{},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XTimesTwo()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/
          {CreateTensor<int64_t>(TensorShape({3, 2}), {0, 2, 4, 6, 8, 10})}};
}

// Test case 2: two inputs for the map function with no captured inputs.
TestCase TestCase2() {
  return {
      /*map_defun_op_params=*/
      MapDefunOpParams(
          /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                               {0, 1, 2, 3, 4, 5}),
                         CreateTensor<int64_t>(TensorShape({3, 2}),
                                               {0, 10, 20, 30, 40, 50})},
          /*captured_inputs=*/{},
          /*type_arguments=*/{DT_INT64, DT_INT64},
          /*type_captured=*/{},
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({2})},
          /*func=*/
          {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
          /*func_lib=*/{test::function::XAddY()},
          /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
      /*expected_outputs=*/
      {CreateTensor<int64_t>(TensorShape({3, 2}), {0, 11, 22, 33, 44, 55})}};
}

// Test case 3: two inputs for the map function with one captured input.
TestCase TestCase3() {
  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/
              {CreateTensor<int64_t>(TensorShape({2}), {10, 100})},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{DT_INT64},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/
          {CreateTensor<int64_t>(TensorShape({3, 2}),
                                 {10, 101, 12, 103, 14, 105})}};
}

TestCase InvalidOutputTypes() {
  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/
              {CreateTensor<int64_t>(TensorShape({2}), {10, 100})},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{DT_INT64},
              /*output_dtypes=*/{DT_FLOAT},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/{}};
}

TestCase InvalidOutputShapes() {
  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/
              {CreateTensor<int64_t>(TensorShape({2}), {10, 100})},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{DT_INT64},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2, 2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/{}};
}

TestCase InvalidInputs() {
  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5}),
                             CreateTensor<int64_t>(TensorShape({2, 2}),
                                                   {0, 1, 2, 3})},
              /*captured_inputs=*/{},
              /*type_arguments=*/{DT_INT64, DT_INT64},
              /*type_captured=*/{},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/{}};
}

class ParameterizedMapDefunOpTest
    : public MapDefunOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedMapDefunOpTest, NormalTests) {
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitializeRuntime(test_case.map_defun_op_params));
  auto input_tensors = test_case.map_defun_op_params.GetInputTensors();
  gtl::InlinedVector<TensorValue, 4> input_values;
  for (auto& input : input_tensors) {
    input_values.push_back(TensorValue(&input));
  }
  std::unique_ptr<OpKernel> map_defun_kernel;
  TF_ASSERT_OK(
      CreateMapDefunOpKernel(test_case.map_defun_op_params, &map_defun_kernel));
  std::unique_ptr<OpKernelContext> context;
  TF_ASSERT_OK(
      CreateMapDefunContext(map_defun_kernel.get(), &input_values, &context));
  TF_ASSERT_OK(RunOpKernel(map_defun_kernel.get(), context.get()));

  EXPECT_EQ(context->num_outputs(), test_case.expected_outputs.size());
  for (int i = 0; i < context->num_outputs(); ++i) {
    TF_EXPECT_OK(ExpectEqual(*context->mutable_output(i),
                             test_case.expected_outputs[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(MapDefunOpTest, ParameterizedMapDefunOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

TEST_F(MapDefunOpTest, InvalidArguments) {
  std::vector<TestCase> test_cases = {InvalidOutputTypes(),
                                      InvalidOutputShapes(), InvalidInputs()};
  for (auto& test_case : test_cases) {
    TF_ASSERT_OK(InitializeRuntime(test_case.map_defun_op_params));
    auto input_tensors = test_case.map_defun_op_params.GetInputTensors();
    gtl::InlinedVector<TensorValue, 4> input_values;
    for (auto& input : input_tensors) {
      input_values.push_back(TensorValue(&input));
    }
    std::unique_ptr<OpKernel> map_defun_kernel;
    TF_ASSERT_OK(CreateMapDefunOpKernel(test_case.map_defun_op_params,
                                        &map_defun_kernel));
    std::unique_ptr<OpKernelContext> context;
    TF_ASSERT_OK(
        CreateMapDefunContext(map_defun_kernel.get(), &input_values, &context));
    EXPECT_EQ(RunOpKernel(map_defun_kernel.get(), context.get()).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
