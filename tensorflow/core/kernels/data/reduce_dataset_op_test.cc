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

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "reduce_dataset";

class ReduceDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ReduceDatasetParams(T input_dataset_params, std::vector<Tensor> initial_state,
                      std::vector<Tensor> other_arguments,
                      FunctionDefHelper::AttrValueWrapper func,
                      std::vector<FunctionDef> func_lib,
                      DataTypeVector type_state, DataTypeVector type_arguments,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      bool use_inter_op_parallelism, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        initial_state_(std::move(initial_state)),
        other_arguments_(std::move(other_arguments)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_state_(std::move(type_state)),
        type_arguments_(std::move(type_arguments)),
        use_inter_op_parallelism_(use_inter_op_parallelism) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = initial_state_;
    input_tensors.insert(input_tensors.end(), other_arguments_.begin(),
                         other_arguments_.end());
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back("input_dataset");
    for (int i = 0; i < initial_state_.size(); ++i) {
      input_names->emplace_back(strings::StrCat("initial_state_", i));
    }
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(strings::StrCat("other_arguments_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    *attr_vector = {{"f", func_},
                    {"Tstate", type_state_},
                    {"Targuments", type_arguments_},
                    {"output_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"use_inter_op_parallelism", use_inter_op_parallelism_}};
    return Status::OK();
  }

  string dataset_type() const override { return "Reduce"; }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> initial_state_;
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_state_;
  DataTypeVector type_arguments_;
  bool use_inter_op_parallelism_;
};

class ReduceDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: the input function has one output.
ReduceDatasetParams ReduceDatasetParams1() {
  return ReduceDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*initial_state=*/CreateTensors<int64>(TensorShape({}), {{1}}),
      /*other_arguments=*/{},
      /*func=*/FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XAddY()},
      /*type_state=*/{DT_INT64},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*node_name=*/kNodeName);
}

// Test case 2: the reduce function has two inputs and two outputs. As the
// number of components of initial_state need to match with the reduce function
// outputs, the initial_state will have two components. It results in that
// the components of initial_state will be all the inputs for the reduce
// function, and the input dataset will not be involved in the
// reduce/aggregation process.
ReduceDatasetParams ReduceDatasetParams2() {
  return ReduceDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(1, 10, 1),
      /*initial_state=*/CreateTensors<int64>(TensorShape({}), {{1}, {1}}),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XPlusOneXTimesY", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XPlusOneXTimesY()},
      /*type_state=*/{DT_INT64, DT_INT64},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*node_name=*/kNodeName);
}

// Test case 3: the input dataset has no outputs, so the reduce dataset just
// returns the initial state.
ReduceDatasetParams ReduceDatasetParams3() {
  return ReduceDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 0, 1),
      /*initial_state=*/CreateTensors<int64>(TensorShape({}), {{1}, {3}}),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XAddY()},
      /*type_state=*/{DT_INT64, DT_INT64},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ReduceDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/
           ReduceDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{46}})},
          {/*dataset_params=*/ReduceDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}),
                                {{10}, {1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9}})},
          {/*dataset_params=*/
           ReduceDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape{}, {{1}, {3}})}};
}

class ParameterizedReduceDatasetOpTest
    : public ReduceDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<ReduceDatasetParams>> {};

TEST_P(ParameterizedReduceDatasetOpTest, Compute) {
  auto test_case = GetParam();
  TF_ASSERT_OK(InitializeRuntime(test_case.dataset_params));
  std::vector<Tensor> output;
  TF_ASSERT_OK(RunDatasetOp(test_case.dataset_params, &output));
  TF_EXPECT_OK(
      ExpectEqual(test_case.expected_outputs, output, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(ReduceDatasetOpTest, ParameterizedReduceDatasetOpTest,
                         ::testing::ValuesIn(GetNextTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
