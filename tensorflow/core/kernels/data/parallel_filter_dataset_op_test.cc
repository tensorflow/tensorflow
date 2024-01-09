/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/parallel_filter_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "parallel_map_dataset";

class ParallelFilterDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ParallelFilterDatasetParams(
      T input_dataset_params, std::vector<Tensor> other_arguments,
      int num_parallel_calls, const std::string& deterministic,
      FunctionDefHelper::AttrValueWrapper pred_func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      DataTypeVector output_dtypes,
      std::vector<PartialTensorShape> output_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        num_parallel_calls_(num_parallel_calls),
        deterministic_(deterministic),
        pred_func_(std::move(pred_func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    auto input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {num_parallel_calls_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->reserve(input_dataset_params_.size() +
                         other_arguments_.size());
    input_names->emplace_back(ParallelFilterDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(ParallelFilterDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(ParallelFilterDatasetOp::kNumParallelCalls);
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {
        {"predicate", pred_func_},         {"Targuments", type_arguments_},
        {"output_shapes", output_shapes_}, {"output_types", output_dtypes_},
        {"deterministic", deterministic_}, {"metadata", ""}};
    return OkStatus();
  }

  string dataset_type() const override {
    return ParallelFilterDatasetOp::kDatasetType;
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> other_arguments_;
  int num_parallel_calls_;
  std::string deterministic_;
  FunctionDefHelper::AttrValueWrapper pred_func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class ParallelFilterDatasetOpTest : public DatasetOpsTestBase {};

// num_parallel_calls = 1, deterministic = true
ParallelFilterDatasetParams ParallelFilterDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// num_parallel_calls = 1, deterministic = false
ParallelFilterDatasetParams ParallelFilterDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// num_parallel_calls = 2, deterministic = true
ParallelFilterDatasetParams ParallelFilterDatasetParams3() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/2,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// num_parallel_calls = 4, deterministic = true
ParallelFilterDatasetParams ParallelFilterDatasetParams4() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/4,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// num_parallel_calls = 4, deterministic = false
ParallelFilterDatasetParams ParallelFilterDatasetParams5() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/4,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// num_parallel_calls = kAutotune, deterministic = true
ParallelFilterDatasetParams ParallelFilterDatasetParams6() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/model::kAutotune,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// the input dataset has no outputs.
ParallelFilterDatasetParams InputHasNoElementParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{0}, {})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

// the filter function returns two outputs.
ParallelFilterDatasetParams InvalidPredFuncFilterDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("GetUnique",
                                     {{"T", DT_INT64}, {"out_idx", DT_INT32}}),
      /*func_lib*/ {test::function::Unique()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})},
      /*node_name=*/kNodeName);
}

// the filter function returns a 1-D bool tensor.
ParallelFilterDatasetParams InvalidPredFuncFilterDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})},
      /*node_name=*/kNodeName);
}

// the filter function returns a scalar int64 tensor.
ParallelFilterDatasetParams InvalidPredFuncFilterDatasetParams3() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return ParallelFilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("NonZero", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::NonZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ParallelFilterDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/ParallelFilterDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/ParallelFilterDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/ParallelFilterDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/ParallelFilterDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/ParallelFilterDatasetParams5(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/ParallelFilterDatasetParams6(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/InputHasNoElementParams(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(ParallelFilterDatasetOpTest,
                         ParallelFilterDatasetParams, GetNextTestCases())

TEST_F(ParallelFilterDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ParallelFilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ParallelFilterDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ParallelFilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ParallelFilterDatasetOp::kDatasetType)));
}

TEST_F(ParallelFilterDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ParallelFilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<ParallelFilterDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/ParallelFilterDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InputHasNoElementParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ParallelFilterDatasetOpTest,
                             ParallelFilterDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ParallelFilterDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/ParallelFilterDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InputHasNoElementParams(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(ParallelFilterDatasetOpTest,
                           ParallelFilterDatasetParams, CardinalityTestCases())

TEST_F(ParallelFilterDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ParallelFilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<ParallelFilterDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/ParallelFilterDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InputHasNoElementParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ParallelFilterDatasetOpTest,
                              ParallelFilterDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ParallelFilterDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ParallelFilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(ParallelFilterDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix())));
}

/*
TEST_F(ParallelFilterDatasetOpTest, InputOutOfRange) {
  auto dataset_params = InputOutOfRangeParams();
}
*/

std::vector<IteratorSaveAndRestoreTestCase<ParallelFilterDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ParallelFilterDatasetParams1(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/InputHasNoElementParams(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ParallelFilterDatasetOpTest,
                                 ParallelFilterDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

class ParameterizedInvalidPredicateFuncTest
    : public ParallelFilterDatasetOpTest,
      public ::testing::WithParamInterface<ParallelFilterDatasetParams> {};

TEST_P(ParameterizedInvalidPredicateFuncTest, InvalidPredicateFunc) {
  auto dataset_params = GetParam();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(out_tensors.empty());
}

INSTANTIATE_TEST_SUITE_P(
    ParallelFilterDatasetOpTest, ParameterizedInvalidPredicateFuncTest,
    ::testing::ValuesIn({InvalidPredFuncFilterDatasetParams1(),
                         InvalidPredFuncFilterDatasetParams2(),
                         InvalidPredFuncFilterDatasetParams3()}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
