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
#include "tensorflow/core/kernels/data/filter_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "filter_dataset";

class FilterDatasetParams : public DatasetParams {
 public:
  template <typename T>
  FilterDatasetParams(T input_dataset_params,
                      std::vector<Tensor> other_arguments,
                      FunctionDefHelper::AttrValueWrapper pred_func,
                      std::vector<FunctionDef> func_lib,
                      DataTypeVector type_arguments,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        pred_func_(std::move(pred_func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return other_arguments_;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->reserve(input_dataset_params_.size() +
                         other_arguments_.size());
    input_names->emplace_back(FilterDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(FilterDatasetOp::kOtherArguments, "_", i));
    }

    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{FilterDatasetOp::kPredicate, pred_func_},
                    {FilterDatasetOp::kTarguments, type_arguments_},
                    {FilterDatasetOp::kOutputShapes, output_shapes_},
                    {FilterDatasetOp::kOutputTypes, output_dtypes_}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override { return FilterDatasetOp::kDatasetType; }

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper pred_func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class FilterDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: norm case.
FilterDatasetParams FilterDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 2: the input dataset has no outputs.
FilterDatasetParams FilterDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{0}, {})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

// Test case 3: the filter function returns two outputs.
FilterDatasetParams InvalidPredFuncFilterDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{3, 3}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("GetUnique",
                                     {{"T", DT_INT64}, {"out_idx", DT_INT32}}),
      /*func_lib*/ {test::function::Unique()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})},
      /*node_name=*/kNodeName);
}

// Test case 4: the filter function returns a 1-D bool tensor.
FilterDatasetParams InvalidPredFuncFilterDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})},
      /*node_name=*/kNodeName);
}

// Test case 5: the filter function returns a scalar int64 tensor.
FilterDatasetParams InvalidPredFuncFilterDatasetParams3() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{9}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("NonZero", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::NonZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<FilterDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                         GetNextTestCases())

TEST_F(FilterDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(FilterDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(FilterDatasetOp::kDatasetType)));
}

TEST_F(FilterDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<FilterDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<FilterDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                           CardinalityTestCases())

TEST_F(FilterDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<FilterDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(FilterDatasetOpTest, IteratorPrefix) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      FilterDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<FilterDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

class ParameterizedInvalidPredicateFuncTest
    : public FilterDatasetOpTest,
      public ::testing::WithParamInterface<FilterDatasetParams> {};

TEST_P(ParameterizedInvalidPredicateFuncTest, InvalidPredicateFunc) {
  auto dataset_params = GetParam();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
  EXPECT_TRUE(out_tensors.empty());
}

INSTANTIATE_TEST_SUITE_P(
    FilterDatasetOpTest, ParameterizedInvalidPredicateFuncTest,
    ::testing::ValuesIn({InvalidPredFuncFilterDatasetParams1(),
                         InvalidPredFuncFilterDatasetParams2(),
                         InvalidPredFuncFilterDatasetParams3()}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
