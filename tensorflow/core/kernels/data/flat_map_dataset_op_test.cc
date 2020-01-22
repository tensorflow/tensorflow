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
#include "tensorflow/core/kernels/data/flat_map_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "flat_map_dataset";

class FlatMapDatasetParams : public DatasetParams {
 public:
  template <typename T>
  FlatMapDatasetParams(T input_dataset_params,
                       std::vector<Tensor> other_arguments,
                       FunctionDefHelper::AttrValueWrapper func,
                       std::vector<FunctionDef> func_lib,
                       DataTypeVector type_arguments,
                       DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        func_(std::move(func)),
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
    input_names->emplace_back(FlatMapDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(FlatMapDatasetOp::kOtherArguments, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{FlatMapDatasetOp::kFunc, func_},
                    {FlatMapDatasetOp::kTarguments, type_arguments_},
                    {FlatMapDatasetOp::kOutputShapes, output_shapes_},
                    {FlatMapDatasetOp::kOutputTypes, output_dtypes_}};
    return Status::OK();
  }

  string dataset_type() const override {
    return FlatMapDatasetOp::kDatasetType;
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class FlatMapDatasetOpTest : public DatasetOpsTestBaseV2 {};

// Test case 1: normal case.
FlatMapDatasetParams FlatMapDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64>(TensorShape{3, 3, 1},
                                          {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  auto func = FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{"Toutput_types", DataTypeVector({DT_INT64})},
                 {"output_shapes",
                  std::vector<PartialTensorShape>({PartialTensorShape({1})})}});
  return FlatMapDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*func=*/func,
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 2: test the case if the function does not return a single scalar
// of dtype DT_VARIANT.
FlatMapDatasetParams InvalidFlatMapDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64>(TensorShape{3, 3, 1},
                                          {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  auto func = FunctionDefHelper::FunctionRef(/*name*/ "NonZero",
                                             /*attrs*/ {{"T", DT_INT64}});
  return FlatMapDatasetParams(std::move(tensor_slice_dataset_params),
                              /*other_arguments=*/{},
                              /*func=*/func,
                              /*func_lib=*/{test::function::NonZero()},
                              /*type_arguments=*/{},
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({1})},
                              /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<FlatMapDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/FlatMapDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape({1}),
                            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})}};
}

ITERATOR_GET_NEXT_TEST_P(FlatMapDatasetOpTest, FlatMapDatasetParams,
                         GetNextTestCases())

TEST_F(FlatMapDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(FlatMapDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(FlatMapDatasetOp::kDatasetType)));
}

TEST_F(FlatMapDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(FlatMapDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

TEST_F(FlatMapDatasetOpTest, Cardinality) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(FlatMapDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(FlatMapDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(FlatMapDatasetOpTest, IteratorPrefix) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      FlatMapDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<FlatMapDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/FlatMapDatasetParams1(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape({1}),
                            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(FlatMapDatasetOpTest, FlatMapDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(FlatMapDatasetOpTest, InvalidMapFunc) {
  auto dataset_params = InvalidFlatMapDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
