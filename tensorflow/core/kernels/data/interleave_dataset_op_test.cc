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
#include "tensorflow/core/kernels/data/interleave_dataset_op.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"
#include "tensorflow/core/platform/test.h"
// IWYU pragma: begin_keep
#include "tensorflow/core/platform/test_benchmark.h"
// IWYU pragma: end_keep
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "interleave_dataset";

class InterleaveDatasetParams : public DatasetParams {
 public:
  template <typename T>
  InterleaveDatasetParams(T input_dataset_params,
                          std::vector<Tensor> other_arguments,
                          int64_t cycle_length, int64_t block_length,
                          FunctionDefHelper::AttrValueWrapper func,
                          std::vector<FunctionDef> func_lib,
                          DataTypeVector type_arguments,
                          DataTypeVector output_dtypes,
                          std::vector<PartialTensorShape> output_shapes,
                          std::string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {cycle_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {block_length_}));
    return input_tensors;
  }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    input_names->clear();
    input_names->reserve(input_dataset_params_.size() +
                         other_arguments_.size() + 2);
    input_names->emplace_back(InterleaveDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(InterleaveDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(InterleaveDatasetOp::kCycleLength);
    input_names->emplace_back(InterleaveDatasetOp::kBlockLength);
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{"f", func_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"metadata", ""}};
    return absl::OkStatus();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  std::string dataset_type() const override {
    return InterleaveDatasetOp::kDatasetType;
  }

 private:
  std::vector<Tensor> other_arguments_;
  int64_t cycle_length_;
  int64_t block_length_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class InterleaveDatasetOpTest : public DatasetOpsTestBase {
 protected:
  void TestBody() override {}
};

FunctionDefHelper::AttrValueWrapper MakeTensorSliceDatasetFunc(
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes) {
  return FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{"Toutput_types", output_types},
                 {"output_shapes", output_shapes}});
}

// test case 1: cycle_length = 1, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 2: cycle_length = 2, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 3: cycle_length = 3, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams3() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/3,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 4: cycle_length = 5, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams4() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/5,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 5: cycle_length = 2, block_length = 2.
InterleaveDatasetParams InterleaveDatasetParams5() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(TensorShape{3, 3, 1},
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 6: cycle_length = 2, block_length = 3.
InterleaveDatasetParams InterleaveDatasetParams6() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(TensorShape{3, 3, 1},
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/3,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 7: cycle_length = 2, block_length = 5.
InterleaveDatasetParams InterleaveDatasetParams7() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(TensorShape{3, 3, 1},
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/5,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

InterleaveDatasetParams InterleaveDatasetParamsWithBounds(
    int64_t cycle_length, int64_t block_length) {
  TensorSliceDatasetParams tensor_slice_dataset_params(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/cycle_length,
      /*block_length=*/block_length,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 8: cycle_length = 0, block_length = 5.
InterleaveDatasetParams InterleaveDatasetParamsWithInvalidCycleLength() {
  return InterleaveDatasetParamsWithBounds(/*cycle_length=*/0,
                                           /*block_length=*/5);
}

// Test case 9: cycle_length = 1, block_length = -1.
InterleaveDatasetParams InterleaveDatasetParamsWithInvalidBlockLength() {
  return InterleaveDatasetParamsWithBounds(/*cycle_length=*/1,
                                           /*block_length=*/-1);
}

// Test case 10: cycle_length = kMaxCycleOrBlockLength + 1, block_length = 5.
InterleaveDatasetParams InterleaveDatasetParamsWithCycleLengthOutOfBounds() {
  return InterleaveDatasetParamsWithBounds(
      /*cycle_length=*/InterleaveDatasetOp::kMaxCycleOrBlockLength + 1,
      /*block_length=*/5);
}

// Test case 11: cycle_length = 1, block_length = kMaxCycleOrBlockLength + 1.
InterleaveDatasetParams InterleaveDatasetParamsWithBlockLengthOutOfBounds() {
  return InterleaveDatasetParamsWithBounds(
      /*cycle_length=*/1,
      /*block_length=*/InterleaveDatasetOp::kMaxCycleOrBlockLength + 1);
}

std::vector<GetNextTestCase<InterleaveDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/InterleaveDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams2(),
       /*expected_outputs=*/CreateTensors<int64_t>(
           TensorShape({1}), {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams3(),
       /*expected_outputs=*/CreateTensors<int64_t>(
           TensorShape({1}), {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams4(),
       /*expected_outputs=*/CreateTensors<int64_t>(
           TensorShape({1}), {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams5(),
       /*expected_outputs=*/CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams6(),
       /*expected_outputs=*/CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams7(),
       /*expected_outputs=*/CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})}};
}

ITERATOR_GET_NEXT_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                         GetNextTestCases())

std::vector<SkipTestCase<InterleaveDatasetParams>> SkipTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 0, /*expected_num_skipped*/ 0, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}})},
          {/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{5}})},
          {/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{5}})},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{7}})},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{7}})},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*num_to_skip*/ 3, /*expected_num_skipped*/ 3, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"e"}})},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*num_to_skip*/ 3, /*expected_num_skipped*/ 3, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"d"}})},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*num_to_skip*/ 3, /*expected_num_skipped*/ 3, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"d"}})},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9}};
}

ITERATOR_SKIP_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                     SkipTestCases())

TEST_F(InterleaveDatasetOpTest, DatasetNodeName) {
  auto dataset_params = InterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(InterleaveDatasetOpTest, DatasetTypeString) {
  auto dataset_params = InterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(InterleaveDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<InterleaveDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_dtypes=*/{DT_STRING}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<InterleaveDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<InterleaveDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<InterleaveDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_dtypes=*/{DT_STRING}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<InterleaveDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(InterleaveDatasetOpTest, IteratorPrefix) {
  auto dataset_params = InterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      InterleaveDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<InterleaveDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/InterleaveDatasetParams1(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams2(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams3(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams4(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams5(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams6(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams7(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(InterleaveDatasetOpTest,
                                 InterleaveDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(InterleaveDatasetOpTest, InvalidCycleLength) {
  const InterleaveDatasetParams dataset_params =
      InterleaveDatasetParamsWithInvalidCycleLength();
  EXPECT_EQ(Initialize(dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(InterleaveDatasetOpTest, InvalidBlockLength) {
  const InterleaveDatasetParams dataset_params =
      InterleaveDatasetParamsWithInvalidBlockLength();
  EXPECT_EQ(Initialize(dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(InterleaveDatasetOpTest, CycleLengthOutOfBounds) {
  const InterleaveDatasetParams dataset_params =
      InterleaveDatasetParamsWithCycleLengthOutOfBounds();
  EXPECT_EQ(Initialize(dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(InterleaveDatasetOpTest, BlockLengthOutOfBounds) {
  const InterleaveDatasetParams dataset_params =
      InterleaveDatasetParamsWithBlockLengthOutOfBounds();
  EXPECT_EQ(Initialize(dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(InterleaveDatasetOpTest, SplitProvider) {
  auto params = InterleaveDatasetParams2();
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params,
      CreateTensors<int64_t>(TensorShape({1}),
                             {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/1,
      CreateTensors<int64_t>(TensorShape({1}), {{3}, {4}, {5}})));
}

TEST_F(InterleaveDatasetOpTest, SymbolicCheckpointSaveAndRestore) {
  auto dataset_params = InterleaveDatasetParams2();
  TF_ASSERT_OK(Initialize(dataset_params));

  IteratorContext::Params params(iterator_ctx_.get());
  params.symbolic_checkpoint = true;
  iterator_ctx_ = std::make_unique<IteratorContext>(std::move(params));

  TF_ASSERT_OK(dataset_->MakeIterator(iterator_ctx_.get(), /*parent=*/nullptr,
                                      dataset_params.iterator_prefix(),
                                      &iterator_));

  const std::vector<Tensor> expected_outputs = CreateTensors<int64_t>(
      TensorShape({1}), {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}});
  const std::vector<int> breakpoints = {0, 4, 7, 11};
  bool end_of_sequence = false;
  int cur_iteration = 0;
  std::vector<Tensor> out_tensors;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_ASSERT_OK(iterator_ctx_->checkpoint()->Save(&writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    // Restore into a fresh `IteratorContext` (whose `MemoryCheckpoint` is
    // empty) to mirror `IteratorResource::Restore`.
    iterator_ctx_ = std::make_unique<IteratorContext>(*iterator_ctx_);
    TF_ASSERT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 dataset_params.iterator_prefix(), *dataset_,
                                 &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_ASSERT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }
  TF_EXPECT_OK(
      ExpectEqual(out_tensors, expected_outputs, /*compare_order=*/true));
}

TEST_F(InterleaveDatasetOpTest, SymbolicCheckpointSkip) {
  auto dataset_params = InterleaveDatasetParams2();
  TF_ASSERT_OK(Initialize(dataset_params));

  IteratorContext::Params params(iterator_ctx_.get());
  params.symbolic_checkpoint = true;
  iterator_ctx_ = std::make_unique<IteratorContext>(std::move(params));
  TF_ASSERT_OK(dataset_->MakeIterator(iterator_ctx_.get(), /*parent=*/nullptr,
                                      dataset_params.iterator_prefix(),
                                      &iterator_));
  // Skipping 7 elements exhausts the first two input elements inside
  // `SkipInternal` (which must purge their sub-iterator state from
  // `iterator_ctx_->checkpoint()` and propagate the input iterator position
  // recorded when the second input element was opened) and opens the third
  // input element.
  bool end_of_sequence = false;
  int num_skipped = 0;
  TF_ASSERT_OK(iterator_->Skip(iterator_ctx_.get(), /*num_to_skip=*/7,
                               &end_of_sequence, &num_skipped));
  EXPECT_FALSE(end_of_sequence);
  EXPECT_EQ(num_skipped, 7);
  {
    VariantTensorDataWriter skip_writer;
    TF_ASSERT_OK(iterator_ctx_->checkpoint()->Save(&skip_writer));
    std::vector<const VariantTensorData*> skip_data;
    skip_writer.GetData(&skip_data);
    VariantTensorDataReader skip_reader(skip_data);
    auto sub_iterator_index_key = [&](int input_element_index) {
      return FullName(
          name_utils::IteratorPrefix(
              TensorSliceDatasetOp::kDatasetType,
              absl::StrCat(iterator_->prefix(), "[", input_element_index, "]")),
          "index");
    };
    EXPECT_FALSE(skip_reader.Contains(sub_iterator_index_key(0)));
    EXPECT_FALSE(skip_reader.Contains(sub_iterator_index_key(1)));
    EXPECT_TRUE(skip_reader.Contains(sub_iterator_index_key(2)));
    int64_t input_index = -1;
    TF_ASSERT_OK(skip_reader.ReadScalar(
        FullName(name_utils::IteratorPrefix(TensorSliceDatasetOp::kDatasetType,
                                            iterator_->prefix()),
                 "index"),
        &input_index));
    EXPECT_EQ(input_index, 2);
  }

  // `DatasetBaseIterator::Skip` does not call `IteratorContext::SaveCheckpoint`
  // (only `GetNext`, `MakeIterator`, and `Restore` do), so call `GetNext` once
  // so the remaining open sub-iterator saves its state before the symbolic
  // checkpoint is saved.
  std::vector<Tensor> out_tensors;
  TF_ASSERT_OK(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence));
  EXPECT_FALSE(end_of_sequence);
  TF_EXPECT_OK(ExpectEqual(out_tensors,
                           CreateTensors<int64_t>(TensorShape({1}), {{7}}),
                           /*compare_order=*/true));

  VariantTensorDataWriter writer;
  TF_ASSERT_OK(iterator_ctx_->checkpoint()->Save(&writer));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);
  VariantTensorDataReader reader(data);
  // Restore into a fresh `IteratorContext` (whose `MemoryCheckpoint` is empty)
  // to mirror `IteratorResource::Restore`.
  iterator_ctx_ = std::make_unique<IteratorContext>(*iterator_ctx_);
  TF_ASSERT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                               dataset_params.iterator_prefix(), *dataset_,
                               &iterator_));
  TF_EXPECT_OK(
      CheckIteratorGetNext(CreateTensors<int64_t>(TensorShape({1}), {{8}}),
                           /*compare_order=*/true));
}

InterleaveDatasetParams InterleaveBenchmarkDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{100, 10, 1})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// `test_benchmark.h` provides `BENCHMARK`, `benchmark::State`, and
// `benchmark::DoNotOptimize` without re-exporting the internal headers that
// define them (`registration.h`, `state.h`, `utils.h`), which do not exist in
// the benchmark release that open-source TensorFlow builds with.
// NOLINTBEGIN(misc-include-cleaner)
void BM_InterleaveDatasetGetNext(benchmark::State& state) {
  auto dataset_params = InterleaveBenchmarkDatasetParams();
  InterleaveDatasetOpTest test_base;
  TF_CHECK_OK(test_base.InitializeRuntime(dataset_params));
  std::unique_ptr<TestDataset> dataset;
  TF_CHECK_OK(test_base.MakeDataset(dataset_params, &dataset));
  std::unique_ptr<TestIterator> iterator;
  std::vector<Tensor> out_tensors;
  for (auto _ : state) {
    TF_CHECK_OK(test_base.MakeIterator(dataset_params, *dataset, &iterator));
    bool end_of_sequence = false;
    while (!end_of_sequence) {
      TF_CHECK_OK(iterator->GetNext(&out_tensors, &end_of_sequence));
      benchmark::DoNotOptimize(out_tensors);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_InterleaveDatasetGetNext);

void BM_InterleaveDatasetSkip(benchmark::State& state) {
  auto dataset_params = InterleaveBenchmarkDatasetParams();
  InterleaveDatasetOpTest test_base;
  TF_CHECK_OK(test_base.InitializeRuntime(dataset_params));
  std::unique_ptr<TestDataset> dataset;
  TF_CHECK_OK(test_base.MakeDataset(dataset_params, &dataset));
  std::unique_ptr<TestIterator> iterator;
  for (auto _ : state) {
    TF_CHECK_OK(test_base.MakeIterator(dataset_params, *dataset, &iterator));
    bool end_of_sequence = false;
    while (!end_of_sequence) {
      int num_skipped = 0;
      TF_CHECK_OK(iterator->iterator()->Skip(iterator->ctx(), /*num_to_skip=*/1,
                                             &end_of_sequence, &num_skipped));
      benchmark::DoNotOptimize(num_skipped);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_InterleaveDatasetSkip);
// NOLINTEND(misc-include-cleaner)

}  // namespace
}  // namespace data
}  // namespace tensorflow
