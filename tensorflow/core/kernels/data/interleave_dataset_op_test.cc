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

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "interleave_dataset";

class InterleaveDatasetParams : public DatasetParams {
 public:
  template <typename T>
  InterleaveDatasetParams(T input_dataset_params,
                          std::vector<Tensor> other_arguments,
                          int64 cycle_length, int64 block_length,
                          FunctionDefHelper::AttrValueWrapper func,
                          std::vector<FunctionDef> func_lib,
                          DataTypeVector type_arguments,
                          DataTypeVector output_dtypes,
                          std::vector<PartialTensorShape> output_shapes,
                          string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64>(TensorShape({}), {cycle_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64>(TensorShape({}), {block_length_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
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
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{InterleaveDatasetOp::kFunc, func_},
                    {InterleaveDatasetOp::kTarguments, type_arguments_},
                    {InterleaveDatasetOp::kOutputShapes, output_shapes_},
                    {InterleaveDatasetOp::kOutputTypes, output_dtypes_}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override {
    return InterleaveDatasetOp::kDatasetType;
  }

 private:
  std::vector<Tensor> other_arguments_;
  int64 cycle_length_;
  int64 block_length_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class InterleaveDatasetOpTest : public DatasetOpsTestBase {};

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
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
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
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
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
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
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
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
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

// test case 8: cycle_length = 0, block_length = 5.
InterleaveDatasetParams InterleaveDatasetParamsWithInvalidCycleLength() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/0,
      /*block_length=*/5,
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

// test case 9: cycle_length = 1, block_length = -1.
InterleaveDatasetParams InterleaveDatasetParamsWithInvalidBlockLength() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64>(TensorShape{3, 3, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/-1,
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

std::vector<GetNextTestCase<InterleaveDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/InterleaveDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape({1}),
                            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams2(),
       /*expected_outputs=*/CreateTensors<int64>(
           TensorShape({1}), {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams3(),
       /*expected_outputs=*/CreateTensors<int64>(
           TensorShape({1}), {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams4(),
       /*expected_outputs=*/CreateTensors<int64>(
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
           CreateTensors<int64>(TensorShape({1}), {{0}})},
          {/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({1}), {{5}})},
          {/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({1}), {{5}})},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({1}), {{7}})},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({1}), {{7}})},
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
       CreateTensors<int64>(TensorShape({1}),
                            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams2(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape({1}),
                            {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams3(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape({1}),
                            {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams4(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape({1}),
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
  auto dataset_params = InterleaveDatasetParamsWithInvalidCycleLength();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(InterleaveDatasetOpTest, InvalidLength) {
  auto dataset_params = InterleaveDatasetParamsWithInvalidBlockLength();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
