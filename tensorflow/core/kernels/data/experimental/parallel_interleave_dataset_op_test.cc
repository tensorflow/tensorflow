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
#include "tensorflow/core/kernels/data/experimental/parallel_interleave_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "parallel_interleave_dataset";
constexpr int kOpVersion = 2;

class ParallelInterleaveDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ParallelInterleaveDatasetParams(
      T input_dataset_params, std::vector<Tensor> other_arguments,
      int64_t cycle_length, int64_t block_length,
      const std::string& deterministic, int64_t buffer_output_elements,
      int64_t prefetch_input_elements, FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        deterministic_(deterministic),
        buffer_output_elements_(buffer_output_elements),
        prefetch_input_elements_(prefetch_input_elements),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    op_version_ = kOpVersion;
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    iterator_prefix_ = name_utils::IteratorPrefix(
        input_dataset_params.dataset_type(),
        input_dataset_params.iterator_prefix(), params);
  }

  std::vector<Tensor> GetInputTensors() const override {
    auto input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {cycle_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {block_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {buffer_output_elements_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {prefetch_input_elements_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->emplace_back(ParallelInterleaveDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(ParallelInterleaveDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(ParallelInterleaveDatasetOp::kCycleLength);
    input_names->emplace_back(ParallelInterleaveDatasetOp::kBlockLength);
    input_names->emplace_back(
        ParallelInterleaveDatasetOp::kBufferOutputElements);
    input_names->emplace_back(
        ParallelInterleaveDatasetOp::kPrefetchInputElements);
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{"f", func_},
                    {"deterministic", deterministic_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"metadata", ""}};
    return OkStatus();
  }

  string dataset_type() const override {
    return ParallelInterleaveDatasetOp::kDatasetType;
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> other_arguments_;
  int64_t cycle_length_;
  int64_t block_length_;
  std::string deterministic_;
  int64_t buffer_output_elements_;
  int64_t prefetch_input_elements_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class ParallelInterleaveDatasetOpTest : public DatasetOpsTestBase {};

FunctionDefHelper::AttrValueWrapper MakeTensorSliceDatasetFunc(
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes) {
  return FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{TensorSliceDatasetOp::kToutputTypes, output_types},
                 {TensorSliceDatasetOp::kOutputShapes, output_shapes}});
}

// Test case 1: cycle_length = 1, block_length = 1, deterministic = true,
// buffer_output_elements = 1, prefetch_input_elements = 1.
ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/1,
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

// Test case 2: cycle_length = 2, block_length = 1, deterministic = true,
// buffer_output_elements = 1, prefetch_input_elements = 0.
ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/0,
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

// Test case 3: cycle_length = 3, block_length = 1, deterministic = false,
// buffer_output_elements = 3, prefetch_input_elements = 2.
ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams3() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/3,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*buffer_output_elements=*/3,
      /*prefetch_input_elements=*/2,
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

// Test case 4: cycle_length = 5, block_length = 1, deterministic = false
// buffer_output_elements = 1, prefetch_input_elements = 2.
ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams4() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/5,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/2,
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

// Test case 5: cycle_length = 2, block_length = 2, deterministic = true
// buffer_output_elements = 2, prefetch_input_elements = 2.
ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams5() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/2,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/2,
      /*prefetch_input_elements=*/2,
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

ParallelInterleaveDatasetParams EmptyInputParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{Tensor{}},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/2,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*buffer_output_elements=*/2,
      /*prefetch_input_elements=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_FLOAT}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_FLOAT},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams InvalidCycleLengthParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/0,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/1,
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

ParallelInterleaveDatasetParams InvalidBlockLengthParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/-1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/1,
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

ParallelInterleaveDatasetParams InvalidBufferOutputElementsParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/0,
      /*prefetch_input_elements=*/1,
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

ParallelInterleaveDatasetParams InvalidPrefetchInputElementsParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/-1,
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

std::vector<GetNextTestCase<ParallelInterleaveDatasetParams>>
GetNextTestCases() {
  return {{/*dataset_params=*/ParallelInterleaveDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams5(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/EmptyInputParams(),
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape{1}, {}),
           /*compare_order=*/true}};
}

ITERATOR_GET_NEXT_TEST_P(ParallelInterleaveDatasetOpTest,
                         ParallelInterleaveDatasetParams, GetNextTestCases())

TEST_F(ParallelInterleaveDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ParallelInterleaveDatasetOp::kDatasetType, params)));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({1})}));
}

std::vector<CardinalityTestCase<ParallelInterleaveDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/ParallelInterleaveDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams3(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams4(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams5(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(ParallelInterleaveDatasetOpTest,
                           ParallelInterleaveDatasetParams,
                           CardinalityTestCases())

TEST_F(ParallelInterleaveDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ParallelInterleaveDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({1})}));
}

TEST_F(ParallelInterleaveDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(ParallelInterleaveDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix(), params)));
}

std::vector<IteratorSaveAndRestoreTestCase<ParallelInterleaveDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ParallelInterleaveDatasetParams1(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams2(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams3(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams4(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams5(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ParallelInterleaveDatasetOpTest,
                                 ParallelInterleaveDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(ParallelInterleaveDatasetOpTest, InvalidArguments) {
  std::vector<ParallelInterleaveDatasetParams> invalid_params = {
      InvalidCycleLengthParams(), InvalidBlockLengthParams(),
      InvalidBufferOutputElementsParams(),
      InvalidPrefetchInputElementsParams()};
  for (auto& dataset_params : invalid_params) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
