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
#include "tensorflow/core/kernels/data/experimental/map_and_batch_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "map_and_batch_dataset";

class MapAndBatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  MapAndBatchDatasetParams(
      T input_dataset_params, std::vector<Tensor> other_arguments,
      int64 batch_size, int64 num_parallel_calls, bool drop_remainder,
      FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      bool preserve_cardinality, DataTypeVector output_dtypes,
      std::vector<PartialTensorShape> output_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        batch_size_(batch_size),
        num_parallel_calls_(num_parallel_calls),
        drop_remainder_(drop_remainder),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)),
        preserve_cardinality_(preserve_cardinality) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> inputs = other_arguments_;
    inputs.emplace_back(CreateTensor<int64>(TensorShape({}), {batch_size_}));
    inputs.emplace_back(
        CreateTensor<int64>(TensorShape({}), {num_parallel_calls_}));
    inputs.emplace_back(CreateTensor<bool>(TensorShape({}), {drop_remainder_}));
    return inputs;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->reserve(input_dataset_params_.size() +
                         other_arguments_.size() + 3);
    input_names->emplace_back(MapAndBatchDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(MapAndBatchDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(MapAndBatchDatasetOp::kBatchSize);
    input_names->emplace_back(MapAndBatchDatasetOp::kNumParallelCalls);
    input_names->emplace_back(MapAndBatchDatasetOp::kDropRemainder);

    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {
        {MapAndBatchDatasetOp::kFunc, func_},
        {MapAndBatchDatasetOp::kTarguments, type_arguments_},
        {MapAndBatchDatasetOp::kOutputShapes, output_shapes_},
        {MapAndBatchDatasetOp::kOutputTypes, output_dtypes_},
        {MapAndBatchDatasetOp::kPreserveCardinality, preserve_cardinality_}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override {
    return MapAndBatchDatasetOp::kDatasetType;
  }

 private:
  std::vector<Tensor> other_arguments_;
  int64 batch_size_;
  int64 num_parallel_calls_;
  bool drop_remainder_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
  bool preserve_cardinality_;
};

class MapAndBatchDatasetOpTest : public DatasetOpsTestBase {};

FunctionDefHelper::AttrValueWrapper MapFunc(const string& func_name,
                                            const DataType& dtype) {
  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

// test case 1: num_parallel_calls = 1, drop_remainder = true,
// preserve_cardinality = false, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams1() {
  return MapAndBatchDatasetParams(RangeDatasetParams(0, 10, 2),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/2,
                                  /*num_parallel_calls=*/1,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("XTimesTwo", DT_INT64),
                                  /*func_lib=*/{test::function::XTimesTwo()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({2})},
                                  /*node_name=*/kNodeName);
}

// test case 2: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = true, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams2() {
  return MapAndBatchDatasetParams(RangeDatasetParams(0, 10, 2),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/2,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("XTimesTwo", DT_INT64),
                                  /*func_lib=*/{test::function::XTimesTwo()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/true,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({2})},
                                  /*node_name=*/kNodeName);
}

// test case 3: num_parallel_calls = 3, drop_remainder = false,
// preserve_cardinality = true, MapFunc = XTimesFour
MapAndBatchDatasetParams MapAndBatchDatasetParams3() {
  return MapAndBatchDatasetParams(
      RangeDatasetParams(0, 10, 2),
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/3,
      /*drop_remainder=*/false,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

// test case 4: num_parallel_calls = 4, drop_remainder = true,
// preserve_cardinality = false, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams4() {
  return MapAndBatchDatasetParams(RangeDatasetParams(0, 10, 2),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/2,
                                  /*num_parallel_calls=*/4,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("XTimesTwo", DT_INT64),
                                  /*func_lib=*/{test::function::XTimesTwo()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({2})},
                                  /*node_name=*/kNodeName);
}

// test case 5: num_parallel_calls = kAutotune, drop_remainder = true,
// preserve_cardinality = true, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams5() {
  return MapAndBatchDatasetParams(
      RangeDatasetParams(0, 10, 2),
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/model::kAutotune,
      /*drop_remainder=*/true,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

// test case 6: num_parallel_calls = 4, drop_remainder = false,
// preserve_cardinality = true, MapFunc = XTimesFour
MapAndBatchDatasetParams MapAndBatchDatasetParams6() {
  return MapAndBatchDatasetParams(
      RangeDatasetParams(0, 10, 2),
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/4,
      /*drop_remainder=*/false,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

MapAndBatchDatasetParams InvalidNumParallelCallsMapAndBatchDatasetParams() {
  return MapAndBatchDatasetParams(
      RangeDatasetParams(0, 10, 2),
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/-4,
      /*drop_remainder=*/false,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

MapAndBatchDatasetParams InvalidBatchSizeMapAndBatchDatasetParams() {
  return MapAndBatchDatasetParams(
      RangeDatasetParams(0, 10, 2),
      /*other_arguments=*/{},
      /*batch_size=*/-2,
      /*num_parallel_calls=*/2,
      /*drop_remainder=*/false,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<MapAndBatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/MapAndBatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/MapAndBatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/MapAndBatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}},
          {/*dataset_params=*/MapAndBatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/MapAndBatchDatasetParams5(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 8}, {16, 24}})},
          {/*dataset_params=*/MapAndBatchDatasetParams6(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}}};
}

ITERATOR_GET_NEXT_TEST_P(MapAndBatchDatasetOpTest, MapAndBatchDatasetParams,
                         GetNextTestCases())

TEST_F(MapAndBatchDatasetOpTest, DatasetNodeName) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(MapAndBatchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(MapAndBatchDatasetOp::kDatasetType)));
}

TEST_F(MapAndBatchDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<MapAndBatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/MapAndBatchDatasetParams1(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams2(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams3(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams4(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams5(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams6(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(MapAndBatchDatasetOpTest, MapAndBatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<MapAndBatchDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/MapAndBatchDatasetParams1(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/MapAndBatchDatasetParams2(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/MapAndBatchDatasetParams3(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/MapAndBatchDatasetParams4(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/MapAndBatchDatasetParams5(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/MapAndBatchDatasetParams6(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(MapAndBatchDatasetOpTest, MapAndBatchDatasetParams,
                           CardinalityTestCases())

TEST_F(MapAndBatchDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<MapAndBatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/MapAndBatchDatasetParams1(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams2(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams3(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams4(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams5(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/MapAndBatchDatasetParams6(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(MapAndBatchDatasetOpTest,
                              MapAndBatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(MapAndBatchDatasetOpTest, IteratorPrefix) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      MapAndBatchDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<MapAndBatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/MapAndBatchDatasetParams1(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/MapAndBatchDatasetParams2(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/MapAndBatchDatasetParams3(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}},
          {/*dataset_params=*/MapAndBatchDatasetParams4(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/MapAndBatchDatasetParams5(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 8}, {16, 24}})},
          {/*dataset_params=*/MapAndBatchDatasetParams6(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(MapAndBatchDatasetOpTest,
                                 MapAndBatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(MapAndBatchDatasetOpTest, InvalidBatchSize) {
  auto dataset_params = InvalidBatchSizeMapAndBatchDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(MapAndBatchDatasetOpTest, InvalidNumParallel) {
  auto dataset_params = InvalidNumParallelCallsMapAndBatchDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
