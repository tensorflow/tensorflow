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
#include "tensorflow/core/kernels/data/map_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "map_dataset";

class MapDatasetOpTest : public DatasetOpsTestBase {};

MapDatasetParams MapDatasetParams1() {
  auto map_dataset_params_0 = MapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*preserve_cardinality=*/true,
      /*node_name=*/"map_dataset_0");
  return MapDatasetParams(
      std::move(map_dataset_params_0),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*preserve_cardinality=*/true,
      /*node_name=*/"map_dataset_1");
}

MapDatasetParams MapDatasetParams2() {
  auto batch_dataset_params =
      BatchDatasetParams(RangeDatasetParams(10, 0, -3),
                         /*batch_size=*/2,
                         /*drop_remainder=*/false,
                         /*parallel_copy=*/true,
                         /*output_dtypes=*/{DT_INT64},
                         /*output_shapes=*/{PartialTensorShape({2})},
                         /*node_name=*/"batch_dataset");
  return MapDatasetParams(
      std::move(batch_dataset_params),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XAddX", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XAddX()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*use_inter_op_parallelism=*/true,
      /*preserve_cardinality=*/false,
      /*node_name=*/kNodeName);
}

// In this test case, the function `XTimesFour()` will call `XTimesTwo()`, so
// both of them are added to the function library.
MapDatasetParams MapDatasetParams3() {
  return MapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XTimesFour", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/false,
      /*preserve_cardinality=*/true,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<MapDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/MapDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})},
          {/*dataset_params=*/MapDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{20, 14}, {8, 2}})},
          {/*dataset_params=*/MapDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})}};
}

ITERATOR_GET_NEXT_TEST_P(MapDatasetOpTest, MapDatasetParams, GetNextTestCases())

TEST_F(MapDatasetOpTest, DatasetNodeName) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(MapDatasetOpTest, DatasetTypeString) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(MapDatasetOp::kDatasetType)));
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<MapDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/MapDatasetParams1(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/MapDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/MapDatasetParams3(),
           /*expected_cardinality=*/4}};
}

DATASET_CARDINALITY_TEST_P(MapDatasetOpTest, MapDatasetParams,
                           CardinalityTestCases())

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(MapDatasetOpTest, IteratorPrefix) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      MapDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<MapDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/MapDatasetParams1(),
           /*breakpoints*/ {0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})},
          {/*dataset_params=*/MapDatasetParams2(),
           /*breakpoints*/ {0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{20, 14}, {8, 2}})},
          {/*dataset_params=*/MapDatasetParams3(),
           /*breakpoints*/ {0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(MapDatasetOpTest, MapDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
