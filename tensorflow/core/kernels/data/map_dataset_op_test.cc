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
constexpr char kIteratorPrefix[] = "Iterator";

class MapDatasetParams : public DatasetParams {
 public:
  MapDatasetParams(int64 start, int64 stop, int64 step,
                   std::vector<Tensor> other_arguments,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   DataTypeVector type_arguments, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   bool use_inter_op_parallelism, bool preserve_cardinality,
                   string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        range_dataset_params(start, stop, step, {DT_INT64},
                             {PartialTensorShape({})}, ""),
        other_arguments(std::move(other_arguments)),
        func(std::move(func)),
        func_lib(std::move(func_lib)),
        type_arguments(std::move(type_arguments)),
        use_inter_op_parallelism(use_inter_op_parallelism),
        preserve_cardinality(preserve_cardinality) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (input_dataset.NumElements() == 0 ||
        input_dataset.dtype() != DT_VARIANT) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset)};
    for (auto& argument : other_arguments) {
      inputs->emplace_back(TensorValue(&argument));
    }
    return Status::OK();
  }

  RangeDatasetParams range_dataset_params;
  Tensor input_dataset;
  std::vector<Tensor> other_arguments;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  DataTypeVector type_arguments;
  bool use_inter_op_parallelism;
  bool preserve_cardinality;
};

class MapDatasetOpTest : public DatasetOpsTestBaseV2<MapDatasetParams> {
 public:
  Status Initialize(MapDatasetParams* map_dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(
        InitFunctionLibraryRuntime(map_dataset_params->func_lib, cpu_num_));

    TF_RETURN_IF_ERROR(
        CreateMapDatasetOpKernel(*map_dataset_params, &dataset_kernel_));
    TF_RETURN_IF_ERROR(
        MakeRangeDataset(map_dataset_params->range_dataset_params,
                         &map_dataset_params->input_dataset));
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(map_dataset_params->MakeInputs(&inputs));
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(iterator_ctx_.get(),
                                              kIteratorPrefix, &iterator_));
    return Status::OK();
  }

 protected:
  // Creates a new MapDataset op kernel.
  Status CreateMapDatasetOpKernel(const MapDatasetParams& map_dataset_params,
                                  std::unique_ptr<OpKernel>* map_kernel) {
    NodeDef map_dataset_node_def = test::function::NDef(
        map_dataset_params.node_name,
        name_utils::OpName(MapDatasetOp::kDatasetType),
        {MapDatasetOp::kInputDataset},
        {{MapDatasetOp::kFunc, map_dataset_params.func},
         {MapDatasetOp::kTarguments, map_dataset_params.type_arguments},
         {MapDatasetOp::kOutputShapes, map_dataset_params.output_shapes},
         {MapDatasetOp::kOutputTypes, map_dataset_params.output_dtypes},
         {MapDatasetOp::kUseInterOpParallelism,
          map_dataset_params.use_inter_op_parallelism},
         {MapDatasetOp::kPreserveCardinality,
          map_dataset_params.preserve_cardinality}});
    TF_RETURN_IF_ERROR(CreateOpKernel(map_dataset_node_def, map_kernel));
    return Status::OK();
  }
};

MapDatasetParams MapDatasetParams1() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/3,
          /*other_arguments=*/{},
          /*func=*/
          FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*type_arguments=*/{},
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*use_inter_op_parallelism=*/true,
          /*preserve_cardinality=*/true,
          /*node_name=*/kNodeName};
}

MapDatasetParams MapDatasetParams2() {
  return {/*start=*/10,
          /*stop=*/0,
          /*step=*/-3,
          /*other_arguments=*/{},
          /*func=*/
          FunctionDefHelper::FunctionRef("XAddX", {{"T", DT_INT64}}),
          /*func_lib=*/{test::function::XAddX()},
          /*type_arguments=*/{},
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*use_inter_op_parallelism=*/true,
          /*preserve_cardinality=*/false,
          /*node_name=*/kNodeName};
}

// In this test case, the function `XTimesFour()` will call `XTimesTwo()`, so
// both of them are added to the function library.
MapDatasetParams MapDatasetParams3() {
  return {
      /*start=*/0,
      /*stop=*/10,
      /*step=*/3,
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XTimesFour", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/false,
      /*preserve_cardinality=*/true,
      /*node_name=*/kNodeName};
}

class ParameterizedGetNextTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<GetNextTestCase<MapDatasetParams>> {
};

std::vector<GetNextTestCase<MapDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/MapDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {6}, {12}, {18}})},
          {/*dataset_params=*/MapDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{20}, {14}, {8}, {2}})},
          {/*dataset_params=*/MapDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})}};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(&test_case.dataset_params));
  TF_ASSERT_OK(
      CheckIteratorGetNext(test_case.expected_outputs, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedGetNextTest,
    ::testing::ValuesIn(
        std::vector<GetNextTestCase<MapDatasetParams>>(GetNextTestCases())));

TEST_F(MapDatasetOpTest, DatasetNodeName) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name));
}

TEST_F(MapDatasetOpTest, DatasetTypeString) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(MapDatasetOp::kDatasetType)));
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

class ParameterizedCardinalityTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<
          CardinalityTestCase<MapDatasetParams>> {};

std::vector<CardinalityTestCase<MapDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/MapDatasetParams1(), /*expected_cardinality=*/4},
          {/*dataset_params=*/MapDatasetParams2(), /*expected_cardinality=*/4},
          {/*dataset_params=*/MapDatasetParams3(), /*expected_cardinality=*/4}};
}

TEST_P(ParameterizedCardinalityTest, Cardinality) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(&test_case.dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(test_case.expected_cardinality));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedCardinalityTest,
    ::testing::ValuesIn(std::vector<CardinalityTestCase<MapDatasetParams>>(
        CardinalityTestCases())));

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(MapDatasetOpTest, IteratorPrefix) {
  auto dataset_params = MapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(MapDatasetOp::kDatasetType, kIteratorPrefix)));
}

class ParameterizedIteratorSaveAndRestoreTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<MapDatasetParams>> {};

std::vector<IteratorSaveAndRestoreTestCase<MapDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/MapDatasetParams1(),
           /*breakpoints*/ {0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {6}, {12}, {18}})},
          {/*dataset_params=*/MapDatasetParams2(),
           /*breakpoints*/ {0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{20}, {14}, {8}, {2}})},
          {/*dataset_params=*/MapDatasetParams3(),
           /*breakpoints*/ {0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})}};
}

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(&test_case.dataset_params));
  TF_ASSERT_OK(CheckIteratorSaveAndRestore(
      kIteratorPrefix, test_case.expected_outputs, test_case.breakpoints));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(
        std::vector<IteratorSaveAndRestoreTestCase<MapDatasetParams>>(
            IteratorSaveAndRestoreTestCases())));

}  // namespace
}  // namespace data
}  // namespace tensorflow
