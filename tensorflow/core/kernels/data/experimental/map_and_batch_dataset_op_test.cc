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
  MapAndBatchDatasetParams(
      RangeDatasetParams range_dataset_params,
      std::vector<Tensor> other_arguments, int64 batch_size,
      int64 num_parallel_calls, bool drop_remainder,
      FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      bool preserve_cardinality, DataTypeVector output_dtypes,
      std::vector<PartialTensorShape> output_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        input_dataset_params(std::move(range_dataset_params)),
        other_arguments(std::move(other_arguments)),
        batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
        num_parallel_calls(
            CreateTensor<int64>(TensorShape({}), {num_parallel_calls})),
        drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        func(std::move(func)),
        func_lib(std::move(func_lib)),
        type_arguments(std::move(type_arguments)),
        preserve_cardinality(preserve_cardinality) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (!IsDatasetTensor(input_dataset)) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset)};
    for (auto& argument : other_arguments) {
      inputs->emplace_back(TensorValue(&argument));
    }
    inputs->insert(inputs->end(),
                   {TensorValue(&batch_size), TensorValue(&num_parallel_calls),
                    TensorValue(&drop_remainder)});
    return Status::OK();
  }

  RangeDatasetParams input_dataset_params;
  Tensor input_dataset;
  std::vector<Tensor> other_arguments;
  Tensor batch_size;
  Tensor num_parallel_calls;
  Tensor drop_remainder;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  DataTypeVector type_arguments;
  bool preserve_cardinality;
};

class MapAndBatchDatasetOpTest
    : public DatasetOpsTestBaseV2<MapAndBatchDatasetParams> {
 public:
  Status Initialize(
      MapAndBatchDatasetParams* map_and_batch_dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime(
        map_and_batch_dataset_params->func_lib, cpu_num_));

    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*map_and_batch_dataset_params, &dataset_kernel_));
    TF_RETURN_IF_ERROR(
        MakeRangeDataset(map_and_batch_dataset_params->input_dataset_params,
                         &map_and_batch_dataset_params->input_dataset));
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(map_and_batch_dataset_params->MakeInputs(&inputs));
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(
        iterator_ctx_.get(), map_and_batch_dataset_params->iterator_prefix,
        &iterator_));
    return Status::OK();
  }

 protected:
  Status MakeDatasetOpKernel(
      const MapAndBatchDatasetParams& map_and_batch_dataset_params,
      std::unique_ptr<OpKernel>* map_and_batch_kernel) override {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(MapAndBatchDatasetOp::kDatasetType),
        {MapAndBatchDatasetOp::kInputDataset, MapAndBatchDatasetOp::kBatchSize,
         MapAndBatchDatasetOp::kNumParallelCalls,
         MapAndBatchDatasetOp::kDropRemainder},
        {{MapAndBatchDatasetOp::kFunc, map_and_batch_dataset_params.func},
         {MapAndBatchDatasetOp::kTarguments,
          map_and_batch_dataset_params.type_arguments},
         {MapAndBatchDatasetOp::kOutputTypes,
          map_and_batch_dataset_params.output_dtypes},
         {MapAndBatchDatasetOp::kOutputShapes,
          map_and_batch_dataset_params.output_shapes},
         {MapAndBatchDatasetOp::kPreserveCardinality,
          map_and_batch_dataset_params.preserve_cardinality}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, map_and_batch_kernel));
    return Status::OK();
  }
};

FunctionDefHelper::AttrValueWrapper MapFunc(const string& func_name,
                                            const DataType& dtype) {
  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

// test case 1: num_parallel_calls = 1, drop_remainder = true,
// preserve_cardinality = false, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams1() {
  return {/*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
          /*node_name=*/kNodeName};
}

// test case 2: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = true, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams2() {
  return {/*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
          /*node_name=*/kNodeName};
}

// test case 3: num_parallel_calls = 3, drop_remainder = false,
// preserve_cardinality = true, MapFunc = XTimesFour
MapAndBatchDatasetParams MapAndBatchDatasetParams3() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
      /*node_name=*/kNodeName};
}

// test case 4: num_parallel_calls = 4, drop_remainder = true,
// preserve_cardinality = false, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams4() {
  return {/*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
          /*node_name=*/kNodeName};
}

// test case 5: num_parallel_calls = kAutotune, drop_remainder = true,
// preserve_cardinality = true, MapFunc = XTimesTwo
MapAndBatchDatasetParams MapAndBatchDatasetParams5() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
      /*node_name=*/kNodeName};
}

// test case 6: num_parallel_calls = 4, drop_remainder = false,
// preserve_cardinality = true, MapFunc = XTimesFour
MapAndBatchDatasetParams MapAndBatchDatasetParams6() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
      /*node_name=*/kNodeName};
}

MapAndBatchDatasetParams InvalidNumParallelCallsMapAndBatchDatasetParams() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
      /*node_name=*/kNodeName};
}

MapAndBatchDatasetParams InvalidBatchSizeMapAndBatchDatasetParams() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
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
      /*node_name=*/kNodeName};
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

TEST_F(MapAndBatchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(MapAndBatchDatasetOp::kDatasetType)));
}

TEST_F(MapAndBatchDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = MapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
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
  TF_ASSERT_OK(Initialize(&dataset_params));
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
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      MapAndBatchDatasetOp::kDatasetType, dataset_params.iterator_prefix)));
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
  EXPECT_EQ(Initialize(&dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(MapAndBatchDatasetOpTest, InvalidNumParallel) {
  auto dataset_params = InvalidNumParallelCallsMapAndBatchDatasetParams();
  EXPECT_EQ(Initialize(&dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
