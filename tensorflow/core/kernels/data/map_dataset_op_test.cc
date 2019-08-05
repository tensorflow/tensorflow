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

class MapDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new MapDataset op kernel.
  Status CreateMapDatasetOpKernel(
      const FunctionDefHelper::AttrValueWrapper& func,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* map_kernel) {
    NodeDef map_dataset_node_def = test::function::NDef(
        kNodeName, name_utils::OpName(MapDatasetOp::kDatasetType),
        {MapDatasetOp::kInputDataset},
        {{MapDatasetOp::kFunc, func},
         {MapDatasetOp::kTarguments, {}},
         {MapDatasetOp::kOutputShapes, output_shapes},
         {MapDatasetOp::kOutputTypes, output_types},
         {MapDatasetOp::kUseInterOpParallelism, true},
         {MapDatasetOp::kPreserveCardinality, false}});
    TF_RETURN_IF_ERROR(CreateOpKernel(map_dataset_node_def, map_kernel));
    return Status::OK();
  }

  // Creates a new MapDataset op kernel context.
  Status CreateMapDatasetContext(
      OpKernel* const map_kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
      std::unique_ptr<OpKernelContext>* map_context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*map_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(map_kernel, inputs, map_context));
    return Status::OK();
  }
};

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

MapDatasetParams MapDataset1() {
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
          /*node_name=*/"map_dataset"};
}

MapDatasetParams MapDataset2() {
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
          /*node_name=*/"map_dataset"};
}

// In this test case, the function `XTimesFour()` will call `XTimesTwo()`, so
// both of them are added to the function library.
MapDatasetParams MapDataset3() {
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
      /*node_name=*/"map_dataset"};
}

class ParameterizedGetNextTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<GetNextTestCase<MapDatasetParams>> {
};

GetNextTestCase<MapDatasetParams> GetNextTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {6}, {12}, {18}})};
}

GetNextTestCase<MapDatasetParams> GetNextTestCase2() {
  return {/*dataset_params=*/MapDataset2(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{20}, {14}, {8}, {2}})};
}

GetNextTestCase<MapDatasetParams> GetNextTestCase3() {
  return {/*dataset_params=*/MapDataset3(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_dataset->MakeIterator(iterator_context.get(),
                                         kIteratorPrefix, &iterator));

  TF_ASSERT_OK(CheckIteratorGetNext(iterator.get(), iterator_context.get(),
                                    test_case.expected_outputs,
                                    /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedGetNextTest,
    ::testing::ValuesIn(std::vector<GetNextTestCase<MapDatasetParams>>(
        {GetNextTestCase1(), GetNextTestCase2(), GetNextTestCase3()})));

DatasetNodeNameTestCase<MapDatasetParams> DatasetNodeNameTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_node_name=*/kNodeName};
}

TEST_F(MapDatasetOpTest, DatasetNodeName) {
  auto test_case = DatasetNodeNameTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  TF_ASSERT_OK(CheckDatasetNodeName(*map_dataset, kNodeName));
}

DatasetTypeStringTestCase<MapDatasetParams> DatasetTypeStringTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_dataset_type_string=*/
          name_utils::OpName(MapDatasetOp::kDatasetType)};
}

TEST_F(MapDatasetOpTest, DatasetTypeString) {
  auto test_case = DatasetTypeStringTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  TF_ASSERT_OK(CheckDatasetTypeString(*map_dataset,
                                      test_case.expected_dataset_type_string));
}

DatasetOutputDtypesTestCase<MapDatasetParams> DatasetOutputDtypesTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  auto test_case = DatasetOutputDtypesTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  TF_ASSERT_OK(
      CheckDatasetOutputDtypes(*map_dataset, test_case.expected_output_dtypes));
}

DatasetOutputShapesTestCase<MapDatasetParams> DatasetOutputShapesTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_output_shapes=*/{PartialTensorShape({})}};
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  auto test_case = DatasetOutputShapesTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  TF_ASSERT_OK(
      CheckDatasetOutputShapes(*map_dataset, test_case.expected_output_shapes));
}

class ParameterizedCardinalityTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<
          CardinalityTestCase<MapDatasetParams>> {};

CardinalityTestCase<MapDatasetParams> CardinalityTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_cardinality=*/4};
}

CardinalityTestCase<MapDatasetParams> CardinalityTestCase2() {
  return {/*dataset_params=*/MapDataset2(),
          /*expected_cardinality=*/4};
}

CardinalityTestCase<MapDatasetParams> CardinalityTestCase3() {
  return {/*dataset_params=*/MapDataset3(),
          /*expected_cardinality=*/4};
}

TEST_P(ParameterizedCardinalityTest, Cardinality) {
  auto test_case = GetParam();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  TF_ASSERT_OK(
      CheckDatasetCardinality(*map_dataset, test_case.expected_cardinality));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedCardinalityTest,
    ::testing::ValuesIn(std::vector<CardinalityTestCase<MapDatasetParams>>(
        {CardinalityTestCase1(), CardinalityTestCase2(),
         CardinalityTestCase3()})));

class ParameterizedDatasetSaveTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<
          DatasetSaveTestCase<MapDatasetParams>> {};

DatasetSaveTestCase<MapDatasetParams> DatasetSaveTestCase1() {
  return {/*dataset_params=*/MapDataset1()};
}

DatasetSaveTestCase<MapDatasetParams> DatasetSaveTestCase2() {
  return {/*dataset_params=*/MapDataset2()};
}

DatasetSaveTestCase<MapDatasetParams> DatasetSaveTestCase3() {
  return {/*dataset_params=*/MapDataset3()};
}

TEST_P(ParameterizedDatasetSaveTest, DatasetSave) {
  auto test_case = GetParam();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  TF_ASSERT_OK(CheckDatasetSave(*map_dataset));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedDatasetSaveTest,
    ::testing::ValuesIn(std::vector<DatasetSaveTestCase<MapDatasetParams>>(
        {DatasetSaveTestCase1(), DatasetSaveTestCase2(),
         DatasetSaveTestCase3()})));

IteratorOutputDtypesTestCase<MapDatasetParams> IteratorOutputDtypesTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  auto test_case = IteratorOutputDtypesTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_dataset->MakeIterator(iterator_context.get(),
                                         kIteratorPrefix, &iterator));

  TF_ASSERT_OK(
      CheckIteratorOutputDtypes(*iterator, test_case.expected_output_dtypes));
}

IteratorOutputShapesTestCase<MapDatasetParams> IteratorOutputShapesTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_output_shapes=*/{PartialTensorShape({})}};
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  auto test_case = IteratorOutputShapesTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_dataset->MakeIterator(iterator_context.get(),
                                         kIteratorPrefix, &iterator));

  TF_ASSERT_OK(
      CheckIteratorOutputShapes(*iterator, test_case.expected_output_shapes));
}

IteratorPrefixTestCase<MapDatasetParams> IteratorPrefixTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
              MapDatasetOp::kDatasetType, kIteratorPrefix)};
}

TEST_F(MapDatasetOpTest, IteratorPrefix) {
  auto test_case = IteratorPrefixTestCase1();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_dataset->MakeIterator(iterator_context.get(),
                                         kIteratorPrefix, &iterator));

  TF_ASSERT_OK(
      CheckIteratorPrefix(*iterator, test_case.expected_iterator_prefix));
}

class ParameterizedIteratorSaveAndRestoreTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<MapDatasetParams>> {};

IteratorSaveAndRestoreTestCase<MapDatasetParams>
IteratorSaveAndRestoreTestCase1() {
  return {/*dataset_params=*/MapDataset1(),
          /*breakpoints*/ {0, 1, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {6}, {12}, {18}})};
}

IteratorSaveAndRestoreTestCase<MapDatasetParams>
IteratorSaveAndRestoreTestCase2() {
  return {/*dataset_params=*/MapDataset2(),
          /*breakpoints*/ {0, 1, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{20}, {14}, {8}, {2}})};
}

IteratorSaveAndRestoreTestCase<MapDatasetParams>
IteratorSaveAndRestoreTestCase3() {
  return {/*dataset_params=*/MapDataset3(),
          /*breakpoints*/ {0, 1, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {12}, {24}, {36}})};
}

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();

  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(
      InitFunctionLibraryRuntime(test_case.dataset_params.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.dataset_params.func, test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &map_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(map_dataset_kernel.get(), &inputs,
                                       &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_dataset->MakeIterator(iterator_context.get(),
                                         kIteratorPrefix, &iterator));

  TF_ASSERT_OK(CheckIteratorSaveAndRestore(
      *map_dataset, iterator_context.get(), kIteratorPrefix,
      test_case.expected_outputs, test_case.breakpoints));
}

INSTANTIATE_TEST_SUITE_P(
    MapDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(
        std::vector<IteratorSaveAndRestoreTestCase<MapDatasetParams>>(
            {IteratorSaveAndRestoreTestCase1(),
             IteratorSaveAndRestoreTestCase2(),
             IteratorSaveAndRestoreTestCase3()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
