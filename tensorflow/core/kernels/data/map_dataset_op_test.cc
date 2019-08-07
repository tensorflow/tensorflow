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

struct TestCase {
  int64 start;
  int64 end;
  int64 step;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

TestCase TestCase1() {
  return {/*start*/ 0,
          /*end*/ 10,
          /*step*/ 3,
          /*func*/
          FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::XTimesTwo()},
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {0}),
           CreateTensor<int64>(TensorShape({}), {6}),
           CreateTensor<int64>(TensorShape({}), {12}),
           CreateTensor<int64>(TensorShape({}), {18})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 5}};
}

TestCase TestCase2() {
  return {/*start*/ 10,
          /*end*/ 0,
          /*step*/ -3,
          /*func*/
          FunctionDefHelper::FunctionRef("XAddX", {{"T", DT_INT64}}),
          /*func_lib*/ {test::function::XAddX()},
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {20}),
           CreateTensor<int64>(TensorShape({}), {14}),
           CreateTensor<int64>(TensorShape({}), {8}),
           CreateTensor<int64>(TensorShape({}), {2})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 5}};
}

// In this test case, the function `XTimesFour()` will call `XTimesTwo()`, so
// both of them are added to the function library.
TestCase TestCase3() {
  return {
      /*start*/ 0,
      /*end*/ 10,
      /*step*/ 3,
      /*func*/
      FunctionDefHelper::FunctionRef("XTimesFour", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({}), {0}),
       CreateTensor<int64>(TensorShape({}), {12}),
       CreateTensor<int64>(TensorShape({}), {24}),
       CreateTensor<int64>(TensorShape({}), {36})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 5}};
}

class ParameterizedMapDatasetOpTest
    : public MapDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedMapDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                   &end_of_sequence));
    if (!end_of_sequence) {
      EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
      TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
      expected_outputs_it++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

TEST_F(MapDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  EXPECT_EQ(map_dataset->node_name(), kNodeName);
}

TEST_F(MapDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  EXPECT_EQ(map_dataset->type_string(),
            name_utils::OpName(MapDatasetOp::kDatasetType));
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(map_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(map_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedMapDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  EXPECT_EQ(map_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedMapDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(map_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(MapDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(MapDatasetOp::kDatasetType, "Iterator"));
}

TEST_P(ParameterizedMapDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.start, test_case.end, test_case.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  // The ownership of range_dataset is transferred to DatasetVariantWrapper,
  // which will handle the release of memory.
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  gtl::InlinedVector<TensorValue, 4> map_dataset_inputs;
  map_dataset_inputs.emplace_back(&range_dataset_tensor);

  std::unique_ptr<OpKernel> map_dataset_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, &map_dataset_kernel));
  std::unique_ptr<OpKernelContext> map_dataset_context;
  TF_ASSERT_OK(CreateMapDatasetContext(
      map_dataset_kernel.get(), &map_dataset_inputs, &map_dataset_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(CreateDataset(map_dataset_kernel.get(),
                             map_dataset_context.get(), &map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(map_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_context.get(), &reader, "Iterator",
                                 *map_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                     &end_of_sequence));
      if (!end_of_sequence) {
        EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
        expected_outputs_it++;
      }
      cur_iteration++;
    }

    if (breakpoint >= test_case.expected_cardinality) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MapDatasetOpTest, ParameterizedMapDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
