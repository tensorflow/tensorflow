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
constexpr char kIteratorPrefix[] = "Iterator";

class MapAndBatchDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `MapAndBatchDataset` op kernel
  Status CreateMapAndBatchDatasetOpKernel(
      const FunctionDefHelper::AttrValueWrapper& func,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      bool preserve_cardinality,
      std::unique_ptr<OpKernel>* map_and_batch_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(MapAndBatchDatasetOp::kDatasetType),
        {MapAndBatchDatasetOp::kInputDataset, MapAndBatchDatasetOp::kBatchSize,
         MapAndBatchDatasetOp::kNumParallelCalls,
         MapAndBatchDatasetOp::kDropRemainder},
        {{MapAndBatchDatasetOp::kFunc, func},
         {MapAndBatchDatasetOp::kTarguments, {}},
         {MapAndBatchDatasetOp::kOutputTypes, output_types},
         {MapAndBatchDatasetOp::kOutputShapes, output_shapes},
         {MapAndBatchDatasetOp::kPreserveCardinality, preserve_cardinality}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, map_and_batch_kernel));
    return Status::OK();
  }

  // Creates a new `MapAndBatchDataset` op kernel context.
  Status CreateMapAndBatchDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  TestCase(int64 start, int64 stop, int64 step, int64 batch_size,
           int64 num_parallel_calls, bool drop_remainder,
           FunctionDefHelper::AttrValueWrapper func,
           std::vector<FunctionDef> func_lib, bool preserve_cardinality,
           std::vector<Tensor> expected_outputs,
           DataTypeVector expected_output_dtypes,
           std::vector<PartialTensorShape> expected_output_shapes,
           int64 expected_cardinality, std::vector<int> breakpoints)
      : start(
            DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {start})),
        stop(DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {stop})),
        step(DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {step})),
        batch_size(DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}),
                                                           {batch_size})),
        num_parallel_calls(DatasetOpsTestBase::CreateTensor<int64>(
            TensorShape({}), {num_parallel_calls})),
        drop_remainder(DatasetOpsTestBase::CreateTensor<bool>(
            TensorShape({}), {drop_remainder})),
        func(std::move(func)),
        func_lib(std::move(func_lib)),
        preserve_cardinality(preserve_cardinality),
        expected_outputs(std::move(expected_outputs)),
        expected_output_dtypes(std::move(expected_output_dtypes)),
        expected_output_shapes(std::move(expected_output_shapes)),
        expected_cardinality(expected_cardinality),
        breakpoints(std::move(breakpoints)) {}

  Tensor start;
  Tensor stop;
  Tensor step;
  Tensor batch_size;
  Tensor num_parallel_calls;
  Tensor drop_remainder;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  bool preserve_cardinality;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

FunctionDefHelper::AttrValueWrapper MapFunc(const string& func_name,
                                            const DataType& dtype) {
  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

// test case 1: num_parallel_calls = 1, drop_remainder = true,
// preserve_cardinality = false, MapFunc = XTimesTwo
TestCase TestCase1() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/2,
          /*batch_size=*/2,
          /*num_parallel_calls=*/1,
          /*drop_remainder=*/true,
          /*func=*/MapFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*preserve_cardinality=*/false,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {0, 4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {8, 12})},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({2})},
          /*expected_cardinality=*/2,
          /*breakpoints=*/{0, 1, 4}};
}

// test case 2: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = true, MapFunc = XTimesTwo
TestCase TestCase2() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/2,
          /*batch_size=*/2,
          /*num_parallel_calls=*/2,
          /*drop_remainder=*/true,
          /*func=*/MapFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*preserve_cardinality=*/true,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {0, 4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {8, 12})},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({2})},
          /*expected_cardinality=*/2,
          /*breakpoints=*/{0, 1, 4}};
}

// test case 3: num_parallel_calls = 3, drop_remainder = false,
// preserve_cardinality = true, MapFunc = XTimesFour
TestCase TestCase3() {
  return {
      /*start=*/0,
      /*stop=*/10,
      /*step=*/2,
      /*batch_size=*/2,
      /*num_parallel_calls=*/3,
      /*drop_remainder=*/false,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*preserve_cardinality=*/true,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {0, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {16, 24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1}), {32})},
      /*expected_output_dtypes=*/{DT_INT64},
      /*expected_output_shapes=*/{PartialTensorShape({2})},
      /*expected_cardinality=*/3,
      /*breakpoints=*/{0, 1, 4}};
}

// test case 4: num_parallel_calls = 4, drop_remainder = true,
// preserve_cardinality = false, MapFunc = XTimesTwo
TestCase TestCase4() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/2,
          /*batch_size=*/2,
          /*num_parallel_calls=*/4,
          /*drop_remainder=*/true,
          /*func=*/MapFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*preserve_cardinality=*/false,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {0, 4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {8, 12})},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({2})},
          /*expected_cardinality=*/2,
          /*breakpoints=*/{0, 1, 4}};
}

// test case 5: num_parallel_calls = kAutotune, drop_remainder = true,
// preserve_cardinality = true, MapFunc = XTimesTwo
TestCase TestCase5() {
  return {
      /*start=*/0,
      /*stop=*/10,
      /*step=*/2,
      /*batch_size=*/2,
      /*num_parallel_calls=*/model::kAutotune,
      /*drop_remainder=*/true,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*preserve_cardinality=*/true,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {0, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {16, 24})},
      /*expected_output_dtypes=*/{DT_INT64},
      /*expected_output_shapes=*/{PartialTensorShape({2})},
      /*expected_cardinality=*/2,
      /*breakpoints=*/{0, 1, 4}};
}

// test case 6: num_parallel_calls = 4, drop_remainder = false,
// preserve_cardinality = true, MapFunc = XTimesTwo
TestCase TestCase6() {
  return {
      /*start=*/0,
      /*stop=*/10,
      /*step=*/2,
      /*batch_size=*/2,
      /*num_parallel_calls=*/4,
      /*drop_remainder=*/false,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*preserve_cardinality=*/false,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {0, 8}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({2}), {16, 24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({1}), {32})},
      /*expected_output_dtypes=*/{DT_INT64},
      /*expected_output_shapes=*/{PartialTensorShape({2})},
      /*expected_cardinality=*/3,
      /*breakpoints=*/{0, 1, 4}};
}

TestCase InvalidNumParallelCallsTestCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/2,
          /*batch_size=*/2,
          /*num_parallel_calls=*/-4,
          /*drop_remainder=*/true,
          /*func=*/MapFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*preserve_cardinality=*/false,
          /*expected_outputs*/ {},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({2})},
          /*expected_cardinality=*/-1,
          /*breakpoints=*/{0, 1, 4}};
}

TestCase InvalidBatchSizeTestCase() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/2,
          /*batch_size=*/-2,
          /*num_parallel_calls=*/2,
          /*drop_remainder=*/true,
          /*func=*/MapFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*preserve_cardinality=*/false,
          /*expected_outputs*/ {},
          /*expected_output_dtypes=*/{DT_INT64},
          /*expected_output_shapes=*/{PartialTensorShape({2})},
          /*expected_cardinality=*/-1,
          /*breakpoints=*/{0, 1, 4}};
}

class ParameterizedMapAndBatchDatasetOpTest
    : public MapAndBatchDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedMapAndBatchDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(map_and_batch_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_and_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                   kIteratorPrefix, &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

TEST_F(MapAndBatchDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  EXPECT_EQ(map_and_batch_dataset->node_name(), kNodeName);
}

TEST_F(MapAndBatchDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  EXPECT_EQ(map_and_batch_dataset->type_string(),
            name_utils::OpName(MapAndBatchDatasetOp::kDatasetType));
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(map_and_batch_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(map_and_batch_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  EXPECT_EQ(map_and_batch_dataset->Cardinality(),
            test_case.expected_cardinality);
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(
      map_and_batch_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(map_and_batch_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_and_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                   kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(map_and_batch_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_and_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                   kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(MapAndBatchDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(map_and_batch_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_and_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                   kIteratorPrefix, &iterator));

  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(MapAndBatchDatasetOp::kDatasetType,
                                       kIteratorPrefix));
}

TEST_P(ParameterizedMapAndBatchDatasetOpTest, Roundtrip) {
  int thread_num = 3, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
  TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.preserve_cardinality,
      &map_and_batch_dataset_kernel));

  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop, test_case.step,
                                {DT_INT64}, {TensorShape({})},
                                &range_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
      {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
       TensorValue(&test_case.num_parallel_calls),
       TensorValue(&test_case.drop_remainder)});

  std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
  TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
      map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
      &map_and_batch_dataset_context));
  DatasetBase* map_and_batch_dataset;
  TF_ASSERT_OK(CreateDataset(map_and_batch_dataset_kernel.get(),
                             map_and_batch_dataset_context.get(),
                             &map_and_batch_dataset));
  core::ScopedUnref scoped_unref_dataset(map_and_batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(CreateIteratorContext(map_and_batch_dataset_context.get(),
                                     &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(map_and_batch_dataset->MakeIterator(iterator_ctx.get(),
                                                   kIteratorPrefix, &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, kIteratorPrefix,
                                 *map_and_batch_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

TEST_F(MapAndBatchDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));

  std::vector<TestCase> test_cases = {InvalidNumParallelCallsTestCase(),
                                      InvalidBatchSizeTestCase()};
  for (TestCase test_case : test_cases) {
    TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

    std::unique_ptr<OpKernel> map_and_batch_dataset_kernel;
    TF_ASSERT_OK(CreateMapAndBatchDatasetOpKernel(
        test_case.func, test_case.expected_output_dtypes,
        test_case.expected_output_shapes, test_case.preserve_cardinality,
        &map_and_batch_dataset_kernel));

    Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_ASSERT_OK(MakeRangeDataset(test_case.start, test_case.stop,
                                  test_case.step, {DT_INT64}, {TensorShape({})},
                                  &range_dataset_tensor));

    gtl::InlinedVector<TensorValue, 4> map_and_batch_dataset_inputs(
        {TensorValue(&range_dataset_tensor), TensorValue(&test_case.batch_size),
         TensorValue(&test_case.num_parallel_calls),
         TensorValue(&test_case.drop_remainder)});

    std::unique_ptr<OpKernelContext> map_and_batch_dataset_context;
    TF_ASSERT_OK(CreateMapAndBatchDatasetContext(
        map_and_batch_dataset_kernel.get(), &map_and_batch_dataset_inputs,
        &map_and_batch_dataset_context));
    DatasetBase* map_and_batch_dataset;
    EXPECT_EQ(CreateDataset(map_and_batch_dataset_kernel.get(),
                            map_and_batch_dataset_context.get(),
                            &map_and_batch_dataset)
                  .code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

INSTANTIATE_TEST_SUITE_P(MapAndBatchDatasetOpTest,
                         ParameterizedMapAndBatchDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6()})));

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
