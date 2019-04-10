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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "parallel_map_dataset";
constexpr char kOpName[] = "ParallelMapDataset";

class ParallelMapDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `ParallelMapDataset` op kernel
  Status CreateParallelMapDatasetOpKernel(
      const FunctionDefHelper::AttrValueWrapper& func,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      bool use_inter_op_parallelism, bool sloppy, bool preserve_cardinality,
      std::unique_ptr<OpKernel>* parallel_map_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "num_parallel_calls"},
        {{"f", func},
         {"Targuments", {}},
         {"output_types", output_types},
         {"output_shapes", output_shapes},
         {"use_inter_op_parallelism", use_inter_op_parallelism},
         {"sloppy", sloppy},
         {"preserve_cardinality", preserve_cardinality}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, parallel_map_kernel));
    return Status::OK();
  }

  // Creates a new `ParallelMapDataset` op kernel context.
  Status CreateParallelMapDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct RangeDatasetParam {
  int64 start;
  int64 end;
  int64 step;
};

struct TestCase {
  RangeDatasetParam range_data_param;
  Tensor num_parallel_calls;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  bool use_inter_op_parallelism;
  bool sloppy;
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

// test case 1: num_parallel_calls = 1, use_inter_op_parallelism = false,
// sloppy = false, preserve_cardinality = false, MapFunc = XTimesTwo
TestCase TestCase1() {
  return {/*range_data_param*/ {0, 10, 3},
          /*num_parallel_calls*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
          /*func*/ MapFunc("XTimesTwo", DT_INT64),
          /*func_lib*/ {test::function::XTimesTwo()},
          /*use_inter_op_parallelism*/ false,
          /*sloppy*/ false,
          /*preserve_cardinality*/ false,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {18})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 9}};
}

// test case 2: num_parallel_calls = 2, use_inter_op_parallelism = true,
// sloppy = true, preserve_cardinality = true, MapFunc = XTimesTwo
TestCase TestCase2() {
  return {/*range_data_param*/ {0, 10, 3},
          /*num_parallel_calls*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
          /*func*/ MapFunc("XTimesTwo", DT_INT64),
          /*func_lib*/ {test::function::XTimesTwo()},
          /*use_inter_op_parallelism*/ true,
          /*sloppy*/ true,
          /*preserve_cardinality*/ true,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {18})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 5}};
}

// test case 3: num_parallel_calls = 3, use_inter_op_parallelism = true,
// sloppy = false, preserve_cardinality = false, MapFunc = XTimesFour
TestCase TestCase3() {
  return {
      /*range_data_param*/ {0, 10, 3},
      /*num_parallel_calls*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
      /*func*/ MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*use_inter_op_parallelism*/ true,
      /*sloppy*/ false,
      /*preserve_cardinality*/ false,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {36})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 5}};
}

// test case 4: num_parallel_calls = 4, use_inter_op_parallelism = false,
// sloppy = false, preserve_cardinality = false, MapFunc = XTimesTwo
TestCase TestCase4() {
  return {/*range_data_param*/ {0, 10, 3},
          /*num_parallel_calls*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
          /*func*/ MapFunc("XTimesTwo", DT_INT64),
          /*func_lib*/ {test::function::XTimesTwo()},
          /*use_inter_op_parallelism*/ false,
          /*sloppy*/ false,
          /*preserve_cardinality*/ false,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {18})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 5}};
}

// test case 5: num_parallel_calls = kAutoTune, use_inter_op_parallelism = true,
// sloppy = true, preserve_cardinality = true, MapFunc = XTimesFour
TestCase TestCase5() {
  return {
      /*range_data_param*/ {0, 10, 3},
      /*num_parallel_calls*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}),
                                              {model::kAutoTune}),
      /*func*/ MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*use_inter_op_parallelism*/ true,
      /*sloppy*/ true,
      /*preserve_cardinality*/ true,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {36})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 5}};
}

// test case 6: num_parallel_calls = 4, use_inter_op_parallelism = true,
// sloppy = false, preserve_cardinality = false, MapFunc = XTimesFour
TestCase TestCase6() {
  return {
      /*range_data_param*/ {0, 10, 3},
      /*num_parallel_calls*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
      /*func*/ MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*use_inter_op_parallelism*/ true,
      /*sloppy*/ false,
      /*preserve_cardinality*/ false,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {36})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 5}};
}

// TODO(feihugis): make this test case work.
// test case 7: num_parallel_calls = 2, use_inter_op_parallelism = false,
// sloppy = false, preserve_cardinality = false, MapFunc = XTimesFour
TestCase TestCase7() {
  return {
      /*range_data_param*/ {0, 10, 3},
      /*num_parallel_calls*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2}),
      /*func*/ MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*use_inter_op_parallelism*/ false,
      /*sloppy*/ false,
      /*preserve_cardinality*/ false,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {36})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 5}};
}

// TODO(feihugis): make this test case work.
// test case 8: num_parallel_calls = kAutoTune, use_inter_op_parallelism =
// false, sloppy = true, preserve_cardinality = true, MapFunc = XTimesFour
TestCase TestCase8() {
  return {
      /*range_data_param*/ {0, 10, 3},
      /*num_parallel_calls*/
      DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}),
                                              {model::kAutoTune}),
      /*func*/ MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*use_inter_op_parallelism*/ false,
      /*sloppy*/ true,
      /*preserve_cardinality*/ true,
      /*expected_outputs*/
      {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {12}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {24}),
       DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {36})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 4,
      /*breakpoints*/ {0, 1, 5}};
}

TestCase InvalidNumParallelCallsTestCase() {
  return {/*range_data_param*/ {0, 10, 3},
          /*num_parallel_calls*/
          DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {-4}),
          /*func*/ MapFunc("XTimesTwo", DT_INT64),
          /*func_lib*/ {test::function::XTimesTwo()},
          /*use_inter_op_parallelism*/ true,
          /*sloppy*/ true,
          /*preserve_cardinality*/ true,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ -1,
          /*breakpoints*/ {0, 1, 5}};
}

class ParameterizedParallelMapDatasetOpTest
    : public ParallelMapDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedParallelMapDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(parallel_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_map_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*expect_items_equal*/ test_case.sloppy));
}

TEST_F(ParallelMapDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  EXPECT_EQ(parallel_map_dataset->node_name(), kNodeName);
}

TEST_F(ParallelMapDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  EXPECT_EQ(parallel_map_dataset->type_string(), kOpName);
}

TEST_P(ParameterizedParallelMapDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(parallel_map_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedParallelMapDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(parallel_map_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedParallelMapDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  EXPECT_EQ(parallel_map_dataset->Cardinality(),
            test_case.expected_cardinality);
}

TEST_P(ParameterizedParallelMapDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(
      parallel_map_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedParallelMapDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(parallel_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_map_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedParallelMapDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(parallel_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_map_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(ParallelMapDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(parallel_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_map_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::ParallelMap");
}

TEST_P(ParameterizedParallelMapDatasetOpTest, Roundtrip) {
  int thread_num = 3, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  TF_ASSERT_OK(CreateDataset(parallel_map_dataset_kernel.get(),
                             parallel_map_dataset_context.get(),
                             &parallel_map_dataset));
  core::ScopedUnref scoped_unref_map_dataset(parallel_map_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(parallel_map_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(parallel_map_dataset->MakeIterator(iterator_ctx.get(),
                                                  "Iterator", &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());

    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(iterator->Restore(iterator_ctx.get(), &reader));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*expect_items_equal*/ test_case.sloppy));
}

TEST_F(ParallelMapDatasetOpTest, InvalidNumParallelCalls) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = InvalidNumParallelCallsTestCase();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> parallel_map_dataset_kernel;
  TF_ASSERT_OK(CreateParallelMapDatasetOpKernel(
      test_case.func, test_case.expected_output_dtypes,
      test_case.expected_output_shapes, test_case.use_inter_op_parallelism,
      test_case.sloppy, test_case.preserve_cardinality,
      &parallel_map_dataset_kernel));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(
      test_case.range_data_param.start, test_case.range_data_param.end,
      test_case.range_data_param.step, "range", &range_dataset));
  Tensor range_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_ASSERT_OK(
      StoreDatasetInVariantTensor(range_dataset, &range_dataset_tensor));
  Tensor num_parallel_calls = test_case.num_parallel_calls;
  gtl::InlinedVector<TensorValue, 4> parallel_map_dataset_inputs(
      {&range_dataset_tensor, &num_parallel_calls});

  std::unique_ptr<OpKernelContext> parallel_map_dataset_context;
  TF_ASSERT_OK(CreateParallelMapDatasetContext(
      parallel_map_dataset_kernel.get(), &parallel_map_dataset_inputs,
      &parallel_map_dataset_context));
  DatasetBase* parallel_map_dataset;
  EXPECT_EQ(
      CreateDataset(parallel_map_dataset_kernel.get(),
                    parallel_map_dataset_context.get(), &parallel_map_dataset)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

INSTANTIATE_TEST_SUITE_P(ParallelMapDatasetOpTest,
                         ParameterizedParallelMapDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3(),
                              TestCase4(), TestCase5(), TestCase6()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
