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
#include "tensorflow/core/kernels/data/experimental/assert_next_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "assert_next_dataset";

struct RangeDatasetParams {
  int start;
  int stop;
  int step;
};

struct TakeDatasetParams {
  int count;
};

class AssertNextDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `AssertNextDataset` op kernel.
  Status CreateAssertNextDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* assert_next_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(AssertNextDatasetOp::kDatasetType),
        {AssertNextDatasetOp::kInputDataset,
         AssertNextDatasetOp::kTransformations},
        {{AssertNextDatasetOp::kOutputTypes, output_types},
         {AssertNextDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, assert_next_dataset_op_kernel));
    return Status::OK();
  }

  // Creates a new `AssertNextDataset` op kernel context.
  Status CreateAssertNextDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }

  // Creates a new `RangeAndTakeDataset` tensor.
  Status MakeRangeAndTakeDatasetTensor(
      const RangeDatasetParams& range_dataset_params,
      const TakeDatasetParams& take_dataset_params,
      Tensor* range_and_take_dataset_tensor) {
    Tensor range_dataset_tensor;
    Tensor start =
        CreateTensor<int64>(TensorShape({}), {range_dataset_params.start});
    Tensor stop =
        CreateTensor<int64>(TensorShape({}), {range_dataset_params.stop});
    Tensor step =
        CreateTensor<int64>(TensorShape({}), {range_dataset_params.step});
    TF_RETURN_IF_ERROR(MakeRangeDataset(start, stop, step, {DT_INT64},
                                        {PartialTensorShape({})},
                                        &range_dataset_tensor));

    TF_RETURN_IF_ERROR(MakeTakeDataset(
        range_dataset_tensor, take_dataset_params.count, {DT_INT64},
        {PartialTensorShape({})}, range_and_take_dataset_tensor));
    return Status::OK();
  }
};

struct TestCase {
  RangeDatasetParams range_dataset_params;
  TakeDatasetParams take_dataset_params;
  Tensor transformations;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

// Test case 1 : assert one transformation.
TestCase TestCase1() {
  return {/*range_dataset_params*/ {/*start*/ 0, /*stop*/ 10, /*step*/ 1},
          /*take_dataset_params*/ {/*count*/ 3},
          /*transformations*/
          CreateTensor<string>(TensorShape({1}), {TakeDatasetOp::kDatasetType}),
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {0}),
           CreateTensor<int64>(TensorShape({}), {1}),
           CreateTensor<int64>(TensorShape({}), {2})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 2 : assert two transformations.
TestCase TestCase2() {
  return {
      /*range_dataset_params*/ {/*start*/ 0, /*stop*/ 10, /*step*/ 1},
      /*take_dataset_params*/ {/*count*/ 3},
      /*transformations*/
      CreateTensor<string>(TensorShape({2}), {TakeDatasetOp::kDatasetType,
                                              RangeDatasetOp::kDatasetType}),
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({}), {0}),
       CreateTensor<int64>(TensorShape({}), {1}),
       CreateTensor<int64>(TensorShape({}), {2})},
      /*expected_output_dtypes*/ {DT_INT64},
      /*expected_output_shapes*/ {PartialTensorShape({})},
      /*expected_cardinality*/ 3,
      /*breakpoints*/ {0, 2, 5}};
}

TestCase AssertNextInvalid() {
  return {/*range_dataset_params*/ {/*start*/ 0, /*stop*/ 10, /*step*/ 1},
          /*take_dataset_params*/ {/*count*/ 3},
          /*transformations*/
          CreateTensor<string>(TensorShape({1}), {"Whoops"}),
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {0}),
           CreateTensor<int64>(TensorShape({}), {1}),
           CreateTensor<int64>(TensorShape({}), {2})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 5}};
}

TestCase AssertNextShort() {
  return {/*range_dataset_params*/ {/*start*/ 0, /*stop*/ 10, /*step*/ 1},
          /*take_dataset_params*/ {/*count*/ 3},
          /*transformations*/
          CreateTensor<string>(TensorShape({3}),
                               {TakeDatasetOp::kDatasetType,
                                RangeDatasetOp::kDatasetType, "Whoops"}),
          /*expected_outputs*/
          {CreateTensor<int64>(TensorShape({}), {0}),
           CreateTensor<int64>(TensorShape({}), {1}),
           CreateTensor<int64>(TensorShape({}), {2})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 3,
          /*breakpoints*/ {0, 2, 5}};
}

class ParameterizedAssertNextDatasetOpTest
    : public AssertNextDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedAssertNextDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(assert_next_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(assert_next_dataset->MakeIterator(iterator_context.get(),
                                                 iterator_prefix, &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_context.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

TEST_F(AssertNextDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  EXPECT_EQ(assert_next_dataset->node_name(), kNodeName);
}

TEST_P(ParameterizedAssertNextDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  EXPECT_EQ(assert_next_dataset->type_string(),
            name_utils::OpName(AssertNextDatasetOp::kDatasetType));
}

TEST_P(ParameterizedAssertNextDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(assert_next_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedAssertNextDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(assert_next_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedAssertNextDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  EXPECT_EQ(assert_next_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedAssertNextDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(assert_next_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(assert_next_dataset->MakeIterator(iterator_context.get(),
                                                 iterator_prefix, &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedAssertNextDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(assert_next_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(assert_next_dataset->MakeIterator(iterator_context.get(),
                                                 iterator_prefix, &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedAssertNextDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(assert_next_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(assert_next_dataset->MakeIterator(iterator_context.get(),
                                                 iterator_prefix, &iterator));

  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(AssertNextDatasetOp::kDatasetType,
                                       iterator_prefix));
}

TEST_P(ParameterizedAssertNextDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  std::unique_ptr<OpKernel> assert_next_dataset_kernel;
  TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(test_case.expected_output_dtypes,
                                               test_case.expected_output_shapes,
                                               &assert_next_dataset_kernel));
  Tensor transformations = test_case.transformations;
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor),
       TensorValue(&transformations)});
  std::unique_ptr<OpKernelContext> assert_next_dataset_context;
  TF_ASSERT_OK(CreateAssertNextDatasetContext(
      assert_next_dataset_kernel.get(), &inputs, &assert_next_dataset_context));

  DatasetBase* assert_next_dataset;
  TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                             assert_next_dataset_context.get(),
                             &assert_next_dataset));
  core::ScopedUnref scoped_unref(assert_next_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(assert_next_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(assert_next_dataset->MakeIterator(iterator_context.get(),
                                                 iterator_prefix, &iterator));

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
    TF_EXPECT_OK(RestoreIterator(iterator_context.get(), &reader,
                                 iterator_prefix, *assert_next_dataset,
                                 &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_context.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      ++cur_iteration;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

INSTANTIATE_TEST_SUITE_P(
    AssertNextDatasetOpTest, ParameterizedAssertNextDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestCase>({TestCase1(), TestCase2()})));

TEST_F(AssertNextDatasetOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::vector<TestCase> test_cases = {AssertNextInvalid(), AssertNextShort()};
  for (TestCase test_case : test_cases) {
    Tensor range_and_take_dataset_tensor;
    TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                               test_case.take_dataset_params,
                                               &range_and_take_dataset_tensor));

    std::unique_ptr<OpKernel> assert_next_dataset_kernel;
    TF_ASSERT_OK(CreateAssertNextDatasetOpKernel(
        test_case.expected_output_dtypes, test_case.expected_output_shapes,
        &assert_next_dataset_kernel));
    Tensor transformations = test_case.transformations;
    gtl::InlinedVector<TensorValue, 4> inputs(
        {TensorValue(&range_and_take_dataset_tensor),
         TensorValue(&transformations)});
    std::unique_ptr<OpKernelContext> assert_next_dataset_context;
    TF_ASSERT_OK(
        CreateAssertNextDatasetContext(assert_next_dataset_kernel.get(),
                                       &inputs, &assert_next_dataset_context));

    DatasetBase* assert_next_dataset;
    TF_ASSERT_OK(CreateDataset(assert_next_dataset_kernel.get(),
                               assert_next_dataset_context.get(),
                               &assert_next_dataset));
    core::ScopedUnref scoped_unref(assert_next_dataset);

    std::unique_ptr<IteratorContext> iterator_context;
    TF_ASSERT_OK(CreateIteratorContext(assert_next_dataset_context.get(),
                                       &iterator_context));
    std::unique_ptr<IteratorBase> iterator;
    string iterator_prefix = name_utils::IteratorPrefix(
        TakeDatasetOp::kDatasetType,
        name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
    EXPECT_EQ(
        assert_next_dataset
            ->MakeIterator(iterator_context.get(), iterator_prefix, &iterator)
            .code(),
        tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
