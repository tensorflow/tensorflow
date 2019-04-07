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

constexpr char kNodeName[] = "range_dataset";
constexpr char kOpName[] = "RangeDataset";

class RangeDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new RangeDataset op kernel context.
  Status CreateRangeDatasetContext(
      OpKernel* const range_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* range_context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*range_kernel, *inputs));
    TF_RETURN_IF_ERROR(
        CreateOpKernelContext(range_kernel, inputs, range_context));
    return Status::OK();
  }
};

struct TestCase {
  int64 start;
  int64 end;
  int64 step;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

TestCase PositiveStepTestCase() {
  return {/*start*/ 0,
          /*end*/ 10,
          /*step*/ 3,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {3}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {6}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {9})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 4}};
}

TestCase NegativeStepTestCase() {
  return {/*start*/ 10,
          /*end*/ 0,
          /*step*/ -3,
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {10}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {7}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {4}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ 4,
          /*breakpoints*/ {0, 1, 4}};
}

TestCase ZeroStepTestCase() {
  return {/*start*/ 0,
          /*end*/ 10,
          /*step*/ 0,
          /*expected_outputs*/ {},
          /*expected_output_dtypes*/ {},
          /*expected_output_shapes*/ {},
          /*expected_cardinality*/ 0,
          /*breakpoints*/ {}};
}

class ParameterizedRangeDatasetOpTest
    : public RangeDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedRangeDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = GetParam();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

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

TEST_F(RangeDatasetOpTest, ZeroStep) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = ZeroStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  EXPECT_EQ(CreateDataset(range_dataset_kernel.get(),
                          range_dataset_context.get(), &range_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(RangeDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  EXPECT_EQ(range_dataset->node_name(), kNodeName);
}

TEST_F(RangeDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  EXPECT_EQ(range_dataset->type_string(), kOpName);
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_EXPECT_OK(VerifyTypesMatch(range_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  TF_EXPECT_OK(VerifyShapesCompatible(range_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedRangeDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = GetParam();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  EXPECT_EQ(range_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_F(RangeDatasetOpTest, DatasetSave) {
  int64 thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));

  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(range_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_F(RangeDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = PositiveStepTestCase();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Range");
}

TEST_P(ParameterizedRangeDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TestCase test_case = GetParam();
  gtl::InlinedVector<TensorValue, 4> inputs;
  Tensor start = CreateTensor<int64>(TensorShape({}), {test_case.start});
  Tensor end = CreateTensor<int64>(TensorShape({}), {test_case.end});
  Tensor step = CreateTensor<int64>(TensorShape({}), {test_case.step});
  inputs.emplace_back(&start);
  inputs.emplace_back(&end);
  inputs.emplace_back(&step);

  std::unique_ptr<OpKernel> range_dataset_kernel;
  TF_ASSERT_OK(
      CreateRangeDatasetOpKernel<int64>(kNodeName, &range_dataset_kernel));
  std::unique_ptr<OpKernelContext> range_dataset_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(range_dataset_kernel.get(), &inputs,
                                         &range_dataset_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateDataset(range_dataset_kernel.get(),
                             range_dataset_context.get(), &range_dataset));
  core::ScopedUnref scoped_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(range_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

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
    TF_EXPECT_OK(iterator->Restore(iterator_context.get(), &reader));

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

INSTANTIATE_TEST_SUITE_P(
    RangeDatasetOpTest, ParameterizedRangeDatasetOpTest,
    ::testing::ValuesIn(std::vector<TestCase>({PositiveStepTestCase(),
                                               NegativeStepTestCase()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
