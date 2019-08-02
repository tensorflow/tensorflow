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
#include "tensorflow/core/kernels/data/experimental/sampling_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "sampling_dataset";

// Parameters for constructing a dataset that returns an ordered sequence
// of numbers
struct RangeDatasetParams {
  int start;
  int stop;
  int step;
};

struct TakeDatasetParams {
  int count;
};

class SamplingDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `SamplingDataset` op kernel.
  // Doesn't initialize the kernel's static parameters because they are inputs,
  // not attributes.
  Status CreateSamplingDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* sampling_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(SamplingDatasetOp::kDatasetType),
        // Inputs
        {SamplingDatasetOp::kInputDataset, SamplingDatasetOp::kRate,
         SamplingDatasetOp::kSeed, SamplingDatasetOp::kSeed2},
        // Attributes
        {{SamplingDatasetOp::kOutputTypes, output_types},
         {SamplingDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, sampling_dataset_op_kernel));
    return Status::OK();
  }

  // Creates an OpKernel context suitable for running a `SamplingDataset`
  // kernel.
  Status CreateSamplingDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }

  // Build a dataset that will return an ordered sequence of numbers in chunks
  // of size `params.count`.
  // Stuffs the returned dataset into a variant tensor.
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

// Common parameters that every test case in this file shares
struct TestCase {
  // Static parameters of the kernel
  float rate;
  int64 seed;
  int64 seed2;

  // Parameters of the sequence of numbers that will serve as the dynamic input
  // of the kernel.
  RangeDatasetParams range_dataset_params;
  TakeDatasetParams take_dataset_params;

  // The tensors that the kernel is expected to return, in the order they
  // should be returned
  std::vector<Tensor> expected_outputs;

  // Information about the returned outputs of the op that the test case
  // creates.
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;

  // Value that the dataset's Cardinality() function returns. May be different
  // from the size of the outputs, as Cardinality() is not supposed to perform
  // expensive computations.
  int64 expected_cardinality;

  // When to insert save and restore steps while scanning the dataset in the
  // "roundtrip" test case.
  std::vector<int> breakpoints;
};

// Test case 1: 100% sample should return all inputs
TestCase TestCase1() {
  return {/*rate*/ 1.0,
          /*seed*/ 42,
          /*seed2*/ 7,
          /*range_dataset_params*/ {/*start*/ 0, /*stop*/ 10, /*step*/ 1},
          /*take_dataset_params*/ {/*count*/ 3},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {0}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {1}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {2})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 2: 10% sample should return about 10% of inputs, and the specific
// inputs returned shouldn't change across build environments.
TestCase TestCase2() {
  return {/*rate*/ 0.1,
          /*seed*/ 42,
          /*seed2*/ 7,
          /*range_dataset_params*/ {/*start*/ 0, /*stop*/ 100, /*step*/ 1},
          /*take_dataset_params*/ {/*count*/ 20},
          /*expected_outputs*/
          {DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {9}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {11}),
           DatasetOpsTestBase::CreateTensor<int64>(TensorShape({}), {19})},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 5}};
}

// Test case 3: 0% sample should return nothing and should not crash.
TestCase TestCase3() {
  return {/*rate*/ 0.0,
          /*seed*/ 42,
          /*seed2*/ 7,
          /*range_dataset_params*/ {/*start*/ 0, /*stop*/ 100, /*step*/ 1},
          /*take_dataset_params*/ {/*count*/ 20},
          /*expected_outputs*/
          {},
          /*expected_output_dtypes*/ {DT_INT64},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 5}};
}

// Parameterized test class shared by the next 6 test cases
class ParameterizedSamplingDatasetOpTest
    : public SamplingDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

// Verify that the GetNext function works and returns the expected outputs
TEST_P(ParameterizedSamplingDatasetOpTest, GetNext) {
  // BEGIN INITIALIZATION CODE
  // This test case and all the other test cases in this file go through the
  // same sequence of initialization steps.
  // Tests that don't examine the results of the op skip step 7.

  // Step 1: Set up enough of a TF runtime to be able to invoke a kernel.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  // Step 2: Create the dataset that will provide input data for the kernel
  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  // Step 3: Box up the four inputs to the kernel inside TensorValue objects
  // inside a vector.
  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  // Step 4: Create a SamplingDataset kernel to test, passing in attributes
  // of the kernel.
  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  // Step 5: Create a context in which the kernel will operate. This is where
  // the kernel gets initialized with its inputs
  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  // Step 6: Unbox the DatasetBase inside the variant tensor backing the
  // kernel.
  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);

  // Step 7: Create an iterator to read the output of the dataset.
  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(sampling_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              iterator_prefix, &iterator));
  // END INITIALIZATION CODE

  // Copy the iterator's output into a vector to make comparison easier.
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

// Verify that the machinery for creating SamplingDataset kernels runs and
// correctly creates kernels of with the node name "SamplingDataset".
TEST_F(SamplingDatasetOpTest, DatasetNodeName) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  EXPECT_EQ(sampling_dataset->node_name(), kNodeName);
}

TEST_P(ParameterizedSamplingDatasetOpTest, DatasetTypeString) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  EXPECT_EQ(sampling_dataset->type_string(),
            name_utils::OpName(SamplingDatasetOp::kDatasetType));
}

TEST_P(ParameterizedSamplingDatasetOpTest, DatasetOutputDtypes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_EXPECT_OK(VerifyTypesMatch(sampling_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedSamplingDatasetOpTest, DatasetOutputShapes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_EXPECT_OK(VerifyShapesCompatible(sampling_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedSamplingDatasetOpTest, Cardinality) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  EXPECT_EQ(sampling_dataset->Cardinality(), test_case.expected_cardinality);
}

// Verify that the Save() function executes without raising an error.
TEST_P(ParameterizedSamplingDatasetOpTest, DatasetSave) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(sampling_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_P(ParameterizedSamplingDatasetOpTest, IteratorOutputDtypes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(sampling_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              iterator_prefix, &iterator));
  // END INITIALIZATION CODE

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedSamplingDatasetOpTest, IteratorOutputShapes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(sampling_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              iterator_prefix, &iterator));
  // END INITIALIZATION CODE

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedSamplingDatasetOpTest, IteratorOutputPrefix) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(sampling_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              iterator_prefix, &iterator));
  // END INITIALIZATION CODE

  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(SamplingDatasetOp::kDatasetType,
                                       iterator_prefix));
}

// Save and restore the dataset while scanning it. Verify the returned tuples.
TEST_P(ParameterizedSamplingDatasetOpTest, Roundtrip) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedSamplingDatasetOpTest::GetNext for explanatory comments.
  const int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_and_take_dataset_tensor;
  TF_ASSERT_OK(MakeRangeAndTakeDatasetTensor(test_case.range_dataset_params,
                                             test_case.take_dataset_params,
                                             &range_and_take_dataset_tensor));

  Tensor rate = CreateTensor<float>(TensorShape({}), {test_case.rate});
  Tensor seed = CreateTensor<int64>(TensorShape({}), {test_case.seed});
  Tensor seed2 = CreateTensor<int64>(TensorShape({}), {test_case.seed2});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&range_and_take_dataset_tensor), TensorValue(&rate),
       TensorValue(&seed), TensorValue(&seed2)});

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(test_case.expected_output_dtypes,
                                             test_case.expected_output_shapes,
                                             &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(sampling_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  string iterator_prefix = name_utils::IteratorPrefix(
      TakeDatasetOp::kDatasetType,
      name_utils::IteratorPrefix(RangeDatasetOp::kDatasetType, "Iterator"));
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              iterator_prefix, &iterator));
  // END INITIALIZATION CODE

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
                                 iterator_prefix, *sampling_dataset,
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

INSTANTIATE_TEST_SUITE_P(SamplingDatasetOpTest,
                         ParameterizedSamplingDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
