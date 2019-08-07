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
constexpr char kIteratorPrefix[] = "Iterator";
constexpr int64 kRandomSeed = 42;
constexpr int64 kRandomSeed2 = 7;
constexpr int64 kStart = 0;
constexpr int64 kStep = 1;

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
};

// TODO(frreiss): Remove this once #31344 goes in and RangeDatasetParams is
// defined in dataset_test_base.h
class LocalRangeDatasetParams : public DatasetParams {
 public:
  LocalRangeDatasetParams(int64 start, int64 num_elements, int64 step,
                          DataTypeVector output_dtypes,
                          std::vector<PartialTensorShape> output_shapes,
                          string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        start(CreateTensor<int64>(TensorShape({}), {start})),
        num_elements(CreateTensor<int64>(TensorShape({}), {num_elements})),
        step(CreateTensor<int64>(TensorShape({}), {step})) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    *inputs = {TensorValue(&start), TensorValue(&num_elements),
               TensorValue(&step)};
    return Status::OK();
  }

  Tensor start;
  Tensor num_elements;
  Tensor step;
};

class SamplingDatasetParams : public DatasetParams {
 public:
  SamplingDatasetParams(float rate, int64 num_elements,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        rate(CreateTensor<float>(TensorShape({}), {rate})),
        range_dataset_params(kStart, num_elements, kStep, {DT_INT64},
                             {PartialTensorShape({})}, "") {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (input_dataset.NumElements() == 0 ||
        input_dataset.dtype() != DT_VARIANT) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset), TensorValue(&rate),
               TensorValue(&seed_tensor_), TensorValue(&seed2_tensor_)};
    return Status::OK();
  }

  // Static parameters of the kernel
  Tensor rate;

  // Parameters of the sequence of numbers that will serve as the dynamic input
  // of the kernel.
  LocalRangeDatasetParams range_dataset_params;

  // RangeDataset kernel wrapped in a variant tensor. Initialized by the test
  // case itself because the MakeRangeDataset() method requires an instance of
  // DatasetOpsTestBase.
  Tensor input_dataset;

 private:
  // Boxed versions of kRandomSeed and kRandomSeed2.
  Tensor seed_tensor_ = CreateTensor<int64>(TensorShape({}), {kRandomSeed});
  Tensor seed2_tensor_ = CreateTensor<int64>(TensorShape({}), {kRandomSeed2});
};

SamplingDatasetParams OneHundredPercentSampleDataset() {
  return {/*rate*/ 1.0,
          /*num_elements*/ 3,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

SamplingDatasetParams TenPercentSampleDataset() {
  return {/*rate*/ 0.1,
          /*num_elements*/ 20,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

SamplingDatasetParams ZeroPercentSampleDataset() {
  return {/*rate*/ 0.0,
          /*num_elements*/ 20,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

class ParameterizedGetNextSamplingDatasetOpTest
    : public SamplingDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<SamplingDatasetParams>> {};

// Test case 1: 100% sample should return all inputs
GetNextTestCase<SamplingDatasetParams> GetNextTestCase1() {
  return {/*dataset_params=*/OneHundredPercentSampleDataset(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})};
}

// Test case 2: 10% sample should return about 10% of inputs, and the specific
// inputs returned shouldn't change across build environments.
GetNextTestCase<SamplingDatasetParams> GetNextTestCase2() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{9}, {11}, {19}})};
}

// Test case 3: 0% sample should return nothing and should not crash.
GetNextTestCase<SamplingDatasetParams> GetNextTestCase3() {
  return {/*dataset_params=*/ZeroPercentSampleDataset(),
          /*expected_outputs=*/{}};
}

TEST_P(ParameterizedGetNextSamplingDatasetOpTest, GetNext) {
  // BEGIN INITIALIZATION CODE
  // This test case and all the other test cases in this file go through the
  // same sequence of initialization steps.
  // Tests that don't examine the results of the op skip step 7.

  // Step 1: Set up enough of a TF runtime to be able to invoke a kernel.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  // Step 2: Create the dataset that will provide input data for the kernel
  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  // Step 3: Box up the four inputs to the kernel inside TensorValue objects
  // inside a vector.
  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  // Step 4: Create a SamplingDataset kernel to test, passing in attributes
  // of the kernel.
  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

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
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              kIteratorPrefix, &iterator));
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

INSTANTIATE_TEST_SUITE_P(
    SamplingDatasetOpTest, ParameterizedGetNextSamplingDatasetOpTest,
    ::testing::ValuesIn(std::vector<GetNextTestCase<SamplingDatasetParams>>(
        {GetNextTestCase1(), GetNextTestCase2(), GetNextTestCase3()})));

DatasetNodeNameTestCase<SamplingDatasetParams> DatasetNodeNameTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_node_name=*/kNodeName};
}

// Verify that the machinery for creating SamplingDataset kernels runs and
// correctly creates kernels of with the node name "SamplingDataset".
TEST_F(SamplingDatasetOpTest, DatasetNodeName) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = DatasetNodeNameTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(
      CheckDatasetNodeName(*sampling_dataset, test_case.expected_node_name));
}

DatasetTypeStringTestCase<SamplingDatasetParams> DatasetTypeStringTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_dataset_type_string=*/
          name_utils::OpName(SamplingDatasetOp::kDatasetType)};
}

TEST_F(SamplingDatasetOpTest, DatasetTypeString) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = DatasetTypeStringTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(CheckDatasetTypeString(*sampling_dataset,
                                      test_case.expected_dataset_type_string));
}

DatasetOutputDtypesTestCase<SamplingDatasetParams>
DatasetOutputDtypesTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(SamplingDatasetOpTest, DatasetOutputDtypes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = DatasetOutputDtypesTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(CheckDatasetOutputDtypes(*sampling_dataset,
                                        test_case.expected_output_dtypes));
}

DatasetOutputShapesTestCase<SamplingDatasetParams>
DatasetOutputShapesTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_output_shapes=*/{PartialTensorShape({})}};
}

TEST_F(SamplingDatasetOpTest, DatasetOutputShapes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = DatasetOutputShapesTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(CheckDatasetOutputShapes(*sampling_dataset,
                                        test_case.expected_output_shapes));
}

class ParameterizedCardinalitySamplingDatasetOpTest
    : public SamplingDatasetOpTest,
      public ::testing::WithParamInterface<
          CardinalityTestCase<SamplingDatasetParams>> {};

CardinalityTestCase<SamplingDatasetParams> CardinalityTestCase1() {
  return {/*dataset_params=*/OneHundredPercentSampleDataset(),
          /*expected_cardinality=*/kUnknownCardinality};
}

CardinalityTestCase<SamplingDatasetParams> CardinalityTestCase2() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_cardinality=*/kUnknownCardinality};
}

CardinalityTestCase<SamplingDatasetParams> CardinalityTestCase3() {
  return {/*dataset_params=*/ZeroPercentSampleDataset(),
          /*expected_cardinality=*/kUnknownCardinality};
}

TEST_P(ParameterizedCardinalitySamplingDatasetOpTest, Cardinality) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(CheckDatasetCardinality(*sampling_dataset,
                                       test_case.expected_cardinality));
}

INSTANTIATE_TEST_SUITE_P(
    SamplingDatasetOpTest, ParameterizedCardinalitySamplingDatasetOpTest,
    ::testing::ValuesIn(std::vector<CardinalityTestCase<SamplingDatasetParams>>(
        {CardinalityTestCase1(), CardinalityTestCase2(),
         CardinalityTestCase3()})));

DatasetSaveTestCase<SamplingDatasetParams> DatasetSaveTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset()};
}

TEST_F(SamplingDatasetOpTest, DatasetSave) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = DatasetSaveTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(CheckDatasetSave(*sampling_dataset));
}

IsStatefulTestCase<SamplingDatasetParams> IsStatefulTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_stateful=*/false};
}

TEST_F(SamplingDatasetOpTest, IsStateful) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = IsStatefulTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

  std::unique_ptr<OpKernelContext> sampling_dataset_context;
  TF_ASSERT_OK(CreateSamplingDatasetContext(
      sampling_dataset_kernel.get(), &inputs, &sampling_dataset_context));

  DatasetBase* sampling_dataset;
  TF_ASSERT_OK(CreateDataset(sampling_dataset_kernel.get(),
                             sampling_dataset_context.get(),
                             &sampling_dataset));
  core::ScopedUnref scoped_unref(sampling_dataset);
  // END INITIALIZATION CODE

  TF_ASSERT_OK(
      CheckDatasetIsStateful(*sampling_dataset, test_case.expected_stateful));
}

IteratorOutputDtypesTestCase<SamplingDatasetParams>
IteratorOutputDtypesTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(SamplingDatasetOpTest, IteratorOutputDtypes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = IteratorOutputDtypesTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

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
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              kIteratorPrefix, &iterator));
  // END INITIALIZATION CODE

  TF_ASSERT_OK(
      CheckIteratorOutputDtypes(*iterator, test_case.expected_output_dtypes));
}

IteratorOutputShapesTestCase<SamplingDatasetParams>
IteratorOutputShapesTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_output_shapes=*/{PartialTensorShape({})}};
}

TEST_F(SamplingDatasetOpTest, IteratorOutputShapes) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = IteratorOutputShapesTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

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
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              kIteratorPrefix, &iterator));
  // END INITIALIZATION CODE

  TF_ASSERT_OK(
      CheckIteratorOutputShapes(*iterator, test_case.expected_output_shapes));
}

IteratorOutputPrefixTestCase<SamplingDatasetParams>
IteratorOutputPrefixTestCase1() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*expected_iterator_prefix=*/
          name_utils::IteratorPrefix(SamplingDatasetOp::kDatasetType,
                                     kIteratorPrefix)};
}

TEST_F(SamplingDatasetOpTest, IteratorOutputPrefix) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = IteratorOutputPrefixTestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

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
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              kIteratorPrefix, &iterator));
  // END INITIALIZATION CODE

  TF_ASSERT_OK(
      CheckIteratorPrefix(*iterator, test_case.expected_iterator_prefix));
}

class ParameterizedIteratorSaveAndRestoreSamplingDatasetOpTest
    : public SamplingDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<SamplingDatasetParams>> {};

IteratorSaveAndRestoreTestCase<SamplingDatasetParams>
IteratorSaveAndRestoreTestCase1() {
  return {/*dataset_params=*/OneHundredPercentSampleDataset(),
          /*breakpoints=*/{0, 2, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})};
}

IteratorSaveAndRestoreTestCase<SamplingDatasetParams>
IteratorSaveAndRestoreTestCase2() {
  return {/*dataset_params=*/TenPercentSampleDataset(),
          /*breakpoints=*/{0, 2, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({}), {{9}, {11}, {19}})};
}

IteratorSaveAndRestoreTestCase<SamplingDatasetParams>
IteratorSaveAndRestoreTestCase3() {
  return {/*dataset_params=*/ZeroPercentSampleDataset(),
          /*breakpoints=*/{0, 2, 5},
          /*expected_outputs=*/{}};
}

// Save and restore the dataset while scanning it. Verify the returned tuples.
TEST_P(ParameterizedIteratorSaveAndRestoreSamplingDatasetOpTest, Roundtrip) {
  // BEGIN INITIALIZATION CODE
  // See ParameterizedGetNextSamplingDatasetOpTest::GetNext for explanatory
  // comments.
  const int thread_num = 2, cpu_num = 2;
  auto test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.num_elements,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernel> sampling_dataset_kernel;
  TF_ASSERT_OK(CreateSamplingDatasetOpKernel(
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &sampling_dataset_kernel));

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
  TF_ASSERT_OK(sampling_dataset->MakeIterator(iterator_context.get(),
                                              kIteratorPrefix, &iterator));
  // END INITIALIZATION CODE

  TF_ASSERT_OK(CheckIteratorSaveAndRestore(
      *sampling_dataset, iterator_context.get(), kIteratorPrefix,
      test_case.expected_outputs, test_case.breakpoints));
}

INSTANTIATE_TEST_SUITE_P(
    SamplingDatasetOpTest,
    ParameterizedIteratorSaveAndRestoreSamplingDatasetOpTest,
    ::testing::ValuesIn(
        std::vector<IteratorSaveAndRestoreTestCase<SamplingDatasetParams>>(
            {IteratorSaveAndRestoreTestCase1(),
             IteratorSaveAndRestoreTestCase2(),
             IteratorSaveAndRestoreTestCase3()})));

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
