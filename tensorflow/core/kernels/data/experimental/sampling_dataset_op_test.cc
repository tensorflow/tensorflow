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

  // Target sample rate, range (0,1], wrapped in a scalar Tensor
  Tensor rate;

  // Parameters of the sequence of numbers that will serve as the dynamic input
  // of the kernel.
  RangeDatasetParams range_dataset_params;

  // RangeDataset kernel wrapped in a variant tensor. Initialized by the test
  // harness class because the MakeRangeDataset() method requires an instance of
  // DatasetOpsTestBase.
  Tensor input_dataset;

 private:
  // Boxed versions of kRandomSeed and kRandomSeed2.
  Tensor seed_tensor_ = CreateTensor<int64>(TensorShape({}), {kRandomSeed});
  Tensor seed2_tensor_ = CreateTensor<int64>(TensorShape({}), {kRandomSeed2});
};

class SamplingDatasetOpTest
    : public DatasetOpsTestBaseV2<SamplingDatasetParams> {
 public:
  Status Initialize(SamplingDatasetParams* dataset_params) override {
    // Step 1: Set up enough of a TF runtime to be able to invoke a kernel.
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime({}, cpu_num_));

    // Step 2: Create the dataset that will provide input data for the kernel
    TF_RETURN_IF_ERROR(MakeRangeDataset(dataset_params->range_dataset_params,
                                        &dataset_params->input_dataset));

    // Step 3: Box up the four inputs to the kernel inside TensorValue objects
    // inside a vector.
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(dataset_params->MakeInputs(&inputs));

    // Step 4: Create a dataset kernel to test, passing in attributes of the
    // kernel.
    TF_RETURN_IF_ERROR(MakeDatasetOpKernel(*dataset_params, &dataset_kernel_));

    // Step 5: Create a context in which the kernel will operate. This is where
    // the kernel gets initialized with its inputs
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));

    // Step 6: Unbox the DatasetBase object inside the variant tensor backing
    // the kernel.
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));

    // Step 7: Create an iterator in case the test needs to read the output of
    // the dataset.
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(iterator_ctx_.get(),
                                              kIteratorPrefix, &iterator_));

    return Status::OK();
  }

  // Creates a new `SamplingDataset` op kernel.
  // Doesn't initialize the kernel's static parameters because they are inputs,
  // not attributes.
  Status MakeDatasetOpKernel(
      const SamplingDatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* sampling_dataset_op_kernel) override {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(SamplingDatasetOp::kDatasetType),
        // Inputs
        {SamplingDatasetOp::kInputDataset, SamplingDatasetOp::kRate,
         SamplingDatasetOp::kSeed, SamplingDatasetOp::kSeed2},
        // Attributes
        {{SamplingDatasetOp::kOutputTypes, dataset_params.output_dtypes},
         {SamplingDatasetOp::kOutputShapes, dataset_params.output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, sampling_dataset_op_kernel));
    return Status::OK();
  }
};

SamplingDatasetParams OneHundredPercentSampleParams() {
  return {/*rate*/ 1.0,
          /*num_elements*/ 3,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

SamplingDatasetParams TenPercentSampleParams() {
  return {/*rate*/ 0.1,
          /*num_elements*/ 20,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

SamplingDatasetParams ZeroPercentSampleParams() {
  return {/*rate*/ 0.0,
          /*num_elements*/ 20,
          /*output_dtypes*/ {DT_INT64},
          /*output_shapes*/ {PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<SamplingDatasetParams>> GetNextTestCases() {
  return {
      // Test case 1: 100% sample should return all inputs
      {/*dataset_params=*/OneHundredPercentSampleParams(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape({}),
                                                 {{0}, {1}, {2}})},

      // Test case 2: 10% sample should return about 10% of inputs, and the
      // specific inputs returned shouldn't change across build environments.
      {/*dataset_params=*/TenPercentSampleParams(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape({}),
                                                 {{9}, {11}, {19}})},

      // Test case 3: 0% sample should return nothing and should not crash.
      {/*dataset_params=*/ZeroPercentSampleParams(), /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                         GetNextTestCases());

std::vector<DatasetNodeNameTestCase<SamplingDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                         DatasetNodeNameTestCases());

std::vector<DatasetTypeStringTestCase<SamplingDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               SamplingDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                           DatasetTypeStringTestCases());

std::vector<DatasetOutputDtypesTestCase<SamplingDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                             DatasetOutputDtypesTestCases());

std::vector<DatasetOutputShapesTestCase<SamplingDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                             DatasetOutputShapesTestCases());

std::vector<CardinalityTestCase<SamplingDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/OneHundredPercentSampleParams(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/TenPercentSampleParams(),
           /*expected,cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ZeroPercentSampleParams(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                           CardinalityTestCases());

std::vector<IteratorOutputDtypesTestCase<SamplingDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                              IteratorOutputDtypesTestCases());

std::vector<IteratorOutputShapesTestCase<SamplingDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                              IteratorOutputShapesTestCases());

std::vector<IteratorPrefixTestCase<SamplingDatasetParams>>
IteratorOutputPrefixTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               SamplingDatasetOp::kDatasetType, kIteratorPrefix)}};
}

ITERATOR_PREFIX_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                       IteratorOutputPrefixTestCases());

std::vector<IteratorSaveAndRestoreTestCase<SamplingDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/OneHundredPercentSampleParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/TenPercentSampleParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{9}, {11}, {19}})},
          {/*dataset_params=*/ZeroPercentSampleParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                                 IteratorSaveAndRestoreTestCases());

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
