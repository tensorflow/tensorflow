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
constexpr int64 kRandomSeed = 42;
constexpr int64 kRandomSeed2 = 7;

class SamplingDatasetParams : public DatasetParams {
 public:
  template <typename T>
  SamplingDatasetParams(T input_dataset_params, float rate,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        rate_(rate) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    Tensor rate = CreateTensor<float>(TensorShape({}), {rate_});
    Tensor seed_tensor = CreateTensor<int64>(TensorShape({}), {seed_tensor_});
    Tensor seed2_tensor = CreateTensor<int64>(TensorShape({}), {seed2_tensor_});
    return {rate, seed_tensor, seed2_tensor};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    *input_names = {SamplingDatasetOp::kInputDataset, SamplingDatasetOp::kRate,
                    SamplingDatasetOp::kSeed, SamplingDatasetOp::kSeed2};

    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{SamplingDatasetOp::kOutputTypes, output_dtypes_},
                    {SamplingDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  string dataset_type() const override {
    return SamplingDatasetOp::kDatasetType;
  }

 private:
  // Target sample rate, range (0,1], wrapped in a scalar Tensor
  float rate_;
  // Boxed versions of kRandomSeed and kRandomSeed2.
  int64 seed_tensor_ = kRandomSeed;
  int64 seed2_tensor_ = kRandomSeed2;
};

class SamplingDatasetOpTest : public DatasetOpsTestBase {};

SamplingDatasetParams OneHundredPercentSampleParams() {
  return SamplingDatasetParams(RangeDatasetParams(0, 3, 1),
                               /*rate=*/1.0,
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/kNodeName);
}

SamplingDatasetParams TenPercentSampleParams() {
  return SamplingDatasetParams(RangeDatasetParams(0, 20, 1),
                               /*rate=*/0.1,
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/kNodeName);
}

SamplingDatasetParams ZeroPercentSampleParams() {
  return SamplingDatasetParams(RangeDatasetParams(0, 20, 1),
                               /*rate=*/0.0,
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/kNodeName);
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
                         GetNextTestCases())

std::vector<DatasetNodeNameTestCase<SamplingDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                         DatasetNodeNameTestCases())

std::vector<DatasetTypeStringTestCase<SamplingDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               SamplingDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                           DatasetTypeStringTestCases())

std::vector<DatasetOutputDtypesTestCase<SamplingDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<SamplingDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<SamplingDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/OneHundredPercentSampleParams(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/TenPercentSampleParams(),
           /*expected,cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ZeroPercentSampleParams(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<SamplingDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<SamplingDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                              IteratorOutputShapesTestCases())

std::vector<IteratorPrefixTestCase<SamplingDatasetParams>>
IteratorOutputPrefixTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               SamplingDatasetOp::kDatasetType,
               TenPercentSampleParams().iterator_prefix())}};
}

ITERATOR_PREFIX_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                       IteratorOutputPrefixTestCases())

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
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
