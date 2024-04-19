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
#include "tensorflow/core/kernels/data/zip_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "zip_dataset";

class ZipDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ZipDatasetParams(std::vector<T> input_dataset_params,
                   DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   int num_input_datasets, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_input_datasets_(num_input_datasets) {
    for (auto& params : input_dataset_params) {
      input_dataset_params_.push_back(std::make_unique<T>(params));
    }

    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params[0].dataset_type(),
                                   input_dataset_params[0].iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    for (int i = 0; i < num_input_datasets_; ++i) {
      input_names->emplace_back(
          absl::StrCat(ZipDatasetOp::kDatasetType, "_", i));
    }
    return absl::OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("N", num_input_datasets_);
    attr_vector->emplace_back("metadata", "");
    return absl::OkStatus();
  }

  string dataset_type() const override { return ZipDatasetOp::kDatasetType; }

 private:
  int32 num_input_datasets_;
};

class ZipDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: the input datasets with same number of outputs.
ZipDatasetParams ZipDatasetParams1() {
  return ZipDatasetParams(
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

// Test case 2: the input datasets with different number of outputs.
ZipDatasetParams ZipDatasetParams2() {
  return ZipDatasetParams(
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 15, 1)},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ZipDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})}};
}

ITERATOR_GET_NEXT_TEST_P(ZipDatasetOpTest, ZipDatasetParams, GetNextTestCases())

TEST_F(ZipDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ZipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ZipDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ZipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(ZipDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<ZipDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<ZipDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ZipDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<ZipDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<ZipDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ZipDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = ZipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ZipDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ZipDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
