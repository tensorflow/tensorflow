/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/options_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptions[] = R"proto(
  deterministic: true
  slack: true
  optimization_options { apply_default_optimizations: true autotune: true }
)proto";

class OptionsDatasetOpTest : public DatasetOpsTestBase {};

OptionsDatasetParams OptionsDatasetParams0() {
  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams OptionsDatasetParams1() {
  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(10, 0, -3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_1");
}

OptionsDatasetParams OptionsDatasetParams2() {
  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 5, 1),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_2");
}

std::vector<GetNextTestCase<OptionsDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/OptionsDatasetParams0(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/OptionsDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})},
          {/*dataset_params=*/OptionsDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}})}};
}

ITERATOR_GET_NEXT_TEST_P(OptionsDatasetOpTest, OptionsDatasetParams,
                         GetNextTestCases())

TEST_F(OptionsDatasetOpTest, DatasetOptions) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  Options expected_options;
  protobuf::TextFormat::ParseFromString(kOptions, &expected_options);
  TF_ASSERT_OK(CheckDatasetOptions(expected_options));
}

TEST_F(OptionsDatasetOpTest, DatasetNodeName) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(OptionsDatasetOpTest, DatasetTypeString) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(OptionsDatasetOp::kDatasetType)));
}

TEST_F(OptionsDatasetOpTest, DatasetoutputDTypes) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(OptionsDatasetOpTest, DatasetoutputShapes) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(OptionsDatasetOpTest, DatasetCardinality) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(4));
}

TEST_F(OptionsDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(OptionsDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(OptionsDatasetOpTest, IteratorPrefix) {
  auto dataset_params = OptionsDatasetParams0();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      RangeDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
