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
#include "tensorflow/core/kernels/data/range_dataset_op.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

class RangeDatasetOpTest : public DatasetOpsTestBase {};

RangeDatasetParams PositiveStepRangeDatasetParams() {
  return RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3);
}

RangeDatasetParams NegativeStepRangeDatasetParams() {
  return RangeDatasetParams(/*start=*/10, /*stop=*/0, /*step=*/-3);
}

RangeDatasetParams Int32OverflowRangeDatasetParames() {
  return RangeDatasetParams(/*start=*/2'147'483'647LL, /*stop=*/2'147'483'649LL,
                            /*step=*/1);
}

RangeDatasetParams UnsignedInt32OverflowRangeDatasetParames() {
  return RangeDatasetParams(/*start=*/4'294'967'295LL, /*stop=*/4'294'967'297LL,
                            /*step=*/1);
}

RangeDatasetParams ZeroStepRangeDatasetParams() {
  return RangeDatasetParams(/*start=*/10, /*stop=*/0, /*step=*/0);
}

RangeDatasetParams RangeDatasetParams1() {
  return RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3,
                            /*output_dtypes=*/{DT_INT32});
}

RangeDatasetParams RangeDatasetParams2() {
  return RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3,
                            /*output_dtypes=*/{DT_INT64});
}

std::vector<GetNextTestCase<RangeDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})},
          {/*dataset_params=*/Int32OverflowRangeDatasetParames(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}),
                                  {{2'147'483'647LL}, {2'147'483'648LL}})},
          {/*dataset_params=*/UnsignedInt32OverflowRangeDatasetParames(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}),
                                  {{4'294'967'295LL}, {4'294'967'296LL}})}};
}

ITERATOR_GET_NEXT_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                         GetNextTestCases())

TEST_F(RangeDatasetOpTest, DatasetNodeName) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(range_dataset_params.node_name()));
}

TEST_F(RangeDatasetOpTest, DatasetTypeString) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(RangeDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<RangeDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/RangeDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT32}},
          {/*dataset_params=*/RangeDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                             DatasetOutputDtypesTestCases())

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<RangeDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*expected_cardinality=*/4}};
}

DATASET_CARDINALITY_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<RangeDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/RangeDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT32}},
          {/*dataset_params=*/RangeDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                              IteratorOutputDtypesTestCases())

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(RangeDatasetOpTest, IteratorPrefix) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      RangeDatasetOp::kDatasetType, range_dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<RangeDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(RangeDatasetOpTest, ZeroStep) {
  auto range_dataset_params = ZeroStepRangeDatasetParams();
  EXPECT_EQ(Initialize(range_dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(RangeDatasetOpTest, SplitProviderPositiveStep) {
  auto params = RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3,
                                   /*output_dtypes=*/{DT_INT64});
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/2, /*shard_index=*/1,
      CreateTensors<int64_t>(TensorShape({}), {{3}, {9}})));
}

TEST_F(RangeDatasetOpTest, SplitProviderNegativeStep) {
  auto params = RangeDatasetParams(/*start=*/10, /*stop=*/0, /*step=*/-3,
                                   /*output_dtypes=*/{DT_INT64});
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/2, /*shard_index=*/0,
      CreateTensors<int64_t>(TensorShape({}), {{10}, {4}})));
}

TEST_F(RangeDatasetOpTest, SplitProviderEmpty) {
  auto params = RangeDatasetParams(/*start=*/0, /*stop=*/0, /*step=*/1,
                                   /*output_dtypes=*/{DT_INT64});
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}), {})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/2,
      CreateTensors<int64_t>(TensorShape({}), {})));
}

template <typename Tag>
struct AccessResult {
  static typename Tag::type ptr;
};
template <typename Tag>
typename Tag::type AccessResult<Tag>::ptr;

template <typename Tag, typename Tag::type p>
struct AccessRob {
  struct Filler {
    Filler() { AccessResult<Tag>::ptr = p; }
  };
  static Filler filler;
};
template <typename Tag, typename Tag::type p>
typename AccessRob<Tag, p>::Filler AccessRob<Tag, p>::filler;

struct MakeIteratorInternalTag {
  typedef std::unique_ptr<IteratorBase> (DatasetBase::*type)(
      const std::string&) const;
};
template class AccessRob<MakeIteratorInternalTag,
                         &DatasetBase::MakeIteratorInternal>;

TEST_F(RangeDatasetOpTest, UninitializedIteratorRestore) {
  auto dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));

  std::vector<const DatasetBase*> inputs;
  const DatasetBase* target_dataset = dataset_;
  while (target_dataset->type_string() != "RangeDataset") {
    inputs.clear();
    TF_ASSERT_OK(target_dataset->InputDatasets(&inputs));
    ASSERT_FALSE(inputs.empty());
    target_dataset = inputs[0];
  }

  auto method = AccessResult<MakeIteratorInternalTag>::ptr;
  auto iterator = (target_dataset->*method)("test_prefix");

  IteratorContext iter_ctx(IteratorContext::Params(iterator_ctx_.get()));
  std::vector<const VariantTensorData*> data;
  VariantTensorDataReader reader(data);
  absl::Status s = iterator->Restore(&iter_ctx, &reader);

  EXPECT_EQ(s.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(s.message(),
            "`Initialize` should be called before saving/restoring from "
            "tf.data checkpoints.");
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
