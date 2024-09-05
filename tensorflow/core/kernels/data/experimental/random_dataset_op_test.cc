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
#include "tensorflow/core/kernels/data/experimental/random_dataset_op.h"

#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/random_seed_ops.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "random_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

// Number of random samples generated per test
constexpr int kCount = 10;

void GenerateExpectedEpochData(int64_t seed, int64_t seed2, int count,
                               std::vector<Tensor>* epoch_data) {
  auto parent_generator = random::PhiloxRandom(seed, seed2);
  auto generator =
      random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator);
  for (int i = 0; i < count; ++i) {
    epoch_data->push_back(
        CreateTensor<int64_t>(TensorShape({}), {generator()}));
  }
}

// Generate the first `count` random numbers that the kernel should produce
// for a given seed/seed2 combo.
// For compatibility with the test harness, return value is a vector of scalar
// Tensors.
std::vector<Tensor> GenerateExpectedData(int64_t seed, int64_t seed2, int count,
                                         bool rerandomize_each_iteration,
                                         int iterations) {
  RandomSeedGenerator parent_seed_generator(RandomSeeds(seed, seed2));
  std::vector<Tensor> ret;
  for (int j = 0; j < iterations; ++j) {
    if (rerandomize_each_iteration) {
      parent_seed_generator.GenerateSeeds(&seed, &seed2);
    }
    GenerateExpectedEpochData(seed, seed2, count, &ret);
  }
  return ret;
}

std::vector<Tensor> GenerateExpectedSaveAndRestoreData(
    int64_t seed, int64_t seed2, int count, bool rerandomize_each_iteration) {
  RandomSeedGenerator parent_seed_generator(RandomSeeds(seed, seed2));
  if (rerandomize_each_iteration) {
    // The save and restore test harness `ITERATOR_SAVE_AND_RESTORE_TEST_P`
    // calls `MakeIterator` twice. There are therefore 2 calls here to generate
    // seeds where the first set of seeds are skipped in order to generate
    // the expected random numbers generated.
    parent_seed_generator.GenerateSeeds(&seed, &seed2);
    parent_seed_generator.GenerateSeeds(&seed, &seed2);
  }
  std::vector<Tensor> ret;
  GenerateExpectedEpochData(seed, seed2, count, &ret);
  return ret;
}

class RandomDatasetParams : public DatasetParams {
 public:
  RandomDatasetParams(int64_t seed, int64_t seed2, int32_t op_version,
                      bool rerandomize_each_iteration,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        seed_(CreateTensor<int64_t>(TensorShape({}), {seed})),
        seed2_(CreateTensor<int64_t>(TensorShape({}), {seed2})),
        dummy_resource_handle_(CreateDummyResourceHandle()),
        seed_generator_resource_(CreateTensor<ResourceHandle>(
            TensorShape({}), {dummy_resource_handle_})),
        rerandomize_each_iteration_(rerandomize_each_iteration) {
    op_version_ = op_version;
  }

  ResourceHandle CreateDummyResourceHandle() { return ResourceHandle(); }

  virtual std::vector<Tensor> GetInputTensors() const override {
    return {seed_, seed2_, seed_generator_resource_};
  }

  virtual Status GetInputNames(
      std::vector<string>* input_names) const override {
    *input_names = {RandomDatasetOp::kSeed, RandomDatasetOp::kSeed2};
    if (op_version_ == 2) {
      input_names->emplace_back("seed_generator");
    }
    return absl::OkStatus();
  }

  virtual Status GetAttributes(AttributeVector* attributes) const override {
    *attributes = {{"output_types", output_dtypes_},
                   {"output_shapes", output_shapes_},
                   {"metadata", ""}};
    if (op_version_ == 2) {
      attributes->emplace_back("rerandomize_each_iteration",
                               rerandomize_each_iteration_);
    }
    return absl::OkStatus();
  }

  virtual string dataset_type() const override {
    return RandomDatasetOp::kDatasetType;
  }

 private:
  Tensor seed_;
  Tensor seed2_;
  ResourceHandle dummy_resource_handle_;
  Tensor seed_generator_resource_;
  bool rerandomize_each_iteration_;
};

class RandomDatasetOpTest : public DatasetOpsTestBase {};

RandomDatasetParams FortyTwo() {
  return {/*seed=*/42,
          /*seed2=*/42,
          /*op_version=*/1,
          /*rerandomize_each_iteration=*/false,
          /*output_dtypes=*/
          {DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just first seed relative to FortyTwo
RandomDatasetParams ChangeSeed() {
  return {/*seed=*/1000,
          /*seed2=*/42,
          /*op_version=*/1,
          /*rerandomize_each_iteration=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just second seed relative to FortyTwo
RandomDatasetParams ChangeSeed2() {
  return {/*seed=*/42,
          /*seed2=*/1000,
          /*op_version=*/1,
          /*rerandomize_each_iteration=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

RandomDatasetParams FortyTwoV2RerandomizeEachIterationFalse() {
  return {/*seed=*/42,
          /*seed2=*/42,
          /*op_version=*/2,
          /*rerandomize_each_iteration=*/false,
          /*output_dtypes=*/
          {DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just first seed relative to FortyTwo
RandomDatasetParams ChangeSeedV2RerandomizeEachIterationFalse() {
  return {/*seed=*/1000,
          /*seed2=*/42,
          /*op_version=*/2,
          /*rerandomize_each_iteration=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just second seed relative to FortyTwo
RandomDatasetParams ChangeSeed2V2RerandomizeEachIterationFalse() {
  return {/*seed=*/42,
          /*seed2=*/1000,
          /*op_version=*/2,
          /*rerandomize_each_iteration=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

RandomDatasetParams FortyTwoV2RerandomizeEachIterationTrue() {
  return {/*seed=*/42,
          /*seed2=*/42,
          /*op_version=*/2,
          /*rerandomize_each_iteration=*/true,
          /*output_dtypes=*/
          {DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just first seed relative to FortyTwo
RandomDatasetParams ChangeSeedV2RerandomizeEachIterationTrue() {
  return {/*seed=*/1000,
          /*seed2=*/42,
          /*op_version=*/2,
          /*rerandomize_each_iteration=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just second seed relative to FortyTwo
RandomDatasetParams ChangeSeed2V2RerandomizeEachIterationTrue() {
  return {/*seed=*/42,
          /*seed2=*/1000,
          /*op_version=*/2,
          /*rerandomize_each_iteration=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

class ParameterizedGetNextTest : public RandomDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<RandomDatasetParams>> {};

std::vector<GetNextTestCase<RandomDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_outputs=*/GenerateExpectedData(
               42, 42, kCount,
               /*rerandomize_each_iteration=*/false,
               /*iterations=*/2)},
          {/*dataset_params=*/ChangeSeed(),
           /*expected_outputs=*/GenerateExpectedData(
               1000, 42, kCount,
               /*rerandomize_each_iteration=*/false,
               /*iterations=*/2)},
          {/*dataset_params=*/ChangeSeed2(),
           /*expected_outputs=*/GenerateExpectedData(
               42, 1000, kCount,
               /*rerandomize_each_iteration=*/false,
               /*iterations=*/2)},
          {/*dataset_params=*/FortyTwoV2RerandomizeEachIterationFalse(),
           /*expected_outputs=*/GenerateExpectedData(
               42, 42, kCount,
               /*rerandomize_each_iteration=*/false,
               /*iterations=*/2)},
          {/*dataset_params=*/ChangeSeedV2RerandomizeEachIterationFalse(),
           /*expected_outputs=*/GenerateExpectedData(
               1000, 42, kCount,
               /*rerandomize_each_iteration=*/false,
               /*iterations=*/2)},
          {/*dataset_params=*/ChangeSeed2V2RerandomizeEachIterationFalse(),
           /*expected_outputs=*/GenerateExpectedData(
               42, 1000, kCount,
               /*rerandomize_each_iteration=*/false,
               /*iterations=*/2)},
          {/*dataset_params=*/FortyTwoV2RerandomizeEachIterationTrue(),
           /*expected_outputs=*/GenerateExpectedData(
               42, 42, kCount,
               /*rerandomize_each_iteration=*/true, /*iterations=*/2)},
          {/*dataset_params=*/ChangeSeedV2RerandomizeEachIterationTrue(),
           /*expected_outputs=*/GenerateExpectedData(
               1000, 42, kCount,
               /*rerandomize_each_iteration=*/true, /*iterations=*/2)},
          {/*dataset_params=*/ChangeSeed2V2RerandomizeEachIterationTrue(),
           /*expected_outputs=*/GenerateExpectedData(
               42, 1000, kCount,
               /*rerandomize_each_iteration=*/true, /*iterations=*/2)}};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  // Can't use DatasetOpsTestBase::CheckIteratorGetNext because the kernel
  // under test produces unbounded input.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (out_tensors.size() < kCount) {
    std::vector<Tensor> next;
    TF_ASSERT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));

    ASSERT_FALSE(end_of_sequence);  // Dataset should never stop

    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_ASSERT_OK(dataset_->MakeIterator(
      iterator_ctx_.get(), /*parent=*/nullptr,
      test_case.dataset_params.iterator_prefix(), &iterator_));
  while (out_tensors.size() < 2 * kCount) {
    std::vector<Tensor> next;
    TF_ASSERT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));

    ASSERT_FALSE(end_of_sequence);  // Dataset should never stop

    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_ASSERT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    RandomDatasetOpTest, ParameterizedGetNextTest,
    ::testing::ValuesIn(
        std::vector<GetNextTestCase<RandomDatasetParams>>(GetNextTestCases())));

std::vector<DatasetNodeNameTestCase<RandomDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/FortyTwo(), /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                         DatasetNodeNameTestCases());

std::vector<DatasetTypeStringTestCase<RandomDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               RandomDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                           DatasetTypeStringTestCases());

std::vector<DatasetOutputDtypesTestCase<RandomDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                             DatasetOutputDtypesTestCases());

std::vector<DatasetOutputShapesTestCase<RandomDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                             DatasetOutputShapesTestCases());

std::vector<CardinalityTestCase<RandomDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_cardinality=*/kInfiniteCardinality}};
}

DATASET_CARDINALITY_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                           CardinalityTestCases());

std::vector<IteratorOutputDtypesTestCase<RandomDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                              IteratorOutputDtypesTestCases());

std::vector<IteratorOutputShapesTestCase<RandomDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                              IteratorOutputShapesTestCases());

std::vector<IteratorPrefixTestCase<RandomDatasetParams>>
IteratorOutputPrefixTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               RandomDatasetOp::kDatasetType, kIteratorPrefix)}};
}

ITERATOR_PREFIX_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                       IteratorOutputPrefixTestCases());

std::vector<IteratorSaveAndRestoreTestCase<RandomDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/FortyTwo(), /*breakpoints=*/{2, 5, 8},
           /*expected_outputs=*/
           GenerateExpectedSaveAndRestoreData(
               42, 42, 9 /* 8 + 1 */,
               /*rerandomize_each_iteration=*/false)},
          {/*dataset_params=*/FortyTwoV2RerandomizeEachIterationFalse(),
           /*breakpoints=*/{2, 5, 8},
           /*expected_outputs=*/
           GenerateExpectedSaveAndRestoreData(
               42, 42, 9 /* 8 + 1 */,
               /*rerandomize_each_iteration=*/false)},
          {/*dataset_params=*/FortyTwoV2RerandomizeEachIterationTrue(),
           /*breakpoints=*/{2, 5, 8},
           /*expected_outputs=*/
           GenerateExpectedSaveAndRestoreData(
               42, 42, 9 /* 8 + 1 */,
               /*rerandomize_each_iteration=*/true)}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                                 IteratorSaveAndRestoreTestCases());

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
