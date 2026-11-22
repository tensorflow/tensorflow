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
#include "tensorflow/core/kernels/data/flat_map_dataset_op.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/platform/test.h"
// IWYU pragma: begin_keep
#include "tensorflow/core/platform/test_benchmark.h"
// IWYU pragma: end_keep

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "flat_map_dataset";

class FlatMapDatasetParams : public DatasetParams {
 public:
  template <typename T>
  FlatMapDatasetParams(T input_dataset_params,
                       std::vector<Tensor> other_arguments,
                       FunctionDefHelper::AttrValueWrapper func,
                       std::vector<FunctionDef> func_lib,
                       DataTypeVector type_arguments,
                       DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       std::string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return other_arguments_;
  }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    input_names->emplace_back(FlatMapDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(FlatMapDatasetOp::kOtherArguments, "_", i));
    }
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{"f", func_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"metadata", ""}};
    return absl::OkStatus();
  }

  std::string dataset_type() const override {
    return FlatMapDatasetOp::kDatasetType;
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class FlatMapDatasetOpTest : public DatasetOpsTestBase {
 protected:
  void TestBody() override {}
};

// Test case 1: normal case.
FlatMapDatasetParams FlatMapDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  auto func = FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{"Toutput_types", DataTypeVector({DT_INT64})},
                 {"output_shapes",
                  std::vector<PartialTensorShape>({PartialTensorShape({1})})}});
  return FlatMapDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*func=*/func,
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 2: test the case if the function does not return a single scalar
// of dtype DT_VARIANT.
FlatMapDatasetParams InvalidFlatMapDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  auto func = FunctionDefHelper::FunctionRef(/*name*/ "NonZero",
                                             /*attrs*/ {{"T", DT_INT64}});
  return FlatMapDatasetParams(std::move(tensor_slice_dataset_params),
                              /*other_arguments=*/{},
                              /*func=*/func,
                              /*func_lib=*/{test::function::NonZero()},
                              /*type_arguments=*/{},
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({1})},
                              /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<FlatMapDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/FlatMapDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})}};
}

ITERATOR_GET_NEXT_TEST_P(FlatMapDatasetOpTest, FlatMapDatasetParams,
                         GetNextTestCases())

std::vector<SkipTestCase<FlatMapDatasetParams>> SkipTestCases() {
  return {{/*dataset_params=*/FlatMapDatasetParams1(),
           /*num_to_skip*/ 2, /*expected_num_skipped*/ 2, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{2}})},
          {/*dataset_params=*/FlatMapDatasetParams1(),
           /*num_to_skip*/ 4, /*expected_num_skipped*/ 4, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{4}})},
          {/*dataset_params=*/FlatMapDatasetParams1(),
           /*num_to_skip*/ 9, /*expected_num_skipped*/ 9, /*get_next*/ false},
          {/*dataset_params=*/FlatMapDatasetParams1(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9, /*get_next*/ false}};
}

ITERATOR_SKIP_TEST_P(FlatMapDatasetOpTest, FlatMapDatasetParams,
                     SkipTestCases())

TEST_F(FlatMapDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(FlatMapDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(FlatMapDatasetOp::kDatasetType)));
}

TEST_F(FlatMapDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(FlatMapDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

TEST_F(FlatMapDatasetOpTest, Cardinality) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(FlatMapDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(FlatMapDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(FlatMapDatasetOpTest, IteratorPrefix) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      FlatMapDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<FlatMapDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/FlatMapDatasetParams1(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(FlatMapDatasetOpTest, FlatMapDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(FlatMapDatasetOpTest, InvalidMapFunc) {
  auto dataset_params = InvalidFlatMapDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST_F(FlatMapDatasetOpTest, SplitProvider) {
  auto params = FlatMapDatasetParams1();
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params,
      CreateTensors<int64_t>(TensorShape({1}),
                             {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/1,
      CreateTensors<int64_t>(TensorShape({1}), {{3}, {4}, {5}})));
}

TEST_F(FlatMapDatasetOpTest, SymbolicCheckpointSaveAndRestore) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));

  IteratorContext::Params params(iterator_ctx_.get());
  params.symbolic_checkpoint = true;
  iterator_ctx_ = std::make_unique<IteratorContext>(std::move(params));

  TF_ASSERT_OK(dataset_->MakeIterator(iterator_ctx_.get(), /*parent=*/nullptr,
                                      dataset_params.iterator_prefix(),
                                      &iterator_));

  const std::vector<Tensor> expected_outputs = CreateTensors<int64_t>(
      TensorShape({1}), {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
  const std::vector<int> breakpoints = {0, 4, 7, 11};
  bool end_of_sequence = false;
  int cur_iteration = 0;
  std::vector<Tensor> out_tensors;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_ASSERT_OK(iterator_ctx_->checkpoint()->Save(&writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_ASSERT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 dataset_params.iterator_prefix(), *dataset_,
                                 &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_ASSERT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }
  TF_EXPECT_OK(
      ExpectEqual(out_tensors, expected_outputs, /*compare_order=*/true));
}

TEST_F(FlatMapDatasetOpTest, SymbolicCheckpointSkip) {
  auto dataset_params = FlatMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));

  IteratorContext::Params params(iterator_ctx_.get());
  params.symbolic_checkpoint = true;
  iterator_ctx_ = std::make_unique<IteratorContext>(std::move(params));
  TF_ASSERT_OK(dataset_->MakeIterator(iterator_ctx_.get(), /*parent=*/nullptr,
                                      dataset_params.iterator_prefix(),
                                      &iterator_));
  TF_ASSERT_OK(CheckIteratorSkip(
      /*num_to_skip=*/4, /*expected_num_skipped=*/4, /*get_next=*/true,
      /*expected_outputs=*/CreateTensors<int64_t>(TensorShape({1}), {{4}}),
      /*compare_order=*/true));
}

FlatMapDatasetParams FlatMapBenchmarkDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{100, 10, 1})},
      /*node_name=*/"tensor_slice");
  auto func = FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{"Toutput_types", DataTypeVector({DT_INT64})},
                 {"output_shapes",
                  std::vector<PartialTensorShape>({PartialTensorShape({1})})}});
  return FlatMapDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*func=*/func,
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// `test_benchmark.h` provides `BENCHMARK`, `benchmark::State`, and
// `benchmark::DoNotOptimize` without re-exporting the internal headers that
// define them (`registration.h`, `state.h`, `utils.h`), which do not exist in
// the benchmark release that open-source TensorFlow builds with.
// NOLINTBEGIN(misc-include-cleaner)
void BM_FlatMapDatasetGetNext(benchmark::State& state) {
  auto dataset_params = FlatMapBenchmarkDatasetParams();
  FlatMapDatasetOpTest test_base;
  TF_CHECK_OK(test_base.InitializeRuntime(dataset_params));
  std::unique_ptr<TestDataset> dataset;
  TF_CHECK_OK(test_base.MakeDataset(dataset_params, &dataset));
  std::unique_ptr<TestIterator> iterator;
  std::vector<Tensor> out_tensors;
  for (auto _ : state) {
    TF_CHECK_OK(test_base.MakeIterator(dataset_params, *dataset, &iterator));
    bool end_of_sequence = false;
    while (!end_of_sequence) {
      TF_CHECK_OK(iterator->GetNext(&out_tensors, &end_of_sequence));
      benchmark::DoNotOptimize(out_tensors);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_FlatMapDatasetGetNext);

void BM_FlatMapDatasetSkip(benchmark::State& state) {
  auto dataset_params = FlatMapBenchmarkDatasetParams();
  FlatMapDatasetOpTest test_base;
  TF_CHECK_OK(test_base.InitializeRuntime(dataset_params));
  std::unique_ptr<TestDataset> dataset;
  TF_CHECK_OK(test_base.MakeDataset(dataset_params, &dataset));
  std::unique_ptr<TestIterator> iterator;
  for (auto _ : state) {
    TF_CHECK_OK(test_base.MakeIterator(dataset_params, *dataset, &iterator));
    bool end_of_sequence = false;
    while (!end_of_sequence) {
      int num_skipped = 0;
      TF_CHECK_OK(iterator->iterator()->Skip(iterator->ctx(), /*num_to_skip=*/1,
                                             &end_of_sequence, &num_skipped));
      benchmark::DoNotOptimize(num_skipped);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_FlatMapDatasetSkip);
// NOLINTEND(misc-include-cleaner)

}  // namespace
}  // namespace data
}  // namespace tensorflow
