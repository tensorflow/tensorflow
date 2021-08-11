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
#include "tensorflow/core/kernels/data/finalize_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/options_dataset_op.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

class FinalizeDatasetParams : public DatasetParams {
 public:
  template <typename T>
  FinalizeDatasetParams(T input_dataset_params, DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        has_captured_ref_(false) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->emplace_back(FinalizeDatasetOp::kInputDataset);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{FinalizeDatasetOp::kHasCapturedRef, has_captured_ref_},
                    {FinalizeDatasetOp::kOutputTypes, output_dtypes_},
                    {FinalizeDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  string dataset_type() const override { return "Finalize"; }

 private:
  bool has_captured_ref_;
};

class FinalizeDatasetOpTest : public DatasetOpsTestBase {
 public:
  void CheckDatasetPipelineTypeStrings(
      const std::vector<std::string>& type_strings) {
    CheckDatasetPipelineTypeString(dataset_, type_strings, 0);
  }

  void CheckDatasetPipelineTypeString(
      const DatasetBase* dataset, const std::vector<std::string>& type_strings,
      int index) {
    EXPECT_GT(type_strings.size(), index);
    EXPECT_EQ(dataset->type_string(), type_strings[index]);
    std::vector<const DatasetBase*> input_datasets;
    TF_ASSERT_OK(dataset->InputDatasets(&input_datasets));
    if (input_datasets.empty()) {
      return;
    }
    EXPECT_EQ(1, input_datasets.size());
    CheckDatasetPipelineTypeString(input_datasets[0], type_strings, index + 1);
  }
};

constexpr char kNoOptimizationOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: false }
)pb";
constexpr char kMaxIntraOpParallelismOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: false }
  threading_options { max_intra_op_parallelism: 10 }
)pb";
constexpr char kPrivateThreadPoolOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: false }
  threading_options { private_threadpool_size: 10 }
)pb";
constexpr char kModelOptions[] = R"proto(
  optimization_options { apply_default_optimizations: false }
)proto";
constexpr char kOptimizationsDefaultOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: true }
)pb";
constexpr char kAllChainedDatasetsOptions[] = R"pb(
  autotune_options { enabled: true }
  optimization_options { apply_default_optimizations: true }
  threading_options { max_intra_op_parallelism: 10 private_threadpool_size: 10 }
)pb";

OptionsDatasetParams NoOptimizationOptionsParams() {
  Options options;
  protobuf::TextFormat::ParseFromString(kNoOptimizationOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams MaxIntraOpParallelismOptionsParams() {
  Options options;
  protobuf::TextFormat::ParseFromString(kMaxIntraOpParallelismOptions,
                                        &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams PrivateThreadPoolOptionsParams() {
  Options options;
  protobuf::TextFormat::ParseFromString(kPrivateThreadPoolOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams ModelOptionsParams() {
  Options options;
  protobuf::TextFormat::ParseFromString(kModelOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams OptimizationsDefaultOptionsParams() {
  Options options;
  protobuf::TextFormat::ParseFromString(kOptimizationsDefaultOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams AllChainedDatasetsOptionsParams() {
  Options options;
  protobuf::TextFormat::ParseFromString(kAllChainedDatasetsOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

FinalizeDatasetParams NoOptimizationFinalizeParams() {
  return FinalizeDatasetParams(NoOptimizationOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"options_dataset_0");
}

FinalizeDatasetParams MaxIntraOpParallelismParams() {
  return FinalizeDatasetParams(MaxIntraOpParallelismOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"MaxIntraOpParallelismDatasetOp");
}

FinalizeDatasetParams PrivateThreadPoolParams() {
  return FinalizeDatasetParams(PrivateThreadPoolOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"PrivateThreadPoolDatasetOp");
}

FinalizeDatasetParams ModelParams() {
  return FinalizeDatasetParams(ModelOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"ModelDatasetOp");
}

FinalizeDatasetParams OptimizationsDefaultParams() {
  return FinalizeDatasetParams(OptimizationsDefaultOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"private_thread_pool");
}

FinalizeDatasetParams AllChainedDatasetsParams() {
  return FinalizeDatasetParams(AllChainedDatasetsOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"ModelDataset/_9");
}

TEST_F(FinalizeDatasetOpTest, NoOptimizationNodeName) {
  auto test_case_params = NoOptimizationFinalizeParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings({"OptionsDataset", "RangeDataset"});
}

std::vector<GetNextTestCase<FinalizeDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/NoOptimizationFinalizeParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/MaxIntraOpParallelismParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/PrivateThreadPoolParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/ModelParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/OptimizationsDefaultParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/AllChainedDatasetsParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})}};
}

ITERATOR_GET_NEXT_TEST_P(FinalizeDatasetOpTest, FinalizeDatasetParams,
                         GetNextTestCases())

TEST_F(FinalizeDatasetOpTest, MaxIntraOpParallelismNodeName) {
  auto test_case_params = MaxIntraOpParallelismParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings(
      {"MaxIntraOpParallelismDataset", "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, PrivateThreadPoolNodeName) {
  auto test_case_params = PrivateThreadPoolParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings(
      {"PrivateThreadPoolDataset", "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, ModelNodeName) {
  auto test_case_params = ModelParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings(
      {"ModelDataset", "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, OptimizationsDefaultNodeName) {
  auto test_case_params = OptimizationsDefaultParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings({"PrivateThreadPoolDataset",
                                   "MaxIntraOpParallelismDataset",
                                   "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, AllChainedDatasetsNodeName) {
  auto test_case_params = AllChainedDatasetsParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings({"ModelDataset", "PrivateThreadPoolDataset",
                                   "MaxIntraOpParallelismDataset",
                                   "OptionsDataset", "RangeDataset"});
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
