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
#include "tensorflow/core/kernels/data/optimize_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "optimize_dataset";
constexpr char kNoopElimination[] = "noop_elimination";

class OptimizeDatasetParams : public DatasetParams {
 public:
  template <typename T>
  OptimizeDatasetParams(T input_dataset_params, string optimizations,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        std::vector<tstring> optimization_configs,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        optimizations_(std::move(optimizations)),
        optimization_configs_(std::move(optimization_configs)) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<tstring>(TensorShape({1}), {optimizations_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    *input_names = {OptimizeDatasetOp::kInputDataset,
                    OptimizeDatasetOp::kOptimizations};
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {
        {OptimizeDatasetOp::kOutputShapes, output_shapes_},
        {OptimizeDatasetOp::kOutputTypes, output_dtypes_},
        {OptimizeDatasetOp::kOptimizationConfigs, optimization_configs_}};
    return Status::OK();
  }

  string dataset_type() const override {
    return OptimizeDatasetOp::kDatasetType;
  }

 private:
  string optimizations_;
  std::vector<tstring> optimization_configs_;
};

class OptimizeDatasetOpTest : public DatasetOpsTestBase {};

TEST_F(OptimizeDatasetOpTest, NoopElimination) {
  auto take_dataset_parmas =
      TakeDatasetParams(RangeDatasetParams(-3, 3, 1),
                        /*count=*/-3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  auto optimize_dataset_params =
      OptimizeDatasetParams(std::move(take_dataset_parmas),
                            /*optimizations=*/{kNoopElimination},
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*optimization_configs=*/{},
                            /*node_name=*/kNodeName);
  std::vector<Tensor> expected_outputs = CreateTensors<int64_t>(
      TensorShape({}), {{-3}, {-2}, {-1}, {0}, {1}, {2}});

  TF_ASSERT_OK(Initialize(optimize_dataset_params));
  TF_EXPECT_OK(CheckIteratorGetNext(expected_outputs, /*compare_order=*/true));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
