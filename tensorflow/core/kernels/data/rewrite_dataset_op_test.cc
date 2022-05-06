/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/core/kernels/data/rewrite_dataset_op.h"

#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "rewrite_dataset";
constexpr char kReplicateOnSplit[] = "replicate_on_split";

class RewriteDatasetParams : public DatasetParams {
 public:
  template <typename T>
  RewriteDatasetParams(T input_dataset_params, string rewrite_name,
                       DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        rewrite_name_(rewrite_name) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<tstring>(TensorShape({}), {rewrite_name_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    *input_names = {RewriteDatasetOp::kInputDataset,
                    RewriteDatasetOp::kRewriteName};
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    return Status::OK();
  }

  string dataset_type() const override {
    return RewriteDatasetOp::kDatasetType;
  }

 private:
  string rewrite_name_;
};

class RewriteDatasetOpTest : public DatasetOpsTestBase {};

TEST_F(RewriteDatasetOpTest, ReplicateOnSplit) {
  auto range_dataset_params = RangeDatasetParams(0, 5, 1);
  auto rewrite_dataset_params =
      RewriteDatasetParams(std::move(range_dataset_params),
                           /*rewrite_name=*/kReplicateOnSplit,
                           /*output_dtypes=*/{DT_INT64},
                           /*output_shapes=*/{PartialTensorShape({})},
                           /*node_name=*/kNodeName);
  std::vector<Tensor> expected_outputs =
      CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}});

  TF_ASSERT_OK(Initialize(rewrite_dataset_params));
  TF_EXPECT_OK(CheckIteratorGetNext(expected_outputs, /*compare_order=*/true));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
