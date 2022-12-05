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
#include "tensorflow/core/kernels/data/get_options_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/options_dataset_op.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptions[] = R"proto(
  deterministic: true
  slack: true
  optimization_options { apply_default_optimizations: true autotune: true }
  distribute_options {}
)proto";

class GetOptionsParams : public DatasetParams {
 public:
  template <typename T>
  GetOptionsParams(T input_dataset_params, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->emplace_back(OptionsDatasetOp::kInputDataset);
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    return OkStatus();
  }

  string dataset_type() const override { return "GetOptions"; }

  string op_name() const override { return dataset_type(); }

 private:
  string serialized_options_;
};

class GetOptionsOpTest : public DatasetOpsTestBase {};

OptionsDatasetParams OptionsDatasetParams0() {
  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

GetOptionsParams GetOptionsParams0() {
  return GetOptionsParams(OptionsDatasetParams0(),
                          /*output_dtypes=*/{DT_INT64},
                          /*output_shapes=*/{PartialTensorShape({})},
                          /*node_name=*/"get_options_0");
}

TEST_F(GetOptionsOpTest, Compute) {
  auto test_case_params = GetOptionsParams0();
  TF_ASSERT_OK(InitializeRuntime(test_case_params));
  std::vector<Tensor> output;
  TF_ASSERT_OK(RunDatasetOp(test_case_params, &output));
  EXPECT_EQ(1, output.size());
  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  Tensor expected_tensor =
      CreateTensor<tstring>(TensorShape({}), {options.SerializeAsString()});
  Tensor result_tensor = output[0];
  string serialized_options = result_tensor.scalar<tstring>()();
  Options result_options;
  result_options.ParseFromString(serialized_options);
  TF_EXPECT_OK(ExpectEqual(expected_tensor, result_tensor));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
