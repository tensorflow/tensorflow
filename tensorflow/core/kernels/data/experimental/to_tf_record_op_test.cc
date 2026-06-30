/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

class TestTensorDatasetParams : public DatasetParams {
 public:
  TestTensorDatasetParams(std::vector<Tensor> components, std::string node_name)
      : DatasetParams(TensorDtypes(components), TensorShapes(components),
                      std::move(node_name)),
        components_(std::move(components)) {}

  std::vector<Tensor> GetInputTensors() const override { return components_; }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    input_names->reserve(components_.size());
    for (size_t i = 0; i < components_.size(); ++i) {
      input_names->emplace_back(absl::StrCat("components_", i));
    }
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{"Toutput_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"metadata", ""}};
    return absl::OkStatus();
  }

  std::string dataset_type() const override { return "Tensor"; }

 private:
  DataTypeVector TensorDtypes(const std::vector<Tensor>& input_components) {
    DataTypeVector dtypes;
    for (const auto& component : input_components) {
      dtypes.emplace_back(component.dtype());
    }
    return dtypes;
  }

  std::vector<PartialTensorShape> TensorShapes(
      const std::vector<Tensor>& input_components) {
    std::vector<PartialTensorShape> shapes;
    for (const auto& component : input_components) {
      shapes.emplace_back(component.shape());
    }
    return shapes;
  }

  std::vector<Tensor> components_;
};

class ToTFRecordOpParams : public DatasetParams {
 public:
  ToTFRecordOpParams(std::shared_ptr<DatasetParams> input_dataset,
                     std::string filename, std::string compression_type,
                     std::string op_name, std::string node_name)
      : DatasetParams(/*output_dtypes=*/{}, /*output_shapes=*/{},
                      std::move(node_name)),
        filename_(filename),
        compression_type_(compression_type),
        op_name_(op_name) {
    input_dataset_params_.push_back(std::move(input_dataset));
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<tstring>(TensorShape({}), {filename_}),
            CreateTensor<tstring>(TensorShape({}), {compression_type_})};
  }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    *input_names = {"input_dataset", "filename", "compression_type"};
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {};
    return absl::OkStatus();
  }

  std::string dataset_type() const override { return "DatasetToTFRecord"; }

  std::string op_name() const override { return op_name_; }

 private:
  std::string filename_;
  std::string compression_type_;
  std::string op_name_;
};

class ToTFRecordOpTest : public DatasetOpsTestBase {};

TEST_F(ToTFRecordOpTest, DatasetToTFRecordSuccess) {
  std::vector<Tensor> components = {
      CreateTensor<tstring>(TensorShape({}), {"hello"})};
  auto tensor_dataset =
      std::make_shared<TestTensorDatasetParams>(components, "tensor_dataset");
  std::string filename = io::JoinPath(testing::TmpDir(), "file.tfrecord");
  ToTFRecordOpParams params(tensor_dataset, filename, "", "DatasetToTFRecord",
                            "to_tf_record_op");
  TF_ASSERT_OK(InitializeRuntime(params));
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(RunDatasetOp(params, &outputs));
}

TEST_F(ToTFRecordOpTest, DatasetToTFRecordNonScalarError) {
  std::vector<Tensor> components = {
      CreateTensor<tstring>(TensorShape({0}), {})};
  auto tensor_dataset =
      std::make_shared<TestTensorDatasetParams>(components, "tensor_dataset");
  std::string filename = io::JoinPath(testing::TmpDir(), "file_error.tfrecord");
  ToTFRecordOpParams params(tensor_dataset, filename, "", "DatasetToTFRecord",
                            "to_tf_record_op");
  TF_ASSERT_OK(InitializeRuntime(params));
  std::vector<Tensor> outputs;
  // Should fail since the tensor is not scalar.
  EXPECT_FALSE(RunDatasetOp(params, &outputs).ok());
}

TEST_F(ToTFRecordOpTest, ExperimentalDatasetToTFRecordSuccess) {
  std::vector<Tensor> components = {
      CreateTensor<tstring>(TensorShape({}), {"hello_experimental"})};
  auto tensor_dataset =
      std::make_shared<TestTensorDatasetParams>(components, "tensor_dataset");
  std::string filename =
      io::JoinPath(testing::TmpDir(), "file_experimental.tfrecord");
  ToTFRecordOpParams params(tensor_dataset, filename, "",
                            "ExperimentalDatasetToTFRecord", "to_tf_record_op");
  TF_ASSERT_OK(InitializeRuntime(params));
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(RunDatasetOp(params, &outputs));
}

TEST_F(ToTFRecordOpTest, ExperimentalDatasetToTFRecordNonScalarError) {
  std::vector<Tensor> components = {
      CreateTensor<tstring>(TensorShape({0}), {})};
  auto tensor_dataset =
      std::make_shared<TestTensorDatasetParams>(components, "tensor_dataset");
  std::string filename =
      io::JoinPath(testing::TmpDir(), "file_experimental_error.tfrecord");
  ToTFRecordOpParams params(tensor_dataset, filename, "",
                            "ExperimentalDatasetToTFRecord", "to_tf_record_op");
  TF_ASSERT_OK(InitializeRuntime(params));
  std::vector<Tensor> outputs;
  // Should fail since the tensor is not scalar.
  EXPECT_FALSE(RunDatasetOp(params, &outputs).ok());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
