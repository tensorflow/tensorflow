/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TESTING_TF_DRIVER_H_
#define TENSORFLOW_LITE_TESTING_TF_DRIVER_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/test_runner.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"

namespace tflite {
namespace testing {

// A test runner that feeds inputs into Tensorflow and generates outputs.
class TfDriver : public TestRunner {
 public:
  explicit TfDriver(const std::vector<string>& input_layer,
                    const std::vector<string>& input_layer_type,
                    const std::vector<string>& input_layer_shape,
                    const std::vector<string>& output_layer);
  ~TfDriver() override {}

  void LoadModel(const string& bin_file_path) override;
  void SetInput(int id, const string& values_as_string) override;
  void Invoke() override;
  string ReadOutput(int id) override;

  const std::vector<int>& GetInputs() override { return input_ids_; }
  const std::vector<int>& GetOutputs() override { return output_ids_; }
  void ReshapeTensor(int id, const string& values_as_string) override;
  // Note: ResetTensor only works for input tensor.
  void ResetTensor(int id) override;

  // no-op. SetInput will overwrite existing data .
  void AllocateTensors() override {}
  // no-op. Tf driver is not supposed to check the results.
  void SetExpectation(int id, const string& values_as_string) override {}
  // no-op. Tf driver is not supposed to check the results.
  void SetShapeExpectation(int id, const string& values_as_string) override {}
  // tf driver is not supposed to check the results.
  bool CheckResults() override { return false; }

 protected:
  void SetInput(const string& values_as_string, tensorflow::Tensor*);
  string ReadOutput(const tensorflow::Tensor& tensor);

 private:
  std::unique_ptr<tensorflow::Session> session_;
  std::vector<int> input_ids_;
  std::vector<string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<tensorflow::DataType> input_types_;
  std::unordered_map<string, tensorflow::Tensor> input_tensors_;

  std::vector<int> output_ids_;
  std::vector<string> output_names_;
  std::vector<::tensorflow::Tensor> output_tensors_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TF_DRIVER_H_
