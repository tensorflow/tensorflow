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
#ifndef TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
#define TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

// This is the base class for processing test data. Each one of the virtual
// methods must be implemented to forward the data to the appropriate executor
// (e.g. TF Lite's interpreter, or the NNAPI).
class TestRunner {
 public:
  TestRunner() {}
  virtual ~TestRunner() {}

  // Loads the given model, as a path relative to SetModelBaseDir().
  // DEPRECATED: use LoadModel with signature instead.
  virtual void LoadModel(const string& bin_file_path) = 0;
  // Loads the given model with signature specification.
  // Model path is relative to SetModelBaseDir().
  virtual void LoadModel(const string& bin_file_path,
                         const string& signature) = 0;

  // The following methods are supported by models with SignatureDef.
  //
  // Reshapes the tensors.
  // Keys are the input tensor names, values are csv string of the shape.
  virtual void ReshapeTensor(const string& name, const string& csv_values) = 0;

  // Sets the given tensor to some initial state, usually zero.
  virtual void ResetTensor(const std::string& name) = 0;

  // Reads the value of the output tensor and format it into a csv string.
  virtual string ReadOutput(const string& name) = 0;

  // Runs the model with signature.
  // Keys are the input tensor names, values are corresponding csv string.
  virtual void Invoke(const std::vector<std::pair<string, string>>& inputs) = 0;

  // Verifies that the contents of all outputs conform to the existing
  // expectations. Return true if there are no expectations or they are all
  // satisfied.
  // Keys are the input tensor names, values are corresponding csv string.
  virtual bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes) = 0;

  // Returns the list of output names in the loaded model for given signature.
  virtual std::vector<string> GetOutputNames() = 0;

  // Reserves memory for all tensors.
  virtual void AllocateTensors() = 0;

  // Sets the base path for loading models.
  void SetModelBaseDir(const string& path) {
    model_base_dir_ = path;
    if (path[path.length() - 1] != '/') {
      model_base_dir_ += "/";
    }
  }

  // Returns the full path of a model.
  string GetFullPath(const string& path) { return model_base_dir_ + path; }

  // Gives an id to the next invocation to make error reporting more meaningful.
  void SetInvocationId(const string& id) { invocation_id_ = id; }
  const string& GetInvocationId() const { return invocation_id_; }

  // Invalidates the test runner, preventing it from executing any further.
  void Invalidate(const string& error_message) {
    std::cerr << error_message << std::endl;
    error_message_ = error_message;
  }
  bool IsValid() const { return error_message_.empty(); }
  const string& GetErrorMessage() const { return error_message_; }

  // Handles the overall success of this test runner. This will be true if all
  // invocations were successful.
  void SetOverallSuccess(bool value) { overall_success_ = value; }
  bool GetOverallSuccess() const { return overall_success_; }

 protected:
  // A helper to check of the given number of values is consistent with the
  // number of bytes in a tensor of type T. When incompatibles sizes are found,
  // the test runner is invalidated and false is returned.
  template <typename T>
  bool CheckSizes(size_t tensor_bytes, size_t num_values) {
    size_t num_tensor_elements = tensor_bytes / sizeof(T);
    if (num_tensor_elements != num_values) {
      Invalidate("Expected '" + std::to_string(num_tensor_elements) +
                 "' elements for a tensor, but only got '" +
                 std::to_string(num_values) + "'");
      return false;
    }
    return true;
  }

 private:
  string model_base_dir_;
  string invocation_id_;
  bool overall_success_ = true;

  string error_message_;
};

}  // namespace testing
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
