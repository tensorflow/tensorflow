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
#ifndef TENSORFLOW_CONTRIB_LITE_TESTING_TEST_RUNNER_H_
#define TENSORFLOW_CONTRIB_LITE_TESTING_TEST_RUNNER_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tensorflow/contrib/lite/string.h"

namespace tflite {
namespace testing {

// This is the base class for processing test data. Each one of the virtual
// methods must be implemented to forward the data to the appropriate executor
// (e.g. TF Lite's interpreter, or the NNAPI).
class TestRunner {
 public:
  TestRunner() {}
  virtual ~TestRunner() {}

  // Load the given model, as a path relative to SetModelBaseDir().
  virtual void LoadModel(const string& bin_file_path) = 0;

  // Return the list of input tensors in the loaded model.
  virtual const std::vector<int>& GetInputs() = 0;

  // Return the list of output tensors in the loaded model.
  virtual const std::vector<int>& GetOutputs() = 0;

  // Prepare for a run by resize the given tensor. The given 'id' is
  // guaranteed to be one of the ids returned by GetInputs().
  virtual void ReshapeTensor(int id, const string& csv_values) = 0;

  // Reserve memory for all tensors.
  virtual void AllocateTensors() = 0;

  // Set the given tensor to some initial state, usually zero. This is
  // used to reset persistent buffers in a model.
  virtual void ResetTensor(int id) = 0;

  // Define the contents of the given input tensor. The given 'id' is
  // guaranteed to be one of the ids returned by GetInputs().
  virtual void SetInput(int id, const string& csv_values) = 0;

  // Define what should be expected for an output tensor after Invoke() runs.
  // The given 'id' is guaranteed to be one of the ids returned by
  // GetOutputs().
  virtual void SetExpectation(int id, const string& csv_values) = 0;

  // Run the model.
  virtual void Invoke() = 0;

  // Verify that the contents of all outputs conform to the existing
  // expectations. Return true if there are no expectations or they are all
  // satisfied.
  virtual bool CheckResults() = 0;

  // Read contents of tensor into csv format.
  // The given 'id' is guaranteed to be one of the ids returned by GetOutputs().
  virtual string ReadOutput(int id) = 0;

  // Set the base path for loading models.
  void SetModelBaseDir(const string& path) {
    model_base_dir_ = path;
    if (path[path.length() - 1] != '/') {
      model_base_dir_ += "/";
    }
  }

  // Return the full path of a model.
  string GetFullPath(const string& path) { return model_base_dir_ + path; }

  // Give an id to the next invocation to make error reporting more meaningful.
  void SetInvocationId(const string& id) { invocation_id_ = id; }
  const string& GetInvocationId() const { return invocation_id_; }

  // Invalidate the test runner, preventing it from executing any further.
  void Invalidate(const string& error_message) {
    std::cerr << error_message << std::endl;
    error_message_ = error_message;
  }
  bool IsValid() const { return error_message_.empty(); }
  const string& GetErrorMessage() const { return error_message_; }

  // Handle the overall success of this test runner. This will be true if all
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
#endif  // TENSORFLOW_CONTRIB_LITE_TESTING_TEST_RUNNER_H_
