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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_

#include <map>
#include <memory>

#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/testing/test_runner.h"

namespace tflite {
namespace testing {

// A test runner that feeds inputs into TF Lite and verifies its outputs.
class TfLiteDriver : public TestRunner {
 public:
  explicit TfLiteDriver(bool use_nnapi, const string& delegate = "",
                        bool reference_kernel = false);
  ~TfLiteDriver() override;

  void LoadModel(const string& bin_file_path) override;
  const std::vector<int>& GetInputs() override {
    return interpreter_->inputs();
  }
  const std::vector<int>& GetOutputs() override {
    return interpreter_->outputs();
  }
  void ReshapeTensor(int id, const string& csv_values) override;
  void AllocateTensors() override;
  void ResetTensor(int id) override;
  void SetInput(int id, const string& csv_values) override;
  void SetExpectation(int id, const string& csv_values) override;
  void Invoke() override;
  bool CheckResults() override;
  string ReadOutput(int id) override;

 private:
  void DeallocateStringTensor(TfLiteTensor* t) {
    if (t) {
      free(t->data.raw);
      t->data.raw = nullptr;
    }
  }
  void AllocateStringTensor(int id, size_t num_bytes, TfLiteTensor* t) {
    t->data.raw = reinterpret_cast<char*>(malloc(num_bytes));
    t->bytes = num_bytes;
    tensors_to_deallocate_[id] = t;
  }

  void ResetLSTMStateTensors();

  class Expectation;

  std::unique_ptr<OpResolver> resolver_;
  std::unique_ptr<FlexDelegate> delegate_;
  bool use_nnapi_ = false;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  std::map<int, std::unique_ptr<Expectation>> expected_output_;
  bool must_allocate_tensors_ = true;
  std::map<int, TfLiteTensor*> tensors_to_deallocate_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
