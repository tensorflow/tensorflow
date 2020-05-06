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

#if !defined(__APPLE__)
#include "tensorflow/lite/delegates/flex/delegate.h"
#endif
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
  enum class DelegateType {
    kNone,
    kNnapi,
    kGpu,
    kFlex,
  };

  /**
   * Creates a new TfLiteDriver
   * @param  delegate         The (optional) delegate to use.
   * @param  reference_kernel Whether to use the builtin reference kernel ops.
   */
  explicit TfLiteDriver(DelegateType delegate_type = DelegateType::kNone,
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
  void SetShapeExpectation(int id, const string& csv_values) override;
  void Invoke() override;
  bool CheckResults() override;
  string ReadOutput(int id) override;
  void SetThreshold(double relative_threshold, double absolute_threshold);
  void SetQuantizationErrorMultiplier(int quantization_error_multiplier);

 protected:
  Interpreter::TfLiteDelegatePtr delegate_;

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

  class DataExpectation;
  class ShapeExpectation;

  std::unique_ptr<OpResolver> resolver_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  std::map<int, std::unique_ptr<DataExpectation>> expected_output_;
  std::map<int, std::unique_ptr<ShapeExpectation>> expected_output_shape_;
  bool must_allocate_tensors_ = true;
  std::map<int, TfLiteTensor*> tensors_to_deallocate_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
