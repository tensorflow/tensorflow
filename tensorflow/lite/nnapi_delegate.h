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
#ifndef TENSORFLOW_LITE_NNAPI_DELEGATE_H_
#define TENSORFLOW_LITE_NNAPI_DELEGATE_H_

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

class ANeuralNetworksModel;
class ANeuralNetworksMemory;
class ANeuralNetworksCompilation;

namespace tflite {

class NNAPIAllocation : public MMAPAllocation {
 public:
  NNAPIAllocation(const char* filename, ErrorReporter* error_reporter);
  ~NNAPIAllocation();

  size_t offset(const void* ptr) const {
    auto signed_offset = reinterpret_cast<const uint8_t*>(ptr) -
                         reinterpret_cast<const uint8_t*>(mmapped_buffer_);

    return static_cast<size_t>(signed_offset);
  }

  ANeuralNetworksMemory* memory() const { return handle_; }
  bool valid() const override { return handle_ != nullptr; }

 private:
  mutable ANeuralNetworksMemory* handle_ = nullptr;
};

class NNAPIDelegate {
 public:
  ~NNAPIDelegate();

  // Convert a tflite graph to NNAPI
  TfLiteStatus BuildGraph(Subgraph* subgraph);

  // Run
  TfLiteStatus Invoke(Subgraph* subgraph);

  // Whether the current platform supports NNAPI delegation.
  static bool IsSupported();

 private:
  // The NN API model handle
  ANeuralNetworksModel* nn_model_ = nullptr;
  // The NN API compilation handle
  ANeuralNetworksCompilation* nn_compiled_model_ = nullptr;
  // Model status
  TfLiteStatus model_status_ = kTfLiteOk;

  // List of state tensors for LSTM, RNN, SVDF.
  // NN API does not allow ops to maintain states across multiple
  // invocations. We need to manually create state input tensors from
  // corresponding state output tensors of TFLite operations, and map them
  // correctly.
  std::vector<int> model_states_inputs_;   // holds NNAPI operand ids
  std::vector<int> model_states_outputs_;  // holds TFLite tensor ids
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_NNAPI_DELEGATE_H_
