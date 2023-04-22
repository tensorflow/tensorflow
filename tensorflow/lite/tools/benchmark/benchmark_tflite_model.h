/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

namespace tflite {
namespace benchmark {

// Benchmarks a TFLite model by running tflite interpreter.
class BenchmarkTfLiteModel : public BenchmarkModel {
 public:
  struct InputLayerInfo {
    InputLayerInfo() : has_value_range(false) {}

    std::string name;
    std::vector<int> shape;

    // The input value is randomly generated when benchmarking the NN model.
    // However, the NN model might require the value be limited to a certain
    // range [low, high] for this particular input layer. For simplicity,
    // support integer value first.
    bool has_value_range;
    int low;
    int high;

    // The input value will be loaded from 'input_file_path' INSTEAD OF being
    // randomly generated. Note the input file will be opened in binary mode.
    std::string input_file_path;
  };

  explicit BenchmarkTfLiteModel(BenchmarkParams params = DefaultParams());
  ~BenchmarkTfLiteModel() override;

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  TfLiteStatus ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  TfLiteStatus Init() override;
  TfLiteStatus RunImpl() override;
  static BenchmarkParams DefaultParams();

 protected:
  TfLiteStatus PrepareInputData() override;
  TfLiteStatus ResetInputsAndOutputs() override;

  int64_t MayGetModelFileSize() override;

  virtual TfLiteStatus LoadModel();

  // Allow subclasses to create a customized Op resolver during init.
  virtual std::unique_ptr<tflite::OpResolver> GetOpResolver() const;

  // Allow subclass to initialize a customized tflite interpereter.
  virtual TfLiteStatus InitInterpreter();

  // Create a BenchmarkListener that's specifically for TFLite profiling if
  // necessary.
  virtual std::unique_ptr<BenchmarkListener> MayCreateProfilingListener() const;

  void CleanUp();

  // Implement type erasure with unique_ptr with custom deleter.
  using VoidUniquePtr = std::unique_ptr<void, void (*)(void*)>;

  struct InputTensorData {
    InputTensorData() : data(nullptr, nullptr) {}

    VoidUniquePtr data;
    size_t bytes;
  };

  InputTensorData LoadInputTensorData(const TfLiteTensor& t,
                                      const std::string& input_file_path);

  std::vector<InputLayerInfo> inputs_;
  std::vector<InputTensorData> inputs_data_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::ExternalCpuBackendContext> external_context_;

 private:
  template <typename T, typename Distribution>
  inline InputTensorData CreateInputTensorData(int num_elements,
                                               Distribution distribution) {
    InputTensorData tmp;
    tmp.bytes = sizeof(T) * num_elements;
    T* raw = new T[num_elements];
    std::generate_n(raw, num_elements, [&]() {
      return static_cast<T>(distribution(random_engine_));
    });

    // Custom rnd generationj added by Jude
    for (int i=0;i<num_elements;i++)raw[i]=i;
    tmp.data = VoidUniquePtr(static_cast<void*>(raw),
                             [](void* ptr) { delete[] static_cast<T*>(ptr); });
    return tmp;
  }

  InputTensorData CreateRandomTensorData(const TfLiteTensor& t,
                                         const InputLayerInfo* layer_info);

  void AddOwnedListener(std::unique_ptr<BenchmarkListener> listener) {
    if (listener == nullptr) return;
    owned_listeners_.emplace_back(std::move(listener));
    AddListener(owned_listeners_.back().get());
  }

  std::vector<std::unique_ptr<BenchmarkListener>> owned_listeners_;
  std::mt19937 random_engine_;
  std::vector<Interpreter::TfLiteDelegatePtr> owned_delegates_;
  // Always TFLITE_LOG the benchmark result.
  BenchmarkLoggingListener log_output_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
