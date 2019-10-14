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

#include <map>
#include <memory>
#include <string>
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
  };

  BenchmarkTfLiteModel();
  explicit BenchmarkTfLiteModel(BenchmarkParams params);
  ~BenchmarkTfLiteModel() override;

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  TfLiteStatus ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  TfLiteStatus Init() override;
  TfLiteStatus RunImpl() override;

 protected:
  static BenchmarkParams DefaultParams();
  TfLiteStatus PrepareInputData() override;
  TfLiteStatus ResetInputsAndOutputs() override;

  // Allow subclasses to create custom delegates to be applied during init.
  using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
  using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;
  virtual TfLiteDelegatePtrMap GetDelegates() const;

  // Allow subclasses to create a customized Op resolver during init.
  virtual std::unique_ptr<tflite::OpResolver> GetOpResolver() const;

  void CleanUp();

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

 private:
  struct InputTensorData {
    InputTensorData() : bytes(0) { data.raw = nullptr; }
    TfLitePtrUnion data;
    size_t bytes;
  };
  std::vector<InputLayerInfo> inputs_;
  std::vector<InputTensorData> inputs_data_;
  std::unique_ptr<BenchmarkListener> profiling_listener_;
  std::unique_ptr<BenchmarkListener> gemmlowp_profiling_listener_;
  TfLiteDelegatePtrMap delegates_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
