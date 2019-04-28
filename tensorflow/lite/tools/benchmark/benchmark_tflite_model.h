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
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

namespace tflite {
namespace benchmark {

// Dumps profiling events if profiling is enabled.
class ProfilingListener : public BenchmarkListener {
 public:
  explicit ProfilingListener() : interpreter_(nullptr), has_profiles_(false) {}

  void SetInterpreter(Interpreter* interpreter);

  void OnSingleRunStart(RunType run_type) override;

  void OnSingleRunEnd() override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  Interpreter* interpreter_;
  profiling::Profiler profiler_;
  profiling::ProfileSummarizer summarizer_;
  bool has_profiles_;
};

// Dumps gemmlowp profiling events if gemmlowp profiling is enabled.
class GemmlowpProfilingListener : public BenchmarkListener {
 public:
  virtual ~GemmlowpProfilingListener() {}

  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;
};

// Benchmarks a TFLite model by running tflite interpreter.
class BenchmarkTfLiteModel : public BenchmarkModel {
 public:
  struct InputLayerInfo {
    std::string name;
    std::vector<int> shape;
  };

  BenchmarkTfLiteModel();
  explicit BenchmarkTfLiteModel(BenchmarkParams params);
  virtual ~BenchmarkTfLiteModel();

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  bool ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  void Init() override;
  void RunImpl() override;

 protected:
  static BenchmarkParams DefaultParams();
  void PrepareInputData() override;
  void ResetInputsAndOutputs() override;

  // Allow subclasses to create custom delegates to be applied during init.
  using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
  using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;
  virtual TfLiteDelegatePtrMap GetDelegates() const;

  void CleanUp();

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;

 private:
  struct InputTensorData {
    TfLitePtrUnion data;
    size_t bytes;
  };
  std::vector<InputLayerInfo> inputs;
  std::vector<InputTensorData> inputs_data_;
  ProfilingListener profiling_listener_;
  GemmlowpProfilingListener gemmlowp_profiling_listener_;
  TfLiteDelegatePtrMap delegates_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
