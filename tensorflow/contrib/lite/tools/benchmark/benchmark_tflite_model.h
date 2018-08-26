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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#ifdef TFLITE_EXTENDED
#include "tensorflow/contrib/lite/delegates/eager/delegate.h"
#endif  // TFLITE_EXTENDED
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/profiling/profile_summarizer.h"
#include "tensorflow/contrib/lite/tools/benchmark/benchmark_model.h"

namespace tflite {
namespace benchmark {

// Dumps profiling events if profiling is enabled
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

// Benchmarks a TFLite model by running tflite interpreter.
class BenchmarkTfLiteModel : public BenchmarkModel {
 public:
  BenchmarkTfLiteModel();
  BenchmarkTfLiteModel(BenchmarkParams params);
  virtual ~BenchmarkTfLiteModel() {}

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  bool ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  void Init() override;
  void RunImpl() override;

  struct InputLayerInfo {
    std::string name;
    std::vector<int> shape;
  };

 private:
#ifdef TFLITE_EXTENDED
  std::unique_ptr<EagerDelegate> delegate_;
#endif  // TFLITE_EXTENDED
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::vector<InputLayerInfo> inputs;
  ProfilingListener profiling_listener_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
