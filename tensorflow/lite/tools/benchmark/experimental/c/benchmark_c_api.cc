/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.h"

#include <utility>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

extern "C" {

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkResults type.
// -----------------------------------------------------------------------------
struct TfLiteBenchmarkResults {
  const tflite::benchmark::BenchmarkResults* results;
};

// Converts the given int64_t stat into a TfLiteBenchmarkInt64Stat struct.
TfLiteBenchmarkInt64Stat ConvertStat(const tensorflow::Stat<int64_t>& stat) {
  return {
      stat.empty(),    stat.first(), stat.newest(),        stat.max(),
      stat.min(),      stat.count(), stat.sum(),           stat.squared_sum(),
      stat.all_same(), stat.avg(),   stat.std_deviation(),
  };
}

TfLiteBenchmarkInt64Stat TfLiteBenchmarkResultsGetInferenceTimeMicroseconds(
    const TfLiteBenchmarkResults* results) {
  return ConvertStat(results->results->inference_time_us());
}

TfLiteBenchmarkInt64Stat TfLiteBenchmarkResultsGetWarmupTimeMicroseconds(
    const TfLiteBenchmarkResults* results) {
  return ConvertStat(results->results->warmup_time_us());
}

int64_t TfLiteBenchmarkResultsGetStartupLatencyMicroseconds(
    const TfLiteBenchmarkResults* results) {
  return results->results->startup_latency_us();
}

uint64_t TfLiteBenchmarkResultsGetInputBytes(
    const TfLiteBenchmarkResults* results) {
  return results->results->input_bytes();
}

double TfLiteBenchmarkResultsGetThroughputMbPerSecond(
    const TfLiteBenchmarkResults* results) {
  return results->results->throughput_MB_per_second();
}

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkListener type.
// -----------------------------------------------------------------------------
class BenchmarkListenerAdapter : public tflite::benchmark::BenchmarkListener {
 public:
  void OnBenchmarkStart(
      const tflite::benchmark::BenchmarkParams& params) override {
    if (on_benchmark_start_fn_ != nullptr) {
      on_benchmark_start_fn_(user_data_);
    }
  }

  void OnSingleRunStart(tflite::benchmark::RunType runType) override {
    if (on_single_run_start_fn_ != nullptr) {
      on_single_run_start_fn_(user_data_, runType == tflite::benchmark::WARMUP
                                              ? TfLiteBenchmarkWarmup
                                              : TfLiteBenchmarkRegular);
    }
  }

  void OnSingleRunEnd() override {
    if (on_single_run_end_fn_ != nullptr) {
      on_single_run_end_fn_(user_data_);
    }
  }

  void OnBenchmarkEnd(
      const tflite::benchmark::BenchmarkResults& results) override {
    if (on_benchmark_end_fn_ != nullptr) {
      TfLiteBenchmarkResults* wrapper = new TfLiteBenchmarkResults{&results};
      on_benchmark_end_fn_(user_data_, wrapper);
      delete wrapper;
    }
  }

  // Keep the user_data pointer provided when setting the callbacks.
  void* user_data_;

  // Function pointers set by the TfLiteBenchmarkListenerSetCallbacks call.
  // Only non-null callbacks will be actually called.
  void (*on_benchmark_start_fn_)(void* user_data);
  void (*on_single_run_start_fn_)(void* user_data,
                                  TfLiteBenchmarkRunType runType);
  void (*on_single_run_end_fn_)(void* user_data);
  void (*on_benchmark_end_fn_)(void* user_data,
                               TfLiteBenchmarkResults* results);
};

struct TfLiteBenchmarkListener {
  std::unique_ptr<BenchmarkListenerAdapter> adapter;
};

TfLiteBenchmarkListener* TfLiteBenchmarkListenerCreate() {
  std::unique_ptr<BenchmarkListenerAdapter> adapter(
      new BenchmarkListenerAdapter());
  return new TfLiteBenchmarkListener{std::move(adapter)};
}

void TfLiteBenchmarkListenerDelete(TfLiteBenchmarkListener* listener) {
  delete listener;
}

void TfLiteBenchmarkListenerSetCallbacks(
    TfLiteBenchmarkListener* listener, void* user_data,
    void (*on_benchmark_start_fn)(void* user_data),
    void (*on_single_run_start_fn)(void* user_data,
                                   TfLiteBenchmarkRunType runType),
    void (*on_single_run_end_fn)(void* user_data),
    void (*on_benchmark_end_fn)(void* user_data,
                                TfLiteBenchmarkResults* results)) {
  listener->adapter->user_data_ = user_data;
  listener->adapter->on_benchmark_start_fn_ = on_benchmark_start_fn;
  listener->adapter->on_single_run_start_fn_ = on_single_run_start_fn;
  listener->adapter->on_single_run_end_fn_ = on_single_run_end_fn;
  listener->adapter->on_benchmark_end_fn_ = on_benchmark_end_fn;
}

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkTfLiteModel type.
// -----------------------------------------------------------------------------
struct TfLiteBenchmarkTfLiteModel {
  std::unique_ptr<tflite::benchmark::BenchmarkTfLiteModel> benchmark_model;
};

TfLiteBenchmarkTfLiteModel* TfLiteBenchmarkTfLiteModelCreate() {
  std::unique_ptr<tflite::benchmark::BenchmarkTfLiteModel> benchmark_model(
      new tflite::benchmark::BenchmarkTfLiteModel());
  return new TfLiteBenchmarkTfLiteModel{std::move(benchmark_model)};
}

void TfLiteBenchmarkTfLiteModelDelete(
    TfLiteBenchmarkTfLiteModel* benchmark_model) {
  delete benchmark_model;
}

TfLiteStatus TfLiteBenchmarkTfLiteModelInit(
    TfLiteBenchmarkTfLiteModel* benchmark_model) {
  return benchmark_model->benchmark_model->Init();
}

TfLiteStatus TfLiteBenchmarkTfLiteModelRun(
    TfLiteBenchmarkTfLiteModel* benchmark_model) {
  return benchmark_model->benchmark_model->Run();
}

TfLiteStatus TfLiteBenchmarkTfLiteModelRunWithArgs(
    TfLiteBenchmarkTfLiteModel* benchmark_model, int argc, char** argv) {
  return benchmark_model->benchmark_model->Run(argc, argv);
}

void TfLiteBenchmarkTfLiteModelAddListener(
    TfLiteBenchmarkTfLiteModel* benchmark_model,
    const TfLiteBenchmarkListener* listener) {
  return benchmark_model->benchmark_model->AddListener(listener->adapter.get());
}

}  // extern "C"
