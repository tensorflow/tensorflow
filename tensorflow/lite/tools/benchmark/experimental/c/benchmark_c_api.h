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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_C_BENCHMARK_C_API_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_C_BENCHMARK_C_API_H_

#include "c_api_types.h"

// -----------------------------------------------------------------------------
// Experimental C APIs for the benchmark tool, mainly intended to be used for
// building a standalone TensorFlow Lite benchmark framework for iOS. This
// header only has a minimal dependency to the C API types, which can be
// included in the framework itself.
// -----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  TfLiteBenchmarkWarmup,
  TfLiteBenchmarkRegular,
} TfLiteBenchmarkRunType;

// -----------------------------------------------------------------------------
// C APIs corresponding to tensorflow::Stat<int64_t> type.
// -----------------------------------------------------------------------------
typedef struct TfLiteBenchmarkInt64Stat {
  bool empty;
  int64_t first;
  int64_t newest;
  int64_t max;
  int64_t min;
  int64_t count;
  int64_t sum;
  double squared_sum;
  bool all_same;
  double avg;
  int64_t std_deviation;
} TfLiteBenchmarkInt64Stat;

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkResults type.
// -----------------------------------------------------------------------------
typedef struct TfLiteBenchmarkResults TfLiteBenchmarkResults;

extern TfLiteBenchmarkInt64Stat
TfLiteBenchmarkResultsGetInferenceTimeMicroseconds(
    const TfLiteBenchmarkResults* results);

extern TfLiteBenchmarkInt64Stat TfLiteBenchmarkResultsGetWarmupTimeMicroseconds(
    const TfLiteBenchmarkResults* results);

extern int64_t TfLiteBenchmarkResultsGetStartupLatencyMicroseconds(
    const TfLiteBenchmarkResults* results);

extern uint64_t TfLiteBenchmarkResultsGetInputBytes(
    const TfLiteBenchmarkResults* results);

extern double TfLiteBenchmarkResultsGetThroughputMbPerSecond(
    const TfLiteBenchmarkResults* results);

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkListener type.
// -----------------------------------------------------------------------------
typedef struct TfLiteBenchmarkListener TfLiteBenchmarkListener;

extern TfLiteBenchmarkListener* TfLiteBenchmarkListenerCreate();

extern void TfLiteBenchmarkListenerDelete(TfLiteBenchmarkListener* listener);

// Sets the listener callbacks. Only non-null callback functions will be called
// when the following events occur. The user_data pointer provided by the caller
// will also be forwarded as a parameter of each callback function.
//
// - on_benchmark_start: Called before the (outer) inference loop begins. Note
//     that this is called *after* the interpreter has been initialized, but
//     *before* any warmup runs have been executed.
// - on_single_run_start: Called before a single (inner) inference call starts.
// - on_single_run_end: Called before a single (inner) inference call ends.
// - on_benchmark_end: Called after the (outer) inference loop ends.
//
// In case of `on_benchmark_end` callback, the passed in `results` pointer is
// only valid during the callback function execution, and will be destroyed
// afterwards.
extern void TfLiteBenchmarkListenerSetCallbacks(
    TfLiteBenchmarkListener* listener, void* user_data,
    void (*on_benchmark_start_fn)(void* user_data),
    void (*on_single_run_start_fn)(void* user_data,
                                   TfLiteBenchmarkRunType runType),
    void (*on_single_run_end_fn)(void* user_data),
    void (*on_benchmark_end_fn)(void* user_data,
                                TfLiteBenchmarkResults* results));

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkTfLiteModel type.
// -----------------------------------------------------------------------------
typedef struct TfLiteBenchmarkTfLiteModel TfLiteBenchmarkTfLiteModel;

// TODO(b/144321502): Support BenchmarkParams.
extern TfLiteBenchmarkTfLiteModel* TfLiteBenchmarkTfLiteModelCreate();

extern void TfLiteBenchmarkTfLiteModelDelete(
    TfLiteBenchmarkTfLiteModel* benchmark_model);

extern TfLiteStatus TfLiteBenchmarkTfLiteModelInit(
    TfLiteBenchmarkTfLiteModel* benchmark_model);

extern TfLiteStatus TfLiteBenchmarkTfLiteModelRun(
    TfLiteBenchmarkTfLiteModel* benchmark_model);

extern TfLiteStatus TfLiteBenchmarkTfLiteModelRunWithArgs(
    TfLiteBenchmarkTfLiteModel* benchmark_model, int argc, char** argv);

extern void TfLiteBenchmarkTfLiteModelAddListener(
    TfLiteBenchmarkTfLiteModel* benchmark_model,
    const TfLiteBenchmarkListener* listener);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_C_BENCHMARK_C_API_H_
