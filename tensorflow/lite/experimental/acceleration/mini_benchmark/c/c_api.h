/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_C_C_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_C_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include <cstdarg>

#ifdef __cplusplus
extern "C" {
#endif

// The result of triggering MiniBenchmark.
typedef struct TfLiteMiniBenchmarkResult {
  // MinibenchmarkStatus of whether test is initialized successfully. The value
  // maps to status_codes.h.
  int init_status;
  // The pointer to a stream of BenchmarkEvent(s). Size of each event is
  // prefixed.
  uint8_t* flatbuffer_data;
  // The byte size of the flatbuffer_data.
  size_t flatbuffer_data_size;
} TfLiteMiniBenchmarkResult;

// Custom validation related info. For forward source compatibility, this
// struct should always be brace-initialized, so that all fields (including any
// that might be added in the future) get zero-initialized.
typedef struct TfLiteMiniBenchmarkCustomValidationInfo {
  // The batch number of custom input.
  int batch_size;
  // Length of buffer_dim.
  int buffer_dim_size;
  // The size of each custom input within buffer.
  size_t* buffer_dim;
  // Pointer to concatenated custom input data. At embedding time, the
  // i-th input tensor buffer starts from sum(buffer_dim[0...i-1]) to
  // sum(buffer_dim[0...i]).
  uint8_t* buffer;
  // Arbitrary data that will be passed  to the `accuracy_validator_func`
  // function via its `user_data` parameter.
  void* accuracy_validator_user_data;
  // Custom validation rule that decides whether a BenchmarkResult passes the
  // accuracy check.
  bool (*accuracy_validator_func)(void* user_data,
                                  uint8_t* benchmark_result_data,
                                  int benchmark_result_data_size);
} TfLiteMiniBenchmarkCustomValidationInfo;

// Mini-benchmark settings. For forward source compatibility, this struct
// should always be brace-initialized, so that all fields (including any that
// might be added in the future) get zero-initialized.
typedef struct TfLiteMiniBenchmarkSettings {
  // The pointer to a flatbuffer data of MinibenchmarkSettings.
  uint8_t* flatbuffer_data;
  // The byte size of the flatbuffer_data.
  size_t flatbuffer_data_size;
  // Custom validation related info.
  TfLiteMiniBenchmarkCustomValidationInfo custom_validation_info;
  // Arbitrary data that will be passed  to the `error_reporter_func`
  // function via its `user_data` parameter.
  void* error_reporter_user_data;
  // Custom error reporter to log error to. If the function is provided, errors
  // will be logged with this function.
  int (*error_reporter_func)(void* user_data, const char* format, va_list args);
} TfLiteMinibenchmarkSettings;

// Trigger validation for `settings` and return the validation result.
// This returns a pointer, that you must free using
// TfLiteMiniBenchmarkResultFree().
TfLiteMiniBenchmarkResult* TfLiteBlockingValidatorRunnerTriggerValidation(
    TfLiteMinibenchmarkSettings* settings);

// Free memory allocated with `result`.
void TfLiteMiniBenchmarkResultFree(TfLiteMiniBenchmarkResult* result);

#ifdef __cplusplus
}  // extern "C".
#endif
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_C_C_API_H_
