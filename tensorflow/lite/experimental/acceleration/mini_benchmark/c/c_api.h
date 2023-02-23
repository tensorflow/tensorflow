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
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// APIs of TfLiteMiniBenchmarkResult.
typedef struct TfLiteMiniBenchmarkResult TfLiteMiniBenchmarkResult;
int TfLiteMiniBenchmarkResultInitStatus(TfLiteMiniBenchmarkResult* result);
uint8_t* TfLiteMiniBenchmarkResultFlatBufferData(
    TfLiteMiniBenchmarkResult* result);
size_t TfLiteMiniBenchmarkResultFlatBufferDataSize(
    TfLiteMiniBenchmarkResult* result);
// Free memory allocated with `result`.
void TfLiteMiniBenchmarkResultFree(TfLiteMiniBenchmarkResult* result);

// APIs of TfLiteMiniBenchmarkCustomValidationInfo.
typedef struct TfLiteMiniBenchmarkCustomValidationInfo
    TfLiteMiniBenchmarkCustomValidationInfo;
void TfLiteMiniBenchmarkCustomValidationInfoSetBuffer(
    TfLiteMiniBenchmarkCustomValidationInfo* custom_validation, int batch_size,
    uint8_t* buffer, size_t* buffer_dim, int buffer_dim_size);
void TfLiteMiniBenchmarkCustomValidationInfoSetAccuracyValidator(
    TfLiteMiniBenchmarkCustomValidationInfo* custom_validation,
    void* accuracy_validator_user_data,
    bool (*accuracy_validator_func)(void* user_data,
                                    uint8_t* benchmark_result_data,
                                    int benchmark_result_data_size));

// APIs of TfLiteMiniBenchmarkSettings.
typedef struct TfLiteMiniBenchmarkSettings TfLiteMiniBenchmarkSettings;
TfLiteMiniBenchmarkSettings* TfLiteMiniBenchmarkSettingsCreate();
TfLiteMiniBenchmarkCustomValidationInfo*
TfLiteMiniBenchmarkSettingsCustomValidationInfo(
    TfLiteMiniBenchmarkSettings* settings);
void TfLiteMiniBenchmarkSettingsSetFlatBufferData(
    TfLiteMiniBenchmarkSettings* settings, uint8_t* flatbuffer_data,
    size_t flatbuffer_data_size);
void TfLiteMiniBenchmarkSettingsSetErrorReporter(
    TfLiteMiniBenchmarkSettings* settings, void* error_reporter_user_data,
    int (*error_reporter_func)(void* user_data, const char* format,
                               va_list args));
void TfLiteMiniBenchmarkSettingsSetGpuPluginHandle(
    TfLiteMiniBenchmarkSettings* settings, void* gpu_plugin_handle);
void TfLiteMiniBenchmarkSettingsFree(TfLiteMiniBenchmarkSettings* settings);

// Others.
// Trigger validation for `settings` and return the validation result.
// This returns a pointer, that you must free using
// TfLiteMiniBenchmarkResultFree().
TfLiteMiniBenchmarkResult* TfLiteBlockingValidatorRunnerTriggerValidation(
    TfLiteMiniBenchmarkSettings* settings);

#ifdef __cplusplus
}  // extern "C".
#endif
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_C_C_API_H_
