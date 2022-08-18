/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_STATUS_CODES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_STATUS_CODES_H_

namespace tflite {
namespace acceleration {
// A unified set of status codes for mini-benchmark.
//
// The overall mini benchmark infrastructure is multi-layered and its behaviour
// depends on app packaging, Android target and device SDK version, delegates
// and drivers. We want to get detailed telemetry so that issues in-the-wild can
// be diagnosed and potentially reproduced.
//
// This enum is used as a single source of truth for the possible error
// conditions encountered. Layers can pass status codes upwards unchanged.
//
// (absl::Status and friends are not allowed in the TFLite codebase for version
// skew and binary size reasons).
enum MinibenchmarkStatus {
  kMinibenchmarkUnknownStatus = 0,

  // First set of error codes that are used as process exit codes to communicate
  // between the parent and child process. The values need to be between 1 and
  // 126 to be passed through popen().
  //
  // Runner main status codes used to indicate inability to dynamically load and
  // execute the validation code.
  // LINT.IfChange
  kMinibenchmarkRunnerMainDlopenFailed = 11,
  kMinibenchmarkRunnerMainSymbolLookupFailed = 12,
  kMinibenchmarkRunnerMainTooFewArguments = 13,
  kMinibenchmarkUnsupportedPlatform = 14,
  // LINT.ThenChange(//tensorflow/lite/experimental/acceleration/mini_benchmark/runner_main.c)
  // General status codes that may be used anywhere
  kMinibenchmarkPreconditionNotMet = 119,
  kMinibenchmarkSuccess = 120,
  // Storage status codes. These are used when storage can not be used to pass
  // status.
  kMinibenchmarkCorruptSizePrefixedFlatbufferFile = 21,
  kMinibenchmarkCantCreateStorageFile = 22,
  kMinibenchmarkFlockingStorageFileFailed = 23,
  kMinibenchmarkErrorReadingStorageFile = 24,
  kMinibenchmarkFailedToOpenStorageFileForWriting = 25,
  kMinibenchmarkErrorWritingStorageFile = 26,
  kMinibenchmarkErrorFsyncingStorageFile = 27,
  kMinibenchmarkErrorClosingStorageFile = 28,

  // Second set of error codes that are used either before launching the child
  // process or communicated through the storage mechanism. These can be > 127.
  //
  // Runner status codes.
  kMinibenchmarkDladdrReturnedZero = 502,
  kMinibenchmarkDliFnameWasNull = 503,
  kMinibenchmarkDliFnameHasApkNameOnly = 504,
  kMinibenchmarkRequestAndroidInfoFailed = 505,
  kMinibenchmarkDliFnameDoesntContainSlashes = 506,
  kMinibenchmarkCouldntOpenTemporaryFileForBinary = 507,
  kMinibenchmarkCouldntChmodTemporaryFile = 508,
  kMinibenchmarkPopenFailed = 509,
  kMinibenchmarkCommandFailed = 510,
  kMiniBenchmarkCannotLoadSupportLibrary = 511,
  kMiniBenchmarkInvalidSupportLibraryConfiguration = 512,
  // Validator status codes.
  kMinibenchmarkDelegateNotSupported = 1000,
  kMinibenchmarkDelegatePluginNotFound = 1001,
  kMinibenchmarkDelegateCreateFailed = 1013,
  kMinibenchmarkModelTooLarge = 1002,  // Safety limit currently set at 100M.
  kMinibenchmarkSeekToModelOffsetFailed = 1003,
  kMinibenchmarkModelReadFailed = 1004,
  kMinibenchmarkInterpreterBuilderFailed = 1005,
  kMinibenchmarkValidationSubgraphNotFound = 1006,
  kMinibenchmarkModifyGraphWithDelegateFailed = 1007,
  kMinibenchmarkAllocateTensorsFailed = 1008,
  kMinibenchmarkInvokeFailed = 1009,
  kMinibenchmarkModelBuildFailed = 1010,
  kMinibenchmarkValidationSubgraphHasTooFewInputs = 1011,
  kMinibenchmarkValidationSubgraphHasTooFewOutputs = 1011,
  kMinibenchmarkValidationSubgraphInputsDontMatchOutputs = 1012,

  // Validator runner status codes.
  kMinibenchmarkChildProcessAlreadyRunning = 1501,
  kMinibenchmarkValidationEntrypointSymbolNotFound = 1502,
  kMinibenchmarkNoValidationRequestFound = 1503,

  // Validator runner recoverable errors
  kMinibenchmarkUnableToSetCpuAffinity = 1601,
};
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_STATUS_CODES_H_
