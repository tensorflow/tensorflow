/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_SHIMS_C_SHIMS_TEST_UTIL_H_
#define TENSORFLOW_LITE_CORE_SHIMS_C_SHIMS_TEST_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

// Initialize TF Lite shims, in a manner appropriate for running unit tests.
// Returns zero on success, or an implementation-defined error code on failure.
// This should be called before calling any other shims functions or methods
// in unit tests.
int TfLiteInitializeShimsForTest(void);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_CORE_SHIMS_SHIMS_TEST_UTIL_H_
