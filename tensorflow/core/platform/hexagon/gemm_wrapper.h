/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_HEXAGON_GEMM_WRAPPER_H_
#define TENSORFLOW_PLATFORM_HEXAGON_GEMM_WRAPPER_H_

// Declaration of APIs provided by hexagon shared library. This header is shared
// with both hexagon library built with qualcomm SDK and tensorflow.
// All functions defined here must have prefix "hexagon_gemm_wrapper" to avoid
// naming conflicts.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
// Returns the version of loaded hexagon wrapper shared library.
// You should assert that the version matches the expected version before
// calling APIs defined in this header.
int hexagon_gemm_wrapper_GetWrapperVersion();
// Returns the version of hexagon binary.
// You should assert that the version matches the expected version before
// calling APIs defined in this header.
int hexagon_gemm_wrapper_GetHexagonBinaryVersion();
// TODO(satok): Support gemm APIs via RPC
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_PLATFORM_HEXAGON_GEMM_WRAPPER_H_
