/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
// This provides utility macros and functions that are inherently platform
// specific or shared across runtime & converter.
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_MACROS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_MACROS_H_

#ifndef TF_LITE_STATIC_MEMORY
// maximum size of a valid flatbuffer
inline constexpr unsigned int flatbuffer_size_max = 2147483648;
// If none zero then the buffer is stored outside of the flatbuffers, string
inline constexpr char tflite_metadata_buffer_location[] = "buffer_location";
// field for minimum runtime version, string
inline constexpr char tflite_metadata_min_runtime_version[] =
    "min_runtime_version";
// the stablehlo op version is supported by the tflite runtime
inline constexpr char tflite_supported_stablehlo_version[] = "1.0.0";
#endif

// LINT.IfChange(TFLITE_NOINLINE)

#ifdef _WIN32
#define TFLITE_NOINLINE __declspec(noinline)
#else
#ifdef __has_attribute
#if __has_attribute(noinline)
#define TFLITE_NOINLINE __attribute__((noinline))
#else
#define TFLITE_NOINLINE
#endif  // __has_attribute(noinline)
#else
#define TFLITE_NOINLINE
#endif  // __has_attribute
#endif  // _WIN32

// LINT.ThenChange(//tensorflow/lite/core/macros.h)

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_MACROS_H_
