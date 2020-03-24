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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_EXPORT_FLAGS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_EXPORT_FLAGS_H_

#include <string>

// These flags are used to control the emission or not of different kinds of ops
// during the flatbuffer translation.
extern bool emit_builtin_tflite_ops;
extern bool emit_select_tf_ops;
extern bool emit_custom_ops;
// The flag to control whether to lower tensorlist ops into TF ops.
extern bool lower_tensor_list_ops;
// The flag to control whether debug info gets stripped on export.
extern bool strip_debug_info;

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_EXPORT_FLAGS_H_
