/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERTER_FLAGS_TO_V2_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERTER_FLAGS_TO_V2_OPTIONS_H_

#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/converter_v2_options.pb.h"

namespace tflite {

// Returns the Converter V2 options that `converter_flags` sets. Keeps the
// `ConverterFlags`-based Python entry of the V2 path working. Flags that the V2
// path does not support are ignored. `enable_composite_direct_lowering` and
// `strict_qdq_mode` are among them: the V2 path always behaves as if both were
// set. So are `model_origin_framework` and `fold_fp16_resource_casts`, which
// the V2 path does not read.
ConverterV2Options ToConverterV2Options(const ConverterFlags& converter_flags);

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERTER_FLAGS_TO_V2_OPTIONS_H_
