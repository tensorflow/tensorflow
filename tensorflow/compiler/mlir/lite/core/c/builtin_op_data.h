/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_BUILTIN_OP_DATA_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_BUILTIN_OP_DATA_H_

// LINT.IfChange(enum)
typedef enum {
  kTfLitePaddingUnknown = 0,
  kTfLitePaddingSame,
  kTfLitePaddingValid,
} TfLitePadding;

// Possible fused activation functions.
typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActReluN1To1,  // min(max(-1, x), 1)
  kTfLiteActRelu6,      // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;
// LINT.ThenChange(//tensorflow/lite/core/c/builtin_op_data.h)

// LINT.IfChange(struct)
// TODO(b/130259536): We should move this out of builtin_op_data.
typedef struct {
  int width;
  int height;
  int width_offset;
  int height_offset;
} TfLitePaddingValues;

typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int filter_width;
  int filter_height;
  TfLiteFusedActivation activation;
  struct {
    TfLitePaddingValues padding;
  } computed;
} TfLitePoolParams;
// LINT.ThenChange(//tensorflow/lite/core/c/builtin_op_data.h)

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_BUILTIN_OP_DATA_H_
