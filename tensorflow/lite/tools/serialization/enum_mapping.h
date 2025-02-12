/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_SERIALIZATION_ENUM_MAPPING_H_
#define TENSORFLOW_LITE_TOOLS_SERIALIZATION_ENUM_MAPPING_H_

#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/builtin_op_data.h"

// TODO(aselle): Ideally extract this from the schema.

namespace tflite {

inline ActivationFunctionType TfLiteActivationToSchemaActivation(
    TfLiteFusedActivation act) {
  switch (act) {
    case kTfLiteActNone:
      return ActivationFunctionType_NONE;
    case kTfLiteActRelu:
      return ActivationFunctionType_RELU;
    case kTfLiteActReluN1To1:
      return ActivationFunctionType_RELU_N1_TO_1;
    case kTfLiteActRelu6:
      return ActivationFunctionType_RELU6;
    case kTfLiteActTanh:
      return ActivationFunctionType_TANH;
    case kTfLiteActSignBit:
      return ActivationFunctionType_SIGN_BIT;
    case kTfLiteActSigmoid:
      return ActivationFunctionType_NONE;  // TODO(aselle): Add to schema
  }
  return ActivationFunctionType_NONE;
}

inline Padding TfLitePaddingToSchemaPadding(TfLitePadding padding) {
  switch (padding) {
    case kTfLitePaddingUnknown:
      return Padding_SAME;  // TODO(aselle): Consider an error.
    case kTfLitePaddingSame:
      return Padding_SAME;
    case kTfLitePaddingValid:
      return Padding_VALID;
  }
  return Padding_SAME;  // TODO(aselle): Consider an error.
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
  switch (type) {
    // case kTfLiteNoType: return TensorType_NONE;
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(aselle): Consider an error.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteBFloat16:
      return TensorType_BFLOAT16;
    case kTfLiteFloat64:
      return TensorType_FLOAT64;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt32:
      return TensorType_UINT32;
    case kTfLiteInt4:
      return TensorType_INT4;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteUInt64:
      return TensorType_UINT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteUInt16:
      return TensorType_UINT16;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
    case kTfLiteComplex128:
      return TensorType_COMPLEX128;
    case kTfLiteResource:
      return TensorType_RESOURCE;
    case kTfLiteVariant:
      return TensorType_VARIANT;
  }
  // TODO(aselle): consider an error
}

inline FullyConnectedOptionsWeightsFormat
FullyConnectedOptionsWeightsFormatToSchema(
    TfLiteFullyConnectedWeightsFormat format) {
  switch (format) {
    case kTfLiteFullyConnectedWeightsFormatDefault:
      return FullyConnectedOptionsWeightsFormat_DEFAULT;
    case kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8:
      return FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8;
  }
}

inline LSTMKernelType LSTMKernelTypeToSchema(TfLiteLSTMKernelType type) {
  switch (type) {
    case kTfLiteLSTMFullKernel:
      return LSTMKernelType_FULL;
    case kTfLiteLSTMBasicKernel:
      return LSTMKernelType_BASIC;
  }
}

inline LSHProjectionType LSHProjectionTypeToSchema(
    TfLiteLSHProjectionType type) {
  switch (type) {
    case kTfLiteLshProjectionUnknown:
      return LSHProjectionType_UNKNOWN;
    case kTfLiteLshProjectionSparse:
      return LSHProjectionType_SPARSE;
    case kTfLiteLshProjectionDense:
      return LSHProjectionType_DENSE;
  }
}

inline MirrorPadMode MirrorPaddingModeToSchema(TfLiteMirrorPaddingMode mode) {
  switch (mode) {
    case kTfLiteMirrorPaddingUnknown:
      return MirrorPadMode_REFLECT;  // TODO(aselle): consider an error
    case kTfLiteMirrorPaddingReflect:
      return MirrorPadMode_REFLECT;
    case kTfLiteMirrorPaddingSymmetric:
      return MirrorPadMode_SYMMETRIC;
  }
}

inline CombinerType CombinerTypeToSchema(TfLiteCombinerType type) {
  switch (type) {
    case kTfLiteCombinerTypeSum:
      return CombinerType_SUM;
    case kTfLiteCombinerTypeMean:
      return CombinerType_MEAN;
    case kTfLiteCombinerTypeSqrtn:
      return CombinerType_SQRTN;
  }
}

// int

}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_SERIALIZATION_ENUM_MAPPING_H_
