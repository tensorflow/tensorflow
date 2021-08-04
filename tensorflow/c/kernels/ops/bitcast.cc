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

#include <sstream>
#include <string>

#include "tensorflow/c/ops.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

static void ComputeNewShape(TF_ShapeInferenceContext* ctx,
                            TF_ShapeHandle* shape, TF_DataType input_type,
                            TF_DataType output_type, TF_Status* status) {
  size_t input_type_size = TF_DataTypeSize(input_type);
  size_t output_type_size = TF_DataTypeSize(output_type);

  if (input_type_size == 0 || output_type_size == 0) {
    std::ostringstream err;
    err << "Cannot bitcast type " << input_type << " to " << output_type
        << " because one of the type sizes is zero";
    TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
    return;
  }

  TF_SetStatus(status, TF_OK, "");
  if (input_type_size < output_type_size) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, shape, 1, shape, status);

    if (TF_GetCode(status) == TF_OK) {
      TF_DimensionHandle* last_dim = TF_NewDimensionHandle();
      size_t divisor_val = output_type_size / input_type_size;
      TF_ShapeInferenceContextDim(ctx, shape, -1, last_dim);
      if (!TF_DimensionHandleValueKnown(last_dim) ||
          TF_DimensionHandleValue(last_dim) == divisor_val) {
        TF_ShapeInferenceContextSubshape(ctx, shape, 0, -1, shape, status);
      } else {
        std::ostringstream err;
        err << "Cannot bitcast from " << input_type << " to " << output_type
            << " due to shape. " << TF_DimensionHandleValue(last_dim)
            << " does not match " << divisor_val;
        TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
      }
      TF_DeleteDimensionHandle(last_dim);
    }
  } else if (input_type_size > output_type_size) {
    // Input type size is larger than output type size.
    size_t divisor_val = input_type_size / output_type_size;
    TF_ShapeHandle* extension =
        TF_ShapeInferenceContextVectorFromSize(ctx, divisor_val);
    TF_ShapeInferenceContextConcatenateShapes(ctx, shape, extension, shape,
                                              status);
    TF_DeleteShapeHandle(extension);
  }
}

static void bitcast_shape_inference_fn(TF_ShapeInferenceContext* ctx,
                                       TF_Status* status) {
  TF_ShapeHandle* result = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, result, status);
  if (TF_GetCode(status) == TF_OK &&
      !TF_ShapeInferenceContextRankKnown(ctx, result)) {
    TF_ShapeInferenceContextSetUnknownShape(ctx, status);
    TF_DeleteShapeHandle(result);
    return;
  }

  // Find the size of the input and output data types.
  TF_DataType input_type;
  TF_DataType output_type;

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContext_GetAttrType(ctx, "T", &input_type, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContext_GetAttrType(ctx, "type", &output_type, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    ComputeNewShape(ctx, result, input_type, output_type, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, result, status);
  }
  TF_DeleteShapeHandle(result);
}

void RegisterBitcastOp() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Bitcast");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: type");
  TF_OpDefinitionBuilderAddAttr(
      op_builder,
      "T: {bfloat16, half, float, double, int64, int32, uint8, uint16, "
      "uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, "
      "qint16, quint16, qint32}");
  TF_OpDefinitionBuilderAddAttr(
      op_builder,
      "type: {bfloat16, half, float, double, int64, int32, uint8, uint16, "
      "uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, "
      "qint16, quint16, qint32}");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &bitcast_shape_inference_fn);

  TF_RegisterOpDefinition(op_builder, status);
  CHECK_EQ(TF_GetCode(status), TF_OK)
      << "Bitcast op registration failed: " << TF_Message(status);
  TF_DeleteStatus(status);
}

TF_ATTRIBUTE_UNUSED static bool IsBitcastOpRegistered = []() {
  if (SHOULD_REGISTER_OP("Bitcast")) {
    RegisterBitcastOp();
  }
  return true;
}();
