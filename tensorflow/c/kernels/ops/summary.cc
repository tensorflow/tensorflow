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

#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

static void scalar_summary_shape_inference_fn(TF_ShapeInferenceContext* ctx,
                                              TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* result = TF_ShapeInferenceContextScalar(ctx);
  TF_ShapeInferenceContextSetOutput(ctx, 0, result, status);
  TF_DeleteShapeHandle(result);
}

void Register_ScalarSummaryOp() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("ScalarSummary");
  TF_OpDefinitionBuilderAddInput(op_builder, "tags: string");
  TF_OpDefinitionBuilderAddInput(op_builder, "values: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "summary: string");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: realnumbertype");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(
      op_builder, &scalar_summary_shape_inference_fn);

  TF_RegisterOpDefinition(op_builder, status);
  CHECK_EQ(TF_GetCode(status), TF_OK)
      << "ScalarSummary op registration failed: " << TF_Message(status);
  TF_DeleteStatus(status);
}

TF_ATTRIBUTE_UNUSED static bool SummaryScalarOpRegistered = []() {
  if (&TF_NewStatus != nullptr && SHOULD_REGISTER_OP("ScalarSummary")) {
    Register_ScalarSummaryOp();
  }
  return true;
}();
