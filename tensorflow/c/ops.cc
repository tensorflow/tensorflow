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

#include "tensorflow/c/ops.h"

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::DataType;
using ::tensorflow::OpDef;
using ::tensorflow::OpDefBuilder;
using ::tensorflow::OpDeprecation;
using ::tensorflow::OpShapeInferenceFn;
using ::tensorflow::Set_TF_Status_from_Status;
using ::tensorflow::Status;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

TF_OpDefinitionBuilder* TF_NewOpDefinitionBuilder(const char* op_name) {
  auto* result = new OpDefBuilder(op_name);
  return reinterpret_cast<TF_OpDefinitionBuilder*>(result);
}

void TF_DeleteOpDefinitionBuilder(TF_OpDefinitionBuilder* builder) {
  delete reinterpret_cast<OpDefBuilder*>(builder);
}

void TF_OpDefinitionBuilderAddInput(TF_OpDefinitionBuilder* builder,
                                    const char* input_spec) {
  reinterpret_cast<OpDefBuilder*>(builder)->Input(input_spec);
}

void TF_OpDefinitionBuilderAddOutput(TF_OpDefinitionBuilder* builder,
                                     const char* output_spec) {
  reinterpret_cast<OpDefBuilder*>(builder)->Output(output_spec);
}

#define DEFINE_BUILDER_BOOL_SETTER(func_name)                             \
  void TF_OpDefinitionBuilder##func_name(TF_OpDefinitionBuilder* builder, \
                                         bool arg_name) {                 \
    reinterpret_cast<OpDefBuilder*>(builder)->func_name();                \
  }

DEFINE_BUILDER_BOOL_SETTER(SetIsCommutative)
DEFINE_BUILDER_BOOL_SETTER(SetIsAggregate)
DEFINE_BUILDER_BOOL_SETTER(SetIsStateful)
DEFINE_BUILDER_BOOL_SETTER(SetAllowsUninitializedInput)

void TF_OpDefinitionBuilderAddAttr(TF_OpDefinitionBuilder* builder,
                                   const char* attr_spec) {
  reinterpret_cast<OpDefBuilder*>(builder)->Attr(attr_spec);
}

void TF_OpDefinitionBuilderDeprecated(TF_OpDefinitionBuilder* builder,
                                      int version, const char* explanation) {
  reinterpret_cast<OpDefBuilder*>(builder)->Deprecated(version, explanation);
}

void TF_RegisterOpDefinition(TF_OpDefinitionBuilder* builder,
                             TF_Status* status) {
  auto* cc_builder = reinterpret_cast<OpDefBuilder*>(builder);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::OpRegistry::Global()->Register(
      [cc_builder](::tensorflow::OpRegistrationData* op_reg_data) -> Status {
        Status result = cc_builder->Finalize(op_reg_data);
        delete cc_builder;
        return result;
      });
}

void TF_OpDefinitionBuilderSetShapeInferenceFunction(
    TF_OpDefinitionBuilder* builder,
    void (*shape_inference_func)(TF_ShapeInferenceContext* ctx,
                                 TF_Status* status)) {
  auto* cc_builder = reinterpret_cast<OpDefBuilder*>(builder);
  cc_builder->SetShapeFn(
      [shape_inference_func](InferenceContext* ctx) -> tensorflow::Status {
        TF_Status* c_status = TF_NewStatus();
        auto c_ctx = reinterpret_cast<TF_ShapeInferenceContext*>(ctx);
        shape_inference_func(c_ctx, c_status);
        tensorflow::Status result = ::tensorflow::StatusFromTF_Status(c_status);
        TF_DeleteStatus(c_status);
        return result;
      });
}

TF_ShapeHandle* TF_NewShapeHandle() {
  return reinterpret_cast<TF_ShapeHandle*>(new ShapeHandle);
}

TF_ShapeHandle* TF_ShapeInferenceContextVectorFromSize(
    TF_ShapeInferenceContext* ctx, size_t size) {
  auto* handle = new ShapeHandle;
  *handle = reinterpret_cast<InferenceContext*>(ctx)->Vector(size);
  return reinterpret_cast<TF_ShapeHandle*>(handle);
}

void TF_ShapeInferenceContextConcatenateShapes(TF_ShapeInferenceContext* ctx,
                                               TF_ShapeHandle* first,
                                               TF_ShapeHandle* second,
                                               TF_ShapeHandle* result,
                                               TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  Status s = cc_ctx->Concatenate(*reinterpret_cast<ShapeHandle*>(first),
                                 *reinterpret_cast<ShapeHandle*>(second),
                                 reinterpret_cast<ShapeHandle*>(result));
  Set_TF_Status_from_Status(status, s);
}

TF_DimensionHandle* TF_NewDimensionHandle() {
  return reinterpret_cast<TF_DimensionHandle*>(new DimensionHandle);
}

int64_t TF_ShapeInferenceContextNumInputs(TF_ShapeInferenceContext* ctx) {
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  return cc_ctx->num_inputs();
}

void TF_ShapeInferenceContextGetInput(TF_ShapeInferenceContext* ctx, int i,
                                      TF_ShapeHandle* handle,
                                      TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  if (0 < i || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "input index out of range");
  }
  if (TF_GetCode(status) == TF_OK) {
    auto* cc_result = reinterpret_cast<ShapeHandle*>(handle);
    *cc_result = cc_ctx->input(i);
  }
}

int TF_ShapeInferenceContextRankKnown(TF_ShapeInferenceContext* ctx,
                                      TF_ShapeHandle* handle) {
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  return cc_ctx->RankKnown(*reinterpret_cast<ShapeHandle*>(handle));
}

void TF_ShapeInferenceContextSetOutput(TF_ShapeInferenceContext* ctx, int i,
                                       TF_ShapeHandle* handle,
                                       TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  if (0 < i || i >= cc_ctx->num_outputs()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "output index out of range");
  }
  if (TF_GetCode(status) == TF_OK) {
    cc_ctx->set_output(i, *(reinterpret_cast<ShapeHandle*>(handle)));
  }
}

void TF_DeleteShapeHandle(TF_ShapeHandle* handle) {
  if (handle == nullptr) {
    return;
  }

  delete reinterpret_cast<ShapeHandle*>(handle);
}

void TF_DeleteDimensionHandle(TF_DimensionHandle* handle) {
  if (handle == nullptr) {
    return;
  }

  delete reinterpret_cast<DimensionHandle*>(handle);
}

#define DEFINE_TF_GETATTR(func, c_type, cc_type)                         \
  void TF_ShapeInferenceContext_GetAttr##func(                           \
      TF_ShapeInferenceContext* ctx, const char* attr_name, c_type* val, \
      TF_Status* status) {                                               \
    TF_SetStatus(status, TF_OK, "");                                     \
    cc_type v;                                                           \
    auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);             \
    Status s = cc_ctx->GetAttr(attr_name, &v);                           \
    Set_TF_Status_from_Status(status, s);                                \
    if (s.ok()) {                                                        \
      *val = static_cast<c_type>(v);                                     \
    }                                                                    \
  }

DEFINE_TF_GETATTR(Type, TF_DataType, tensorflow::DataType)

#define DEFINE_RANK_FUNC(func_name)                                        \
  void TF_ShapeInferenceContext##func_name(                                \
      TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle, int64_t rank, \
      TF_ShapeHandle* result, TF_Status* status) {                         \
    auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);               \
    auto* cc_handle = reinterpret_cast<ShapeHandle*>(handle);              \
    auto* cc_result = reinterpret_cast<ShapeHandle*>(result);              \
    Status s = cc_ctx->func_name(*cc_handle, rank, cc_result);             \
    Set_TF_Status_from_Status(status, s);                                  \
  }

DEFINE_RANK_FUNC(WithRank)
DEFINE_RANK_FUNC(WithRankAtLeast)
DEFINE_RANK_FUNC(WithRankAtMost)

int64_t TF_ShapeInferenceContextRank(TF_ShapeInferenceContext* ctx,
                                     TF_ShapeHandle* handle) {
  return reinterpret_cast<InferenceContext*>(ctx)->Rank(
      *reinterpret_cast<ShapeHandle*>(handle));
}

void TF_ShapeInferenceContextDim(TF_ShapeInferenceContext* ctx,
                                 TF_ShapeHandle* shape_handle, int64_t i,
                                 TF_DimensionHandle* result) {
  int64_t rank = TF_ShapeInferenceContextRank(ctx, shape_handle);
  auto* cc_result = reinterpret_cast<DimensionHandle*>(result);

  if (i < -rank || i >= rank) {
    *cc_result = DimensionHandle();
    return;
  }

  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  auto* cc_shape_handle = reinterpret_cast<ShapeHandle*>(shape_handle);
  *cc_result = cc_ctx->Dim(*cc_shape_handle, i);
}

int TF_DimensionHandleValueKnown(TF_DimensionHandle* dim_handle) {
  return InferenceContext::ValueKnown(
      *reinterpret_cast<DimensionHandle*>(dim_handle));
}

void TF_ShapeInferenceContextSetUnknownShape(TF_ShapeInferenceContext* ctx,
                                             TF_Status* status) {
  Status s = ::tensorflow::shape_inference::UnknownShape(
      reinterpret_cast<InferenceContext*>(ctx));
  Set_TF_Status_from_Status(status, s);
}

void TF_ShapeInferenceContextSubshape(TF_ShapeInferenceContext* ctx,
                                      TF_ShapeHandle* shape_handle,
                                      int64_t start, int64_t end,
                                      TF_ShapeHandle* result,
                                      TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  auto* cc_result = reinterpret_cast<ShapeHandle*>(result);
  Status s = cc_ctx->Subshape(*reinterpret_cast<ShapeHandle*>(shape_handle),
                              start, end, cc_result);
  Set_TF_Status_from_Status(status, s);
}

int64_t TF_DimensionHandleValue(TF_DimensionHandle* dim_handle) {
  return InferenceContext::Value(
      *reinterpret_cast<DimensionHandle*>(dim_handle));
}
