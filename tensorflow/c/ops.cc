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

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::DataType;
using ::tensorflow::OpDef;
using ::tensorflow::OpDeprecation;
using ::tensorflow::OpShapeInferenceFn;
using ::tensorflow::Set_TF_Status_from_Status;
using ::tensorflow::Status;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

typedef struct TF_OpDefinitionBuilder {
  // The op definition proto representing the op.
  tensorflow::OpDef op_def;

  // The shape inference function, or nullptr if none is provided for this op.
  OpShapeInferenceFn shape_inference_func;
} TF_OpDefinitionBuilder;

TF_OpDefinitionBuilder* TF_NewOpDefinitionBuilder(const char* op_name) {
  auto* result = new TF_OpDefinitionBuilder;
  result->op_def.set_name(op_name);
  return result;
}

void TF_DeleteOpDefinitionBuilder(TF_OpDefinitionBuilder* builder) {
  delete builder;
}

static void PopulateArg(OpDef::ArgDef* arg, const char* name,
                        TF_DataType type) {
  arg->set_name(name);
  arg->set_type(static_cast<DataType>(type));
}

void TF_OpDefinitionBuilderAddInput(TF_OpDefinitionBuilder* builder,
                                    const char* name, TF_DataType type) {
  PopulateArg(builder->op_def.add_input_arg(), name, type);
}

void TF_OpDefinitionBuilderAddOutput(TF_OpDefinitionBuilder* builder,
                                     const char* name, TF_DataType type) {
  PopulateArg(builder->op_def.add_output_arg(), name, type);
}

#define DEFINE_BUILDER_BOOL_SETTER(func_name, builder_setter_name, arg_name) \
  void TF_OpDefinitionBuilder##func_name(TF_OpDefinitionBuilder* builder,    \
                                         bool arg_name) {                    \
    builder->op_def.builder_setter_name(arg_name);                           \
  }

DEFINE_BUILDER_BOOL_SETTER(SetIsCommutative, set_is_commutative, is_commutative)
DEFINE_BUILDER_BOOL_SETTER(SetIsAggregate, set_is_aggregate, is_aggregate)
DEFINE_BUILDER_BOOL_SETTER(SetIsStateful, set_is_stateful, is_stateful)
DEFINE_BUILDER_BOOL_SETTER(SetAllowsUninitializedInput,
                           set_allows_uninitialized_input,
                           allows_unintialized_input)

static OpDef::AttrDef* AddAttribute(TF_OpDefinitionBuilder* builder,
                                    const char* name, const char* type_name) {
  OpDef::AttrDef* attr = builder->op_def.add_attr();
  attr->set_name(name);
  attr->set_type(type_name);
  return attr;
}

#define DEFINE_ATTR_SETTER(attr_type, type_name, field_c_type, field_name)     \
  void TF_OpDefinitionBuilderAdd##attr_type##Attr(                             \
      TF_OpDefinitionBuilder* builder, const char* name) {                     \
    AddAttribute(builder, name, type_name);                                    \
  }                                                                            \
                                                                               \
  void TF_OpDefinitionBuilderAdd##attr_type##AttrWithDefaultValue(             \
      TF_OpDefinitionBuilder* builder, const char* name,                       \
      field_c_type field_name) {                                               \
    OpDef::AttrDef* attr = AddAttribute(builder, name, type_name);             \
    attr->mutable_default_value()->set_##field_name(field_name);               \
  }                                                                            \
                                                                               \
  void TF_OpDefinitionBuilderAdd##attr_type##ListAttrWithDefaultValues(        \
      TF_OpDefinitionBuilder* builder, const char* name,                       \
      field_c_type field_name[], size_t n) {                                   \
    OpDef::AttrDef* attr = AddAttribute(builder, name, "list(" type_name ")"); \
    for (int _i = 0; _i < n; ++_i) {                                           \
      attr->mutable_default_value()->mutable_list()->add_##field_name(         \
          field_name[_i]);                                                     \
    }                                                                          \
  }                                                                            \
                                                                               \
  void TF_OpDefinitionBuilderAdd##attr_type##ListAttr(                         \
      TF_OpDefinitionBuilder* builder, const char* name) {                     \
    TF_OpDefinitionBuilderAdd##attr_type##ListAttrWithDefaultValues(           \
        builder, name, NULL, 0);                                               \
  }

DEFINE_ATTR_SETTER(String, "string", const char*, s)
DEFINE_ATTR_SETTER(Int, "int", int64_t, i)
DEFINE_ATTR_SETTER(Float, "float", float, f)
DEFINE_ATTR_SETTER(Bool, "bool", bool, b)

void TF_OpDefinitionBuilderDeprecated(TF_OpDefinitionBuilder* builder,
                                      int version, const char* explanation) {
  OpDeprecation* dep = builder->op_def.mutable_deprecation();
  dep->set_version(version);
  dep->set_explanation(explanation);
}

void TF_RegisterOpDefinition(TF_OpDefinitionBuilder* builder,
                             TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::OpRegistry::Global()->Register(
      [builder](::tensorflow::OpRegistrationData* op_reg_data) -> Status {
        op_reg_data->op_def.Clear();
        op_reg_data->op_def.MergeFrom(builder->op_def);
        op_reg_data->shape_inference_fn = builder->shape_inference_func;
        return Status::OK();
      });

  // Calling ProcessRegistrations ensures that the cc_builder's finalize method
  // is called and that the builder can be deleted.
  Set_TF_Status_from_Status(
      status, ::tensorflow::OpRegistry::Global()->ProcessRegistrations());

  delete builder;
}

void TF_OpDefinitionBuilderSetShapeInferenceFunction(
    TF_OpDefinitionBuilder* builder,
    void (*shape_inference_func)(TF_ShapeInferenceContext* ctx,
                                 TF_Status* status)) {
  builder->shape_inference_func =
      [shape_inference_func](InferenceContext* ctx) -> tensorflow::Status {
    TF_Status* c_status = TF_NewStatus();
    auto c_ctx = reinterpret_cast<TF_ShapeInferenceContext*>(ctx);
    shape_inference_func(c_ctx, c_status);
    tensorflow::Status result = ::tensorflow::StatusFromTF_Status(c_status);
    TF_DeleteStatus(c_status);
    return result;
  };
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
