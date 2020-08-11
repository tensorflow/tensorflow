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

// Routines for registering new ops and for implementing op shape inference
// functions.
//
// This API is alpha software and is subject to change.
//
// REGISTRATION
// ------------
//
// In order to register a new op, create a new TF_OpDefinitionBuilder:
//
// TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("OpName");
//
// Inputs, outputs and attributes can be added to the builder with the
// corresponding functions, e.g.
//
// TF_OpDefinitionBuilderAddInput(builder, "input1: int32");
// TF_OpDefinitionBuilderAddOutput(builder, "output1: int64");
// TF_OpDefinitionBuilderAddAttr(builder, "attr: int32");
//
// The builder may then be registered with TensorFlow using the
// TF_RegisterOpDefinition function. E.g.
//
// TF_Status* status = TF_NewStatus();
// TF_RegisterOpDefinition(builder, &status);
// if (TF_GetCode(status) != TF_OK) {
//   // handle error
// }
//
// SHAPE INFERENCE
// ---------------
//
// You can provide a shape inference function that TensorFlow will call when it
// wants to understand the shape of outputs that the op will produce. Use the
// TF_OpDefinitionBuilderSetShapeInferenceFunction function to register a shape
// inference function pointer with TensorFlow. The following is an example of a
// very simple shape inference function:
//
// void identity_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
//   TF_ShapeHandle* input = TF_NewShapeHandle();
//   TF_ShapeInferenceContextGetInput(ctx, 0, input, status);
//   if (TF_GetCode(status) == TF_OK) {
//     TF_ShapeInferenceContextSetOutput(ctx, 0, input, status);
//   }
//   TF_DeleteShapeHandle(input);
// }
//
// The following code registers the inference function with TensorFlow:
//
// TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &identity_shape_fn);
//
// For more details about shape inference, see the documentation for
// TF_OpDefinitionBuilderSetShapeInferenceFunction.

#ifndef TENSORFLOW_C_OPS_H_
#define TENSORFLOW_C_OPS_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

struct TF_DimensionHandle;
struct TF_OpDefinitionBuilder;
struct TF_ShapeHandle;
struct TF_ShapeInferenceContext;

// Returns a newly allocated op definition builder for the given op name. The
// returned builder may be customized with the `TF_OpDefinitionBuilder...`
// functions and then registered with TensorFlow with TF_RegisterOpDefinition.
//
// The returned pointer is either freed by a call to TF_RegisterOpDefinition, or
// can be manually deleted by TF_DeleteOpDefinitionBuilder if it is never
// registered.
TF_CAPI_EXPORT extern TF_OpDefinitionBuilder* TF_NewOpDefinitionBuilder(
    const char* op_name);

// Registers the given op builder with TensorFlow. Indicates success or
// otherwise in the given status.
//
// `builder` is freed whether the op was successfully registered or not. You
// must call either this function or TF_DeleteOpDefinitionBuilder to free the
// builder, but never both.
TF_CAPI_EXPORT extern void TF_RegisterOpDefinition(
    TF_OpDefinitionBuilder* builder, TF_Status* status);

// Frees the given op definition builder. You must call either this function or
// TF_RegisterOpDefinition to free the builder, but never both.
TF_CAPI_EXPORT extern void TF_DeleteOpDefinitionBuilder(
    TF_OpDefinitionBuilder* builder);

//----------------------------------------------------
// Attribute functions.

// Adds an attr to the given TF_OpDefinitionBuilder. The spec has
// format "<name>:<type>" or "<name>:<type>=<default>"
// where <name> matches regexp [a-zA-Z][a-zA-Z0-9_]*.
// By convention, names containing only capital letters are reserved for
// attributes whose values can be inferred by the operator implementation if not
// supplied by the user. If the attribute name contains characters other than
// capital letters, the operator expects the user to provide the attribute value
// at operation runtime.
//
// <type> can be:
//   "string", "int", "float", "bool", "type", "shape", or "tensor"
//   "numbertype", "realnumbertype", "quantizedtype"
//       (meaning "type" with a restriction on valid values)
//   "{int32,int64}" or {realnumbertype,quantizedtype,string}"
//       (meaning "type" with a restriction containing unions of value types)
//   "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
//       (meaning "string" with a restriction on valid values)
//   "list(string)", ..., "list(tensor)", "list(numbertype)", ...
//       (meaning lists of the above types)
//   "int >= 2" (meaning "int" with a restriction on valid values)
//   "list(string) >= 2", "list(int) >= 2"
//       (meaning "list(string)" / "list(int)" with length at least 2)
// <default>, if included, should use the Proto text format
// of <type>.  For lists use [a, b, c] format.
//
// Note that any attr specifying the length of an input or output will
// get a default minimum of 1 unless the >= # syntax is used.
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderAddAttr(
    TF_OpDefinitionBuilder* builder, const char* attr_spec);

// Adds an input to this TF_OpDefinitionBuilder.
// The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
// where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
// * For a single tensor: <type>
// * For a sequence of tensors with the same type: <number>*<type>
// * For a sequence of tensors with different types: <type-list>
// Where:
//   <type> is either one of "float", "int32", "string", ...
//          or the name of an attr (see TF_OpDefinitionBuilderAddAttr)
//          with type "type".
//   <number> is the name of an attr with type "int".
//   <type-list> is the name of an attr with type "list(type)".
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderAddInput(
    TF_OpDefinitionBuilder* builder, const char* input_spec);

// Adds an output to this TF_OpDefinitionBuilder.
// The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
// where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
// * For a single tensor: <type>
// * For a sequence of tensors with the same type: <number>*<type>
// * For a sequence of tensors with different types: <type-list>
// Where:
//   <type> is either one of "float", "int32", "string", ...
//          or the name of an attr (see TF_OpDefinitionBuilderAddAttr)
//          with type "type".
//   <number> is the name of an attr with type "int".
//   <type-list> is the name of an attr with type "list(type)".
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderAddOutput(
    TF_OpDefinitionBuilder* builder, const char* output_spec);

// Sets the commutative property for the op built by the given builder.
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderSetIsCommutative(
    TF_OpDefinitionBuilder* builder, bool is_commutative);

// Sets the is_aggregate property of the builder to the given value.
//
// If is_aggregate is true, then the operation produced by this builder accepts
// N >= 2 inputs and produces 1 output all of the same type. Should be
// associative and commutative, and produce output with the same shape as the
// input. The optimizer may replace an aggregate op taking input from multiple
// devices with a tree of aggregate ops that aggregate locally within each
// device (and possibly within groups of nearby devices) before communicating.
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderSetIsAggregate(
    TF_OpDefinitionBuilder* builder, bool is_aggregate);

// Sets the is_stateful property of the builder to the given value.
//
// The op built by this builder is stateful if its behavior depends on some
// state beyond its input tensors (e.g. variable reading op) or if it has a
// side-effect (e.g. printing or asserting ops). Equivalently, stateless ops
// must always produce the same output for the same input and have no
// side-effects.
//
// By default Ops may be moved between devices. Stateful ops should either not
// be moved, or should only be moved if that state can also be moved (e.g. via
// some sort of save / restore). Stateful ops are guaranteed to never be
// optimized away by Common Subexpression Elimination (CSE).
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderSetIsStateful(
    TF_OpDefinitionBuilder* builder, bool is_stateful);

// Sets the allows_uninitialized_input property of the operation built by this
// builder.
//
// By default, all inputs to an Op must be initialized Tensors. Ops that may
// initialize tensors for the first time should set this field to true, to allow
// the Op to take an uninitialized Tensor as input.
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderSetAllowsUninitializedInput(
    TF_OpDefinitionBuilder* builder, bool allows_uninitialized_input);

// Adds a deprecation warning for the given op. This indicates to the user that
// `version` is the first TensorFlow GraphDef version for which the operation is
// deprecated. `explanation` should contain the reason for the deprecation and
// what to use instead.
//
// This function is only an indicator that the operation may disappear in a
// version of TensorFlow after `version`. It does not affect op registration.
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderDeprecated(
    TF_OpDefinitionBuilder* builder, int version, const char* explanation);

// Sets the shape inference function for the op.
TF_CAPI_EXPORT extern void TF_OpDefinitionBuilderSetShapeInferenceFunction(
    TF_OpDefinitionBuilder* builder,
    void (*shape_inference_func)(TF_ShapeInferenceContext* ctx,
                                 TF_Status* status));

//----------------------------------------------------
// Functions for TF_ShapeInferenceContext.
//
// Functions for implementing shape inference functions. TensorFlow uses these
// functions to determine the shape of tensors produced by an operation without
// having to actually run the operation. If an operation chooses to provide a
// shape inference function, it will be invoked by TensorFlow as needed.
//
// When invoked by TensorFlow, the shape inference function is provided with a
// TF_ShapeInferenceContext pointer. The function's implementation will use the
// accessor and mutator functions with names beginning with
// TF_ShapeInferenceContext to examine the input state and determine the output
// shape.

// Returns the number of inputs in the given shape inference context.
TF_CAPI_EXPORT extern int64_t TF_ShapeInferenceContextNumInputs(
    TF_ShapeInferenceContext* ctx);

// Returns a newly allocated shape handle. The shapes represented by these
// handles may be queried or mutated with the corresponding
// TF_ShapeInferenceContext...  functions.
TF_CAPI_EXPORT extern TF_ShapeHandle* TF_NewShapeHandle();

// Places the ith input of the given shape inference context into the given
// shape handle, or returns a status other than TF_OK indicating why the input
// could not be retrieved
// (for example, if i < 0 || i >= TF_ShapeInferenceContextNumInputs(ctx)).
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextGetInput(
    TF_ShapeInferenceContext* ctx, int i, TF_ShapeHandle* handle,
    TF_Status* status);

// Places the given shape handle into the `i`th output position of the given
// context. Internally, the shape handle is copied; the caller may subsequently
// delete `handle`.
TF_CAPI_EXPORT
extern void TF_ShapeInferenceContextSetOutput(TF_ShapeInferenceContext* ctx,
                                              int i, TF_ShapeHandle* handle,
                                              TF_Status* status);

// Returns a newly-allocated scalar shape handle. The returned handle should
// be freed with TF_DeleteShapeHandle.
TF_CAPI_EXPORT extern TF_ShapeHandle* TF_ShapeInferenceContextScalar(
    TF_ShapeInferenceContext* ctx);

// Returns a newly-allocate shape handle representing a vector of the given
// size. The returned handle should be freed with TF_DeleteShapeHandle.
TF_CAPI_EXPORT extern TF_ShapeHandle* TF_ShapeInferenceContextVectorFromSize(
    TF_ShapeInferenceContext* ctx, size_t size);

// Returns a newly allocated dimension handle. It must be freed with
// TF_DeleteDimensionHandle.
TF_CAPI_EXPORT extern TF_DimensionHandle* TF_NewDimensionHandle();

// Interprets the named shape inference context attribute as a TF_DataType and
// places it into *val. *status is set to TF_OK.
//
// If the attribute could not be found or could not be interpreted as
// TF_DataType, *status is populated with an error.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContext_GetAttrType(
    TF_ShapeInferenceContext* ctx, const char* attr_name, TF_DataType* val,
    TF_Status* status);

// Returns the rank of the shape represented by the given handle.
TF_CAPI_EXPORT extern int64_t TF_ShapeInferenceContextRank(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle);

// Returns 1 if `handle` has a known rank, 0 otherwise.
TF_CAPI_EXPORT extern int TF_ShapeInferenceContextRankKnown(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle);

// If <handle> has rank <rank>, or its rank is unknown, return OK and return the
// shape with asserted rank in <*result>. Otherwise an error is placed into
// `status`.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextWithRank(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle, int64_t rank,
    TF_ShapeHandle* result, TF_Status* status);

// If <handle> has rank at least <rank>, or its rank is unknown, return OK and
// return the shape with asserted rank in <*result>. Otherwise an error is
// placed into `status`.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextWithRankAtLeast(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle, int64_t rank,
    TF_ShapeHandle* result, TF_Status* status);

// If <handle> has rank at most <rank>, or its rank is unknown, return OK and
// return the shape with asserted rank in <*result>. Otherwise an error is
// placed into `status`.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextWithRankAtMost(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle, int64_t rank,
    TF_ShapeHandle* result, TF_Status* status);

// Places a handle to the ith dimension of the given shape into *result.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextDim(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* shape_handle, int64_t i,
    TF_DimensionHandle* result);

// Returns in <*result> a sub-shape of <shape_handle>, with dimensions
// [start:end]. <start> and <end> can be negative, to index from the end of the
// shape. <start> and <end> are set to the rank of <shape_handle> if > rank of
// <shape_handle>.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextSubshape(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* shape_handle, int64_t start,
    int64_t end, TF_ShapeHandle* result, TF_Status* status);

// Places an unknown shape in all outputs for the given inference context. Used
// for shape inference functions with ops whose output shapes are unknown.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextSetUnknownShape(
    TF_ShapeInferenceContext* ctx, TF_Status* status);

// Returns whether the given handle represents a known dimension.
TF_CAPI_EXPORT extern int TF_DimensionHandleValueKnown(
    TF_DimensionHandle* dim_handle);

// Returns the value of the given dimension.
TF_CAPI_EXPORT extern int64_t TF_DimensionHandleValue(
    TF_DimensionHandle* dim_handle);

// Returns in <*result> the result of appending the dimensions of <second> to
// those of <first>.
TF_CAPI_EXPORT extern void TF_ShapeInferenceContextConcatenateShapes(
    TF_ShapeInferenceContext* ctx, TF_ShapeHandle* first,
    TF_ShapeHandle* second, TF_ShapeHandle* result, TF_Status* status);

// Frees the given shape handle.
TF_CAPI_EXPORT extern void TF_DeleteShapeHandle(TF_ShapeHandle* handle);

// Frees the given dimension handle.
TF_CAPI_EXPORT extern void TF_DeleteDimensionHandle(TF_DimensionHandle* handle);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_OPS_H_
