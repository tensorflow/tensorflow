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
#ifndef TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_H_
#define TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_H_

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Unified Execution APIs for Eager and tracing backends.
// =============================================================================

// -----------------------------------------------------------------------------
// Core APIs
// -----------------------------------------------------------------------------

// A TF_ExecutionContext stores knowledge about how to execute an operation.
// E.g. it could know whether we're in eager mode or in graph mode, keeps track
// of gradient tapes, etc.
typedef struct TF_ExecutionContext TF_ExecutionContext;

// A TF_AbstractTensor is an input to an operation. E.g. it could be a union
// type of eager and graph tensors. It is also the result of executing an
// operation.
typedef struct TF_AbstractTensor TF_AbstractTensor;

// A TF_AbstractOp is the metadata we need to execute an operation. E.g. this
// could contain the op type and other attributes.
typedef struct TF_AbstractOp TF_AbstractOp;

// Stores a function representation that can be used for execution or for
// setting functional attributes of other composite ops e.g. control flow.
typedef struct TF_AbstractFunction TF_AbstractFunction;

// This allows the client to swap the implementation of the tracing engine.
// Any future call to TF_CreateFunction will use the implementation defined
// here.
void TF_SetTracingImplementation(const char* name);

// Creates a new TensorFlow function. A Function is an execution context, and as
// such it can trace operations through TF_ExecuteOperation. After completing
// tracing, a function can be obtained by TF_FinalizeFunction.
TF_ExecutionContext* TF_CreateFunction(const char* fn_name, TF_Status* status);

// Creates a context for eager execution of operations.
TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions*,
                                                 TF_Status* s);
void TF_DeleteExecutionContext(TF_ExecutionContext*);

// Add a new parameter to a TensorFlow Function.
// TODO(aminim): what about shape?
TF_AbstractTensor* TF_AddFunctionParameter(TF_ExecutionContext* func,
                                           TF_DataType dtype, TF_Status* s);

// Create an operation suitable to use with the provided context. The operation
// requires its type (e.g. "AddV2") to be set independently.
TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* ctx);
void TF_DeleteAbstractOp(TF_AbstractOp*);

// TODO(srbs): Add APIs for specifying attrs etc.
// `op_type` must outlive `op`.
void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s);
// `op_name` must outlive `op`.
void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s);
// `attr_name` must outlive `op`.
void TF_AbstractOpSetAttrType(TF_AbstractOp* op, const char* const attr_name,
                              TF_DataType value, TF_Status* s);

void TF_DeleteAbstractTensor(TF_AbstractTensor*);

// TF_OutputList holds the list of TF_AbstractTensor that results from executing
// an operation.
// It just lets us not specify the number of outputs of an operation
// beforehand. This forces a memory allocation in the runtime, which is bad, but
// it allows for generic code.
// TODO(aminim): the description above isn't clear with respect to
// TF_OutputListNumOutputs and the current eager implementation which requires
// the number of outputs to be set by the client.
typedef struct TF_OutputList TF_OutputList;
TF_OutputList* TF_NewOutputList();
void TF_DeleteOutputList(TF_OutputList* o);
void TF_OutputListSetNumOutputs(TF_OutputList* o, int, TF_Status*);
int TF_OutputListNumOutputs(TF_OutputList* o);
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i);

// TF_ExecuteOperation will, if in eager mode, execute, if in graph mode, maybe
// capture some inputs and then add a node in the graph. The output tensors are
// returned through the provided TF_OutputList.
// Any active tape will observe the effects of this execution.
void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s);

// Creates a new TF_AbstractFunction from the current tracing states in the
// context. The provided `ctx` is consumed by this API call and deleted.
// The returned TF_AbstractFunction must be deleted by the client,
// TODO(aminim): clarify the contract on the state of the context after this
// call.
TF_AbstractFunction* TF_FinalizeFunction(TF_ExecutionContext* ctx,
                                         TF_OutputList*, TF_Status*);

void TF_DeleteAbstractFunction(TF_AbstractFunction*);

// Register the function with the given context. This is particularly useful for
// making a function available to an eager context.
void TF_ExecutionContextRegisterFunction(TF_ExecutionContext*,
                                         TF_AbstractFunction*, TF_Status*);

// -----------------------------------------------------------------------------
// APIs specific to Eager modes
// -----------------------------------------------------------------------------

// Temporary APIs till we figure out how to create scalar valued Eager
// tensors and how to get value out of eager abstract tensors.
TF_AbstractTensor* TF_CreateAbstractTensorFromEagerTensor(TFE_TensorHandle* t,
                                                          TF_Status* s);
TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s);
TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext*);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_H_
