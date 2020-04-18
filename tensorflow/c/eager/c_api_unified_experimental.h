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
// type of eager and graph tensors.
typedef struct TF_AbstractTensor TF_AbstractTensor;
// A TF_AbstractOp is the metadata we need to execute an operation. E.g. this
// could contain the op type and other attributes.
typedef struct TF_AbstractOp TF_AbstractOp;

// `TF_ExecutionContextOptions` define what type of `TF_ExecutionContext` is
// created. It can be used to pass context specific params.
typedef struct TF_ExecutionContextOptions TF_ExecutionContextOptions;
void TF_DeleteExecutionContextOptions(TF_ExecutionContextOptions*);

TF_ExecutionContext* TF_NewExecutionContext(TF_ExecutionContextOptions*,
                                            TF_Status* s);
void TF_DeleteExecutionContext(TF_ExecutionContext*);

TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* ctx);
void TF_DeleteAbstractOp(TF_AbstractOp*);

void TF_DeleteAbstractTensor(TF_AbstractTensor*);
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

// TF_OutputList just lets us not specify the number of outputs of an operation
// beforehand. This forces a memory allocation in the runtime, which is bad, but
// it allows for generic code.
typedef struct TF_OutputList TF_OutputList;
TF_OutputList* TF_NewOutputList();
void TF_DeleteOutputList(TF_OutputList* o);
void TF_OutputListSetNumOutputs(TF_OutputList* o, int, TF_Status*);
int TF_OutputListNumOutputs(TF_OutputList* o);
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i);

// Stores a function representation that can be used for execution or for
// setting functional attributes of other composite ops e.g. control flow.
typedef struct TF_AbstractFunction TF_AbstractFunction;
TF_AbstractFunction* TF_ExecutionContextToFunction(
    const TF_ExecutionContext* fn_body, const char* fn_name, int num_inputs,
    const TF_AbstractTensor* inputs, int num_outputs,
    const TF_AbstractTensor* outputs, TF_Status* status);
void TF_DeleteAbstractFunction(TF_AbstractFunction*);
void TF_ExecutionContextRegisterFunction(TF_ExecutionContext*,
                                         TF_AbstractFunction*, TF_Status*);

// TF_ExecuteOperation will, if in eager mode, execute, if in graph mode, maybe
// capture some inputs and then add a node in the graph, and after
// execution/node creation it'll go and record things that happened in any tape
// which happens to be active.
void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s);

// -----------------------------------------------------------------------------
// APIs specific to Eager and graph modes
// -----------------------------------------------------------------------------

TF_ExecutionContextOptions* TF_NewGraphContextOptions();
TF_ExecutionContextOptions* TF_NewEagerContextOptions(TFE_ContextOptions*);

// Temporary APIs till we figure out how to create scalar valued Eager
// tensors and how to get value out of eager abstract tensors.
TF_AbstractTensor* TF_NewAbstractTensor();
void TF_AbstractTensorSetEagerTensor(
    TF_AbstractTensor* at, TFE_TensorHandle* t,
    TF_Status* s);  // `at` takes ownership of `t`.
TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s);
TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext*);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_H_
