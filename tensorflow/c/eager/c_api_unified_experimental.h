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

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

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

TF_ExecutionContext* TF_NewExecutionContext();
void TF_DeleteExecutionContext(TF_ExecutionContext*);

TF_AbstractOp* TF_NewAbstractOp();
void TF_DeleteAbstractOp(TF_AbstractOp*);

TF_AbstractTensor* TF_NewAbstractTensor();
void TF_DeleteAbstractTensor(TF_AbstractTensor*);

// -----------------------------------------------------------------------------
// APIs for Eager and graph modes
// -----------------------------------------------------------------------------

// Keeps track of the current graph and other state e.g. captures etc.
typedef struct TF_GraphContext TF_GraphContext;
TF_GraphContext* TF_NewGraphContext(TF_Graph*);
void TF_DeleteGraphContext(TF_GraphContext*);

// `eager_context` must outlive `context`.
void TF_ExecutionContextSetEagerContext(TF_ExecutionContext* context,
                                        TFE_Context* eager_context, TF_Status*);
// `graph_context` must outlive `context`.
void TF_ExecutionContextSetGraphContext(TF_ExecutionContext* context,
                                        TF_GraphContext* graph_context,
                                        TF_Status*);

// TODO(srbs): Add APIs for specifying attrs etc.
// `op_type` must outlive `op`.
void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s);
// `op_name` must outlive `op`.
void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s);

// Wrapper for TF_Output but contains a pointer to TF_GraphContext as well.
typedef struct TF_GraphTensor TF_GraphTensor;
TF_GraphTensor* TF_NewGraphTensor(TF_GraphContext* c, TF_Output t,
                                  TF_Status* s);
TF_Output TF_GraphTensorToOutput(const TF_GraphTensor* const t, TF_Status* s);
void TF_DeleteGraphTensor(TF_GraphTensor* t);

// `t` must outlive `at`.
void TF_AbstractTensorSetEagerTensor(TF_AbstractTensor* at, TFE_TensorHandle* t,
                                     TF_Status* s);
TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s);

// `t` must outlive `at`.
void TF_AbstractTensorSetGraphTensor(TF_AbstractTensor* at, TF_GraphTensor* t,
                                     TF_Status* s);
TF_GraphTensor* TF_AbstractTensorGetGraphTensor(TF_AbstractTensor* at,
                                                TF_Status* s);

// TF_OutputList just lets us not specify the number of outputs of an operation
// beforehand. This forces a memory allocation in the runtime, which is bad, but
// it allows for generic code.
typedef struct TF_OutputList TF_OutputList;
TF_OutputList* TF_NewOutputList();
void TF_DeleteOutputList(TF_OutputList* o);
void TF_OutputListSetNumOutputs(TF_OutputList* o, int, TF_Status*);
int TF_OutputListNumOutputs(TF_OutputList* o);
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i);

// TF_ExecuteOperation will, if in eager mode, execute, if in graph mode, maybe
// capture some inputs and then add a node in the graph, and after
// execution/node creation it'll go and record things that happened in any tape
// which happens to be active.
void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_H_
