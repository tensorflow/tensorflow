/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_C_API_H_
#define TENSORFLOW_C_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_core_api.h"
#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

// --------------------------------------------------------------------------
// Non-core C API for TensorFlow.
//
// This file contains the non-core C API for TensorFlow.  Most of the
// API documentation and functionality resides in c_core_api.h.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_WhileParams {
  // The number of inputs to the while loop, i.e. the number of loop variables.
  // This is the size of cond_inputs, body_inputs, and body_outputs.
  const int ninputs;

  // The while condition graph. The inputs are the current values of the loop
  // variables. The output should be a scalar boolean.
  TF_Graph* const cond_graph;
  const TF_Output* const cond_inputs;
  TF_Output cond_output;

  // The loop body graph. The inputs are the current values of the loop
  // variables. The outputs are the updated values of the loop variables.
  TF_Graph* const body_graph;
  const TF_Output* const body_inputs;
  TF_Output* const body_outputs;

  // Unique null-terminated name for this while loop. This is used as a prefix
  // for created operations.
  const char* name;
} TF_WhileParams;

// Creates a TF_WhileParams for creating a while loop in `g`. `inputs` are
// outputs that already exist in `g` used as initial values for the loop
// variables.
//
// The returned TF_WhileParams will have all fields initialized except
// `cond_output`, `body_outputs`, and `name`. The `body_outputs` buffer will be
// allocated to size `ninputs`. The caller should build `cond_graph` and
// `body_graph` starting from the inputs, and store the final outputs in
// `cond_output` and `body_outputs`.
//
// If `status` is OK, the caller must call either TF_FinishWhile or
// TF_AbortWhile on the returned TF_WhileParams. If `status` isn't OK, the
// returned TF_WhileParams is not valid, and the caller should not call
// TF_FinishWhile() or TF_AbortWhile().
//
// Missing functionality (TODO):
// - Gradients
// - Reference-type inputs
// - Directly referencing external tensors from the cond/body graphs (this is
//   possible in the Python API)
TF_CAPI_EXPORT extern TF_WhileParams TF_NewWhile(TF_Graph* g, TF_Output* inputs,
                                                 int ninputs,
                                                 TF_Status* status);

// Builds the while loop specified by `params` and returns the output tensors of
// the while loop in `outputs`. `outputs` should be allocated to size
// `params.ninputs`.
//
// `params` is no longer valid once this returns.
//
// Either this or TF_AbortWhile() must be called after a successful
// TF_NewWhile() call.
TF_CAPI_EXPORT extern void TF_FinishWhile(const TF_WhileParams* params,
                                          TF_Status* status,
                                          TF_Output* outputs);

// Frees `params`s resources without building a while loop. `params` is no
// longer valid after this returns. Either this or TF_FinishWhile() must be
// called after a successful TF_NewWhile() call.
TF_CAPI_EXPORT extern void TF_AbortWhile(const TF_WhileParams* params);

// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
//
// `dx` are used as initial gradients (which represent the symbolic partial
// derivatives of some loss function `L` w.r.t. `y`).
// `dx` must be nullptr or have size `ny`.
// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
// shapes in `y`.
// The partial derivatives are returned in `dy`. `dy` should be allocated to
// size `nx`.
//
// Gradient nodes are automatically named under the "gradients/" prefix. To
// guarantee name uniqueness, subsequent calls to the same graph will
// append an incremental tag to the prefix: "gradients_1/", "gradients_2/", ...
// See TF_AddGradientsWithPrefix, which provides a means to specify a custom
// name prefix for operations added to a graph to compute the gradients.
//
// WARNING: This function does not yet support all the gradients that python
// supports. See
// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
// for instructions on how to add C++ more gradients.
TF_CAPI_EXPORT void TF_AddGradients(TF_Graph* g, TF_Output* y, int ny,
                                    TF_Output* x, int nx, TF_Output* dx,
                                    TF_Status* status, TF_Output* dy);

// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
// This is a variant of TF_AddGradients that allows to caller to pass a custom
// name prefix to the operations added to a graph to compute the gradients.
//
// `dx` are used as initial gradients (which represent the symbolic partial
// derivatives of some loss function `L` w.r.t. `y`).
// `dx` must be nullptr or have size `ny`.
// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
// shapes in `y`.
// The partial derivatives are returned in `dy`. `dy` should be allocated to
// size `nx`.
// `prefix` names the scope into which all gradients operations are being added.
// `prefix` must be unique within the provided graph otherwise this operation
// will fail. If `prefix` is nullptr, the default prefixing behaviour takes
// place, see TF_AddGradients for more details.
//
// WARNING: This function does not yet support all the gradients that python
// supports. See
// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
// for instructions on how to add C++ more gradients.
TF_CAPI_EXPORT void TF_AddGradientsWithPrefix(TF_Graph* g, const char* prefix,
                                              TF_Output* y, int ny,
                                              TF_Output* x, int nx,
                                              TF_Output* dx, TF_Status* status,
                                              TF_Output* dy);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_H_
