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

#ifndef TENSORFLOW_C_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_C_C_API_EXPERIMENTAL_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api.h"

// --------------------------------------------------------------------------
// Experimental C API for TensorFlow.
//
// The API here is subject to changes in the future.
// --------------------------------------------------------------------------

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.$a
#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(COMPILER_MSVC)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // COMPILER_MSVC
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

// When `enable` is true, set
// tensorflow.ConfigProto.OptimizerOptions.global_jit_level to ON_1, and also
// set XLA flag values to prepare for XLA compilation. Otherwise set
// global_jit_level to OFF.
//
// This API is syntax sugar over TF_SetConfig(), and is used by clients that
// cannot read/write the tensorflow.ConfigProto proto.
TF_CAPI_EXPORT extern void TF_EnableXLACompilation(TF_SessionOptions* options,
                                                   unsigned char enable);

// Sets up TPU execution, by rewriting the graph accordingly, and initializing
// TPU system.
//
// On success, returns a shutdown node to be used in a subsequent
// TF_ShutdownTPUExecution(), and sets the new output nodes in
// `new_output_nodes` for caller to fetch from. Must be called exactly once
// before TF_SessionRun().
//
// The API and logic is modeled after the python counterparts
// tpu.{initialize_system(), rewrite(), shutdown_system()}.
//
// TODO(b/74774824): Create separate APIs for initializing TPU system and graph
// rewrite.
TF_CAPI_EXPORT extern TF_Output TF_SetupTPUExecution(
    TF_Session* session, int num_input_nodes, const TF_Output* input_nodes,
    int num_output_nodes, const TF_Output* output_nodes,
    TF_Output* new_output_nodes, TF_Status* status);

// Shuts down TPU system. For any `session` where TF_SetupTPUExecution() has
// been successfully called, this call must be made exactly once before the
// session is closed.
TF_CAPI_EXPORT extern void TF_ShutdownTPUExecution(TF_Session* session,
                                                   TF_Output shutdown_node,
                                                   TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_EXPERIMENTAL_H_
