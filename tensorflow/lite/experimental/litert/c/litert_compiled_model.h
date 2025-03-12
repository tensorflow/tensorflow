// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILED_MODEL_H_

#include <stddef.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The LiteRtCompiledModel is a higher level inference API. It is created by
// provided model with compilation options. Internally, it instantiates runtime
// and applies Delegates mapped to the compilation options.
// It also supports getting LiteRtTensorBufferRequirements to create
// input/output TensorBuffers, and it allows to invoke the model with the
// input/output TensorBuffers.
//
// Example user flow:
//
// 1. Create LiteRtCompiledModel
// 2. Query the model input/output LiteRtTensorBufferRequirements
// 3. Create input/output LiteRtTensorBuffer
// 4. Fill the input LiteRtTensorBuffer with input data
// 5. Invoke the model with the input/output LiteRtTensorBuffer
// 6. Evaluate the output LiteRtTensorBuffer

LITERT_DEFINE_HANDLE(LiteRtCompiledModel);

// Creates a LiteRtCompiledModel from a LiteRtModel object. Parameter
// `jit_compilation_options` is optional and can be null, and is owned by the
// caller.  The model is loaded into memory and the caller takes ownership of
// the returned object.
LiteRtStatus LiteRtCreateCompiledModel(
    LiteRtEnvironment environment, LiteRtModel model,
    LiteRtCompilationOptions compilation_options,
    LiteRtCompiledModel* compiled_model);

// Returns the buffer requirements for the given n-th input tensor. The returned
// LiteRtTensorBufferRequirements is used to create the input tensor
// buffer.
//
// Parameters:
// - compiled_model: the target `LiteRtCompiledModel` object.
// - signature_index: the index of the signature in `LiteRtModel`.
// - input_index: the index of the input tensor in the signature (subgraph).
// - buffer_requirements: the returned `LiteRtTensorBufferRequirements`.
LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index,
    LiteRtTensorBufferRequirements* buffer_requirements);

// Returns the buffer requirements for the given n-th output tensor. The
// returned LiteRtTensorBufferRequirements is used to create the output tensor
// buffer.
//
// Parameters:
// - compiled_model: the target `LiteRtCompiledModel` object.
// - signature_index: the index of the signature in `LiteRtModel`.
// - input_index: the index of the input tensor in the signature (subgraph).
// - buffer_requirements: the returned `LiteRtTensorBufferRequirements`.
LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements);

// Runs the model of the given signature synchronously, with the provided
// input/output LiteRtTensorBuffer.
//
// Parameters:
// - compiled_model: the target `LiteRtCompiledModel` object.
// - signature_index: the index of the signature in `LiteRtModel`.
// - num_input_buffers: the number of input `LiteRtTensorBuffer`.
// - input_buffers: the array of input `LiteRtTensorBuffer`.
// - num_output_buffers: the number of output `LiteRtTensorBuffer`.
// - output_buffers: the array of output LiteRtTensorBuffer.
LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel compiled_model,
                                    LiteRtParamIndex signature_index,
                                    size_t num_input_buffers,
                                    LiteRtTensorBuffer* input_buffers,
                                    size_t num_output_buffers,
                                    LiteRtTensorBuffer* output_buffers);

// Runs the model of the given signature asynchronously, if possible, with the
// provided input/output LiteRtTensorBuffers. If asynchronous execution is
// possible, then the function sets parameter `async` to true; if asynchronous
// execution is not possible, then the function runs the model synchronously and
// sets parameter `async` to false. Note that:
//
// - Asynchronous execution is possible only in certain cases, based on the ops
//   included in the model, the selected HW accelerator(s), and the capability
//   of the user device hardware.
//
// - If asynchronous execution is indeed possible, it may be that only some
//   parts of the model are run asynchronously (e.g., ops mapped to the GPU)
//   while other parts of the model are still run synchronously with the
//   invocation of this call (e.g., ops mapped to the CPU).
//
// - In case of asynchronous execution some or all of the output tensor buffers
//   will have a synchronization event attached to them and the caller is
//   responsible for passing such events to a downstream processing step.
//
// Parameters:
// - async: optional boolean to let the caller know if the model is being run
//   asynchronously.
LiteRtStatus LiteRtRunCompiledModelAsync(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    size_t num_input_buffers, LiteRtTensorBuffer* input_buffers,
    size_t num_output_buffers, LiteRtTensorBuffer* output_buffers, bool* async);

void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled_model);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILED_MODEL_H_
