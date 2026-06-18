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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_DELEGATE_H_

#include <GLES3/gl31.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
//
// GPU delegate declared in this file is OBSOLETE and replaced with the delegate
// declared in delegate.h. New delegate combines all GL, CL and soon
// Vulkan-based implementations in one.
// Please migrate before end of 2019.
//
// WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING

// LINT.IfChange
enum TfLiteGlObjectType {
  TFLITE_GL_OBJECT_TYPE_FASTEST = 0,
  TFLITE_GL_OBJECT_TYPE_TEXTURE = 1,
  TFLITE_GL_OBJECT_TYPE_BUFFER = 2,
};

// Shader compilation options.
// Always use TfLiteGlCompileOptionsDefault() method to create new instance
// of TfLiteGlCompileOptions, otherwise every new added option may break
// inference.
// TODO(impjdi): Unify with opengl::CompilationOptions.
typedef struct {
  // When set to zero, computations are carried out in 32-bit floating point.
  // Otherwise, the GPU may quantify tensors, downcast values, process in FP16
  // (recommended).
  int32_t precision_loss_allowed;

  // User's preferred GL object to represent tensors.  When set to:
  // * `TFLITE_GL_OBJECT_TYPE_FASTEST`, the delegate chooses a GL object type
  //   automatically that will perform fastest (recommended).
  // * `TFLITE_GL_OBJECT_TYPE_TEXTURE`: GL textures are used to represent
  //   tensors which often work faster on Adreno-based devices, but may use more
  //   memory.
  // * `TFLITE_GL_OBJECT_TYPE_BUFFER`: GL shader storage buffer objects are used
  //   to represent tensors.
  int32_t preferred_gl_object_type;

  // When set to zero, dynamic batching is disabled and input/output tensors
  // must have a batch size of 1 (probably what you unless you use LSTMs).
  // Otherwise, enables dynamic batching and input/output tensor can have a
  // batch size greater than 1.
  int32_t dynamic_batch_enabled;

  // Parameters will be inlined into a shader. This in turn will generated more
  // unique shaders where each will need to be compiled.
  int32_t inline_parameters;
} TfLiteGlCompileOptions;

// Populates TfLiteGlCompileOptions as follows:
//   precision_loss_allowed = 0;
//   preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST;
//   dynamic_batch_enabled = 0;
//   inline_parameters = 0;
TFL_CAPI_EXPORT TfLiteGlCompileOptions TfLiteGlCompileOptionsDefault();

// Always use TfLiteGpuDelegateOptionsDefault() method to create new instance
// of TfLiteGpuDelegateOptions, otherwise every new added option may break
// inference.
typedef struct {
  const uint8_t* metadata;  // Internal.
  TfLiteGlCompileOptions compile_options;
} TfLiteGpuDelegateOptions;

// Populates TfLiteGlCompileOptions as follows:
//   metadata = nullptr;
//   compile_options = TfLiteGlCompileOptionsDefault();
TFL_CAPI_EXPORT TfLiteGpuDelegateOptions TfLiteGpuDelegateOptionsDefault();

// LINT.ThenChange(//tensorflow/lite/delegates/gpu/java/src/main/java/org/tensorflow/lite/gpu/GpuDelegate.java)

// Creates a new delegate instance that need to be destroyed with
// TfLiteGpuDelegateDelete when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
// .metadata = nullptr,
// .compile_options = {
//   .precision_loss_allowed = false,
//   .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
//   .dynamic_batch_enabled = false,
// },
ABSL_DEPRECATED("Use TfLiteGpuDelegateV2Create defined in delegate.h instead.")
TFL_CAPI_EXPORT TfLiteDelegate* TfLiteGpuDelegateCreate(
    const TfLiteGpuDelegateOptions* options);

// Destroys a delegate created with `TfLiteGpuDelegateCreate` call.
TFL_CAPI_EXPORT void TfLiteGpuDelegateDelete(TfLiteDelegate* delegate);

// Binds GL shader storage object to an input or an output tensor in the
// initialized delegate.  Bound buffer should have sufficient storage to
// accommodate all elements of a tensor.
//
// *** Must be called *before* `Interpreter::ModifyGraphWithDelegate`. ***
TFL_CAPI_EXPORT TfLiteStatus TfLiteGpuDelegateBindBufferToTensor(
    TfLiteDelegate* delegate, GLuint buffer, int tensor_index);

#ifndef TFLITE_GPU_BINARY_RELEASE
// Returns the metadata of `tflite_model` if it has one, or `nullptr` otherwise.
// Designed to be used with `TfLiteGpuDelegateOptions.metadata`.
TFL_CAPI_EXPORT const uint8_t* TfLiteGpuDelegateGetModelMetadata(
    const void* tflite_model);
#endif  // TFLITE_GPU_BINARY_RELEASE

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_DELEGATE_H_
