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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OPTIONS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OPTIONS_H_

#include "tensorflow/lite/delegates/gpu/gl/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// Default constructor for options turns on all optimizations.
struct CompilationOptions {
  // Allows to quantify tensors, downcast values, process in float16 etc.
  bool allow_precision_loss = false;

  // When set few operations are fused into a single shader. Therefore, there
  // will be less shaders, but each shader will become larger.
  bool fuse_operations = true;

  // Parameters will be inlined into a shader. This in turn will generated more
  // unique shaders where each will need to be compiled.
  bool inline_parameters = false;

  // If true, shaders, that have auto-input and auto-output, will use a single
  // object for reading and writing.
  bool inline_objects = true;  // TODO(akulik): unsupported

  // Can be only Textures or Buffers
  ObjectType preferred_obj_type = ObjectType::UNKNOWN;
  // User has an option to choose between textures and buffers. Textures work
  // better on Adreno and buffers are better for Mali.

  // Chooses object type to represent intermediate tensors. Buffers have more
  // efficient memory usage because they represent opaque memory blob, but
  // textures work better on Adreno.
  // TODO(akulik): may be better name?
  ObjectType ref_obj_type = ObjectType::UNKNOWN;

  // If true, a user may change BATCH dimension at runtime. Otherwise, static
  // batch size will be fixed during compile time.
  // Dynamic mode uses less memory, while static mode may yield better
  // performance for small models.
  bool dynamic_batch = false;

  // Fuses consequent nodes which have auto output and auto input.
  bool auto_input_fusion = true;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OPTIONS_H_
