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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_PROGRAM_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_PROGRAM_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

// A wrapper around opengl program id that needs to be recycled when not needed.
// Encapsulates logic needed to bind parameters, link a program and execute it.
class GlProgram {
 public:
  // Creates invalid program.
  GlProgram() : id_(0) {}

  // Creates new program, initializes it, attaches the given shader and links
  // a program. Thus, if this call returns a program, one may set parameters and
  // finally execute a program.
  // therefore it needs to be handled elsewhere.
  static Status CreateWithShader(const GlShader& shader, GlProgram* gl_program);

  // Same as CreateWithShader but takes compiled shader in a binary form,
  // therefore compilation step is avoided.
  static Status CreateWithBinaryShader(const BinaryShader& shader,
                                       GlProgram* gl_program);

  // move-only
  GlProgram(GlProgram&& program);
  GlProgram& operator=(GlProgram&& program);
  GlProgram(const GlProgram&) = delete;
  GlProgram& operator=(const GlProgram&) = delete;

  ~GlProgram();

  GLuint id() const { return id_; }

  // Returns a binary representation for a shader currently attached and linked
  // into this program.
  Status GetBinary(BinaryShader* binary_shader);

  Status SetParameter(const Variable& param);

  // Executes program
  Status Dispatch(const uint3& workgroups) const;

  bool is_valid() const { return id_ != 0; }

 private:
  explicit GlProgram(GLuint program_id) : id_(program_id) {}

  void Invalidate();

  GLint GetUniformId(const std::string& name);

  GLuint id_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_PROGRAM_H_
