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

#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "absl/strings/ascii.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

Status RequestGpuInfo(GpuInfo* gpu_info) {
  GpuInfo info;

  const GLubyte* renderer_name = glGetString(GL_RENDERER);
  if (renderer_name) {
    info.renderer_name = reinterpret_cast<const char*>(renderer_name);
    GetGpuModelAndType(info.renderer_name, &info.gpu_model, &info.type);
  }

  const GLubyte* vendor_name = glGetString(GL_VENDOR);
  if (vendor_name) {
    info.vendor_name = reinterpret_cast<const char*>(vendor_name);
  }

  const GLubyte* version_name = glGetString(GL_VERSION);
  if (version_name) {
    info.version = reinterpret_cast<const char*>(version_name);
  }

  glGetIntegerv(GL_MAJOR_VERSION, &info.major_version);
  glGetIntegerv(GL_MINOR_VERSION, &info.minor_version);

  GLint extensions_count;
  glGetIntegerv(GL_NUM_EXTENSIONS, &extensions_count);
  info.extensions.resize(extensions_count);
  for (int i = 0; i < extensions_count; ++i) {
    info.extensions[i] = std::string(
        reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i)));
  }
  glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &info.max_ssbo_bindings);
  glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &info.max_image_bindings);
  info.max_work_group_size.resize(3);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0,
                  &info.max_work_group_size[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,
                  &info.max_work_group_size[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2,
                  &info.max_work_group_size[2]);
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                &info.max_work_group_invocations);
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &info.max_texture_size);
  glGetIntegerv(GL_MAX_IMAGE_UNITS, &info.max_image_units);
  glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &info.max_array_texture_layers);
  RETURN_IF_ERROR(GetOpenGlErrors());
  *gpu_info = info;
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
