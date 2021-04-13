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

absl::Status RequestOpenGlInfo(OpenGlInfo* gl_info) {
  const GLubyte* renderer_name = glGetString(GL_RENDERER);
  if (renderer_name) {
    gl_info->renderer_name = reinterpret_cast<const char*>(renderer_name);
  }

  const GLubyte* vendor_name = glGetString(GL_VENDOR);
  if (vendor_name) {
    gl_info->vendor_name = reinterpret_cast<const char*>(vendor_name);
  }

  const GLubyte* version_name = glGetString(GL_VERSION);
  if (version_name) {
    gl_info->version = reinterpret_cast<const char*>(version_name);
  }

  glGetIntegerv(GL_MAJOR_VERSION, &gl_info->major_version);
  glGetIntegerv(GL_MINOR_VERSION, &gl_info->minor_version);

  return absl::OkStatus();
}

absl::Status RequestGpuInfo(GpuInfo* gpu_info) {
  GpuInfo info;
  RETURN_IF_ERROR(RequestOpenGlInfo(&info.opengl_info));

  GetGpuInfoFromDeviceDescription(info.opengl_info.renderer_name,
                                  GpuApi::kOpenGl, &info);

  GLint extensions_count;
  glGetIntegerv(GL_NUM_EXTENSIONS, &extensions_count);
  info.opengl_info.extensions.resize(extensions_count);
  for (int i = 0; i < extensions_count; ++i) {
    info.opengl_info.extensions[i] = std::string(
        reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i)));
  }
  glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
                &info.opengl_info.max_ssbo_bindings);
  glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS,
                &info.opengl_info.max_image_bindings);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0,
                  &info.opengl_info.max_compute_work_group_size_x);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,
                  &info.opengl_info.max_compute_work_group_size_y);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2,
                  &info.opengl_info.max_compute_work_group_size_z);
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                &info.opengl_info.max_work_group_invocations);
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &info.opengl_info.max_texture_size);
  glGetIntegerv(GL_MAX_IMAGE_UNITS, &info.opengl_info.max_image_units);
  glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS,
                &info.opengl_info.max_array_texture_layers);
  glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,
                &info.opengl_info.max_fragment_image_units);
  glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS,
                &info.opengl_info.max_fragment_uniform_vec4_count);
  glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE,
                &info.opengl_info.max_renderbuffer_size);
  GLint max_viewport_dims[2];
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_viewport_dims);
  info.opengl_info.max_viewport_width = max_viewport_dims[0];
  info.opengl_info.max_viewport_height = max_viewport_dims[1];
  GLint max_color_atttachments;
  glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &max_color_atttachments);
  GLint max_draw_buffers;
  glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buffers);
  info.opengl_info.max_color_atttachments =
      std::min(max_color_atttachments, max_draw_buffers);
  RETURN_IF_ERROR(GetOpenGlErrors());
  *gpu_info = info;
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
