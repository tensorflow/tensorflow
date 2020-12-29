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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_CALL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_CALL_H_

#include <string>
#include <type_traits>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {

// Primary purpose of this file is to provide useful macro for calling GL
// functions and checking errors. It also attaches a context to status in case
// of a GL error.
//
// Use TFLITE_GPU_CALL_GL as follows:
//
//   For GL functions with a return value:
//     Before:
//       GLint result = glFunc(...);
//       RETURN_IF_ERROR(GetOpenGlErrors());
//     After:
//       GLint result;
//       RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glFunc, &result, ...));
//
//   For GL functions without a return value:
//     Before:
//       glFunc(...);
//       RETURN_IF_ERROR(GetOpenGlErrors());
//     After:
//       RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glFunc, ...));

namespace gl_call_internal {

// For GL functions with a return value.
template <typename T>
struct Caller {
  template <typename F, typename ErrorF, typename... Params>
  absl::Status operator()(const std::string& context, F func, ErrorF error_func,
                          T* result, Params&&... params) {
    *result = func(std::forward<Params>(params)...);
    const auto status = error_func();
    if (status.ok()) return absl::OkStatus();
    return absl::Status(status.code(),
                        std::string(status.message()) + ": " + context);
  }
};

// For GL functions without a return value.
template<>
struct Caller<void> {
  template <typename F, typename ErrorF, typename... Params>
  absl::Status operator()(const std::string& context, F func, ErrorF error_func,
                          Params&&... params) {
    func(std::forward<Params>(params)...);
    const auto status = error_func();
    if (status.ok()) return absl::OkStatus();
    return absl::Status(status.code(),
                        std::string(status.message()) + ": " + context);
  }
};

template <typename F, typename ErrorF, typename ResultT, typename... ParamsT>
absl::Status CallAndCheckError(const std::string& context, F func,
                               ErrorF error_func, ResultT* result,
                               ParamsT&&... params) {
  return Caller<ResultT>()(context, func, error_func, result,
                           std::forward<ParamsT>(params)...);
}

template <typename F, typename ErrorF, typename... Params>
absl::Status CallAndCheckError(const std::string& context, F func,
                               ErrorF error_func, Params&&... params) {
  return Caller<void>()(context, func, error_func,
                        std::forward<Params>(params)...);
}

}  // namespace gl_call_internal

// XX_STRINGIFY is a helper macro to effectively apply # operator to an
// arbitrary value.
#define TFLITE_GPU_INTERNAL_STRINGIFY_HELPER(x) #x
#define TFLITE_GPU_INTERNAL_STRINGIFY(x) TFLITE_GPU_INTERNAL_STRINGIFY_HELPER(x)
#define TFLITE_GPU_FILE_LINE \
  __FILE__ ":" TFLITE_GPU_INTERNAL_STRINGIFY(__LINE__)

#define TFLITE_GPU_CALL_GL(method, ...)                   \
  ::tflite::gpu::gl::gl_call_internal::CallAndCheckError( \
      #method " in " TFLITE_GPU_FILE_LINE, method,        \
      ::tflite::gpu::gl::GetOpenGlErrors, __VA_ARGS__)

#define TFLITE_GPU_CALL_EGL(method, ...)                  \
  ::tflite::gpu::gl::gl_call_internal::CallAndCheckError( \
      #method " in " TFLITE_GPU_FILE_LINE, method,        \
      ::tflite::gpu::gl::GetEglError, __VA_ARGS__)

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_CALL_H_
