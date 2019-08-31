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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OBJECT_ACCESSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OBJECT_ACCESSOR_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// This rewrite handles access to objects both reads and writes.
//
// The following syntax is supported to access objects:
//
//   READ:
//     vec4 value = $data[i]$;
//       where data is a buffer or 1D texture
//     vec4 value = $data[i,j]$;
//       where data is 2D texture
//     vec4 value = $data[i,j,k]$;
//       where data is 3D texture
//
//   WRITE:
//     $data[i] = value$;
//       where data is a buffer or 1D texture
//     $data[i,j] = value$;
//       where data is 2D texture
//     $data[i,j,k] = value$;
//       where data is 3D texture
//
// Accessor supports all types (gvecN) as well as float16.
//
// TODO(akulik): support field in data[x,y,z].x
//
class ObjectAccessor : public InlineRewrite {
 public:
  ObjectAccessor(bool is_mali, VariableAccessor* variable_accessor)
      : ObjectAccessor(is_mali, /*sampler_textures=*/false, variable_accessor) {
  }

  ObjectAccessor(bool is_mali, bool sampler_textures,
                 VariableAccessor* variable_accessor)
      : is_mali_(is_mali),
        sampler_textures_(sampler_textures),
        variable_accessor_(variable_accessor) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final;

  // Return true if object was successfully added.
  bool AddObject(const std::string& name, Object object);

  // Returns objects declarations that need to be added in a shader's code.
  std::string GetObjectDeclarations() const;

  // Returns functions declarations that need to be added in a shader's code.
  // These functions are used by code accessing objects.
  std::string GetFunctionsDeclarations() const;

  // Returns a collection of registered objects
  std::vector<Object> GetObjects() const;

 private:
  RewriteStatus RewriteRead(absl::string_view location, std::string* output);

  RewriteStatus RewriteWrite(absl::string_view location,
                             absl::string_view value, std::string* output);

  std::unordered_map<std::string, Object> name_to_object_;

  const bool is_mali_;
  const bool sampler_textures_;
  VariableAccessor* variable_accessor_;
};

// Implementation details below.

namespace object_accessor_internal {

// Refers to an element in an object.
struct IndexedElement {
  absl::string_view object_name;
  std::vector<absl::string_view> indices;
};

// Splits name[index1, index2...] into 'name' and {'index1', 'index2'...}.
IndexedElement ParseElement(absl::string_view input);

}  // namespace object_accessor_internal
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OBJECT_ACCESSOR_H_
