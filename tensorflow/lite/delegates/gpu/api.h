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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_API_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_API_H_

// Usage example:
//
//   // Builder is created from a model using GPU-specific parameters.
//   std::unique_ptr<InferenceBuilder> builder = ...;
//
//   // input data is coming from a texture
//   // output data goes to CPU
//   builder->SetInputObjectDef(0, {DataType::FLOAT16, DataLayout::PHWC4,
//                                  ObjectType::OPENGL_TEXTURE, true});
//   builder->SetOutputObjectDef(0, {DataType::FLOAT32, DataLayout::BHWC,
//                                  ObjectType::CPU_MEMORY, false});
//   std::unique_ptr<InferenceRunner> runner;
//   RETURN_IF_ERROR(builder->Build(&runner));  // may take significant time.
//   RETURN_IF_ERROR(
//       runner->SetInputObject(0, OpenGlTexture{texture_ud, texture_format}));
//   RETURN_IF_ERROR(runner->Run());

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include <CL/cl.h>
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {

// Common abbreviations:
//   B  - batch
//   H  - height
//   W  - width
//   C  - channels
//   D  - depth := IntegralDivideRoundUp(C, 4)
//   C4 - is the constant = 4.
enum class DataLayout {
  UNKNOWN,
  BHWC,
  DHWC4,
  HWDC4,
  HDWC4,
};

enum class ObjectType {
  UNKNOWN,
  OPENGL_SSBO,
  OPENGL_TEXTURE,
  CPU_MEMORY,
  OPENCL_TEXTURE,
  OPENCL_BUFFER,
};

struct OpenGlBuffer {
  OpenGlBuffer() = default;
  explicit OpenGlBuffer(GLuint new_id) : id(new_id) {}

  GLuint id = GL_INVALID_INDEX;
};

struct OpenGlTexture {
  OpenGlTexture() = default;
  OpenGlTexture(GLuint new_id, GLenum new_format)
      : id(new_id), format(new_format) {}

  GLuint id = GL_INVALID_INDEX;
  GLenum format = GL_INVALID_ENUM;
};

struct OpenClBuffer {
  OpenClBuffer() = default;
  explicit OpenClBuffer(cl_mem new_memobj) : memobj(new_memobj) {}

  cl_mem memobj = nullptr;
};

struct OpenClTexture {
  OpenClTexture() = default;
  explicit OpenClTexture(cl_mem new_memobj) : memobj(new_memobj) {}

  cl_mem memobj = nullptr;
  // TODO(akulik): should it specify texture format?
};

struct CpuMemory {
  CpuMemory() = default;
  CpuMemory(void* new_data, size_t new_size_bytes)
      : data(new_data), size_bytes(new_size_bytes) {}

  void* data = nullptr;
  size_t size_bytes = 0;
};

template <typename T>
inline CpuMemory MakeCpuMemory(absl::Span<T> t) {
  CpuMemory m;
  m.data = t.data();
  m.size_bytes = t.size() * sizeof(T);
  return m;
}

template <typename T>
inline CpuMemory MakeReadableCpuMemory(absl::Span<const T> t) {
  CpuMemory m;
  m.data = const_cast<T*>(t.data());
  m.size_bytes = t.size() * sizeof(T);
  return m;
}

// Defines object representation.
struct ObjectDef {
  DataType data_type = DataType::UNKNOWN;
  DataLayout data_layout = DataLayout::UNKNOWN;
  ObjectType object_type = ObjectType::UNKNOWN;

  // If true, then object is managed externally and needs to be provided to
  // InferenceRunner by a user before running inference.
  //
  // User-provided objects will not be re-used internally for any purpose to
  // lower overall memory usage.
  bool user_provided = false;

  bool operator==(const ObjectDef& other) const {
    return data_type == other.data_type && data_layout == other.data_layout &&
           object_type == other.object_type &&
           user_provided == other.user_provided;
  }
};

bool IsValid(const ObjectDef& def);

struct Dimensions {
  Dimensions() : b(1), h(1), w(1), c(1) {}

  Dimensions(int32_t batch, int32_t height, int32_t width, int32_t channels)
      : b(batch), h(height), w(width), c(channels) {}

  int32_t d() const { return IntegralDivideRoundUp(c, 4); }

  int32_t product() const { return b * h * w * c; }

  bool operator==(const Dimensions& other) const {
    return b == other.b && h == other.h && w == other.w && c == other.c;
  }

  int32_t b;
  int32_t h;
  int32_t w;
  int32_t c;
};

// Connects tensor shape with corresponding object definition.
struct TensorObjectDef {
  // Dimensions semantic is defined by corresponding DataLayout.
  Dimensions dimensions;
  ObjectDef object_def;

  bool operator==(const TensorObjectDef& other) const {
    return dimensions == other.dimensions && object_def == other.object_def;
  }
};

// @return true if tensor object def is defined.
bool IsValid(const TensorObjectDef& def);

// @return the number of elements in a tensor object.
uint32_t NumElements(const TensorObjectDef& def);

using TensorObject = absl::variant<absl::monostate, OpenGlBuffer, OpenGlTexture,
                                   CpuMemory, OpenClBuffer, OpenClTexture>;

// @return true if object is set and corresponding values are defined.
bool IsValid(const TensorObjectDef& def, const TensorObject& object);

ObjectType GetType(const TensorObject& object);

// @return true if corresponding object is set for the given type
bool IsObjectPresent(ObjectType type, const TensorObject& obj);

class InferenceRunner;

// Allows to inspect and change input and output definitions before a graph is
// prepared for the inference.
class InferenceBuilder {
 public:
  virtual ~InferenceBuilder() {}

  // Returns inference graph inputs and outputs definitions.
  virtual std::vector<TensorObjectDef> inputs() const = 0;
  virtual std::vector<TensorObjectDef> outputs() const = 0;

  // Sets new shape for the input if underlying implementation and graph
  // structure allows dynamic tensors.
  virtual Status SetInputShape(int index, const Dimensions& dimensions) = 0;

  // Updates object definitions for the given index. Implementation may allow
  // to use different layouts and/or data type conversions between objects
  // defined in a graph and given objects, for example:
  //   input '0' is DataType::FLOAT32, DataLayout::BHWC.
  //   A user, however, has an input in DataType::FLOAT16, DataLayout::PHWC4.
  //   An implementation may allow this transformation to happen automatically
  //   under the hood.
  virtual Status SetInputObjectDef(int index, ObjectDef def) = 0;
  virtual Status SetOutputObjectDef(int index, ObjectDef def) = 0;
  virtual Status SetAllInputObjectDefsTo(ObjectDef def) {
    auto input_defs = inputs();
    for (int i = 0; i < input_defs.size(); ++i) {
      RETURN_IF_ERROR(SetInputObjectDef(i, def));
    }
    return OkStatus();
  }
  virtual Status SetAllOutputObjectDefsTo(ObjectDef def) {
    auto output_defs = outputs();
    for (int i = 0; i < output_defs.size(); ++i) {
      RETURN_IF_ERROR(SetOutputObjectDef(i, def));
    }
    return OkStatus();
  }

  // Creates new instance of the inference runner. InferenceBuilder stays valid
  // and could be used to create another inference runner if needed.
  //
  // This method may take significant time to prepare new inference runner. For
  // example, it may require to compile OpenGL shaders.
  virtual Status Build(std::unique_ptr<InferenceRunner>* runner) = 0;
};

// Runs prepared inference. Every object marked as external needs to be set
// prior calling Run method.
class InferenceRunner {
 public:
  virtual ~InferenceRunner() {}

  // Returns inference graph inputs and outputs definitions.
  virtual std::vector<TensorObjectDef> inputs() const = 0;
  virtual std::vector<TensorObjectDef> outputs() const = 0;

  // Getters provide access to underlying objects for the given index.
  // Setters allow to set or change external object for the given index. Note,
  // object need to match object definition set before in InferenceBuilder.

  virtual Status GetInputObject(int index, TensorObject* object) = 0;
  virtual Status GetOutputObject(int index, TensorObject* object) = 0;
  virtual Status SetInputObject(int index, TensorObject object) = 0;
  virtual Status SetOutputObject(int index, TensorObject object) = 0;

  virtual Status Run() = 0;
};

// Encapsulated compilation/runtime tradeoffs.
enum class InferenceUsage {
  UNKNOWN,

  // InferenceRunner will be used only once. Therefore, it is important to
  // minimize bootstrap time as well.
  FAST_SINGLE_ANSWER,

  // Prefer maximizing the throughput. Same inference runner will be used
  // repeatedly on different inputs.
  SUSTAINED_SPEED,
};

// Defines aspects to control while instantiating a runner.
enum class InferencePriority {
  UNKNOWN,

  MIN_LATENCY,

  MAX_PRECISION,

  MIN_MEMORY_USAGE,
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_API_H_
