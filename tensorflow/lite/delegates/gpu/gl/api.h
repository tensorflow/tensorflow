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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_API_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_API_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler_options.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/object_manager.h"
#include "tensorflow/lite/delegates/gpu/gl/runtime_options.h"
#include "tensorflow/lite/delegates/gpu/gl/stats.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"

namespace tflite {
namespace gpu {
namespace gl {

class InferenceContext;

// Represents a model that was prepared for execution. It is stored in a format
// most suitable for execution and optionally may include pre-generated or
// pre-compiled GPU shaders or whatever is needed for efficient execution.
class CompiledModel {
 public:
  virtual ~CompiledModel() = default;

  virtual CompilerStats stats() const = 0;

  // Creates new inference context. Result can outlive @this.
  //
  // NewRun call as well as subsequent calls to InferenceContext methods should
  // be done from the same EGL context.
  virtual Status NewRun(
      const RuntimeOptions& options, const ObjectManager* objects,
      CommandQueue* command_queue,
      std::unique_ptr<InferenceContext>* inference_context) const = 0;

#ifndef TFLITE_GPU_BINARY_RELEASE
  // Serializes compiled model to a string.
  // @return true if serialization finished successfully.
  virtual Status Serialize(
      std::vector<uint8_t>* serialized_compiled_model) const = 0;
#endif  // TFLITE_GPU_BINARY_RELEASE
};

// Turns the given model into "compiled" form that is suitable for inference.
Status Compile(const CompilationOptions& options, const GraphFloat32& model,
               const std::unordered_set<int>& tflite_graph_io,
               const NodeShader& node_shader,
               const WorkgroupsCalculator& workgroup_calculator,
               std::unique_ptr<CompiledModel>* compiled_model);

#ifndef TFLITE_GPU_BINARY_RELEASE
// Reads serialized representation previously created with
// CompiledModel::Serialize call.
Status ReadSerializedModel(const std::vector<uint8_t>& serialized_model,
                           std::unique_ptr<CompiledModel>* compiled_model);
#endif  // TFLITE_GPU_BINARY_RELEASE

// Encapsulates everything needed for one or more inference executions done
// sequentially.
//
// Thread-safe.
class InferenceContext {
 public:
  virtual ~InferenceContext() = default;

  virtual RuntimeStats stats() const = 0;

  // Executes inference.
  virtual Status Execute() = 0;

  // Asks context to reset it for another round. Keep in mind that does not
  // affect inputs nor outputs which are not cleared, so it is possible to
  // re-use them.
  // It is an error to call Reset while previous run is still in progress.
  virtual Status Reset() = 0;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_API_H_
