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

#include "tensorflow/lite/delegates/gpu/gl/api.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/runtime.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "tensorflow/lite/delegates/gpu/gl/serialization.h"
#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {
namespace {

using ObjectsSizes = std::unordered_map<ValueId, size_t>;

enum class InferenceContextState {
  NOT_STARTED,
  IN_PROGRESS,
};

class InferenceContextImpl : public InferenceContext {
 public:
  explicit InferenceContextImpl(std::unique_ptr<Runtime> runtime)
      : runtime_(std::move(runtime)) {}

  Status Execute() final {
    std::lock_guard<std::mutex> lock(guard_);
    if (state_ != InferenceContextState::NOT_STARTED) {
      return FailedPreconditionError("InferenceContext is not reset");
    }
    state_ = InferenceContextState::IN_PROGRESS;
    return runtime_->Execute();
  }

  Status Reset() final {
    std::lock_guard<std::mutex> lock(guard_);
    // TODO(akulik): should Reset not return Status?
    state_ = InferenceContextState::NOT_STARTED;
    return OkStatus();
  }

  RuntimeStats stats() const final { return runtime_->stats(); }

 private:
  std::unique_ptr<Runtime> runtime_;

  mutable std::mutex guard_;
  InferenceContextState state_ = InferenceContextState::NOT_STARTED;
};

class InferenceContextWithBatchImpl : public InferenceContext {
 public:
  InferenceContextWithBatchImpl(const ObjectsSizes& sizes,
                                const ObjectManager* objects,
                                std::unique_ptr<ObjectManager> refs,
                                std::unique_ptr<Runtime> runtime)
      : sizes_(sizes),
        objects_(objects),
        refs_(std::move(refs)),
        runtime_(std::move(runtime)) {}

  Status Execute() final {
    std::lock_guard<std::mutex> lock(guard_);
    if (state_ != InferenceContextState::NOT_STARTED) {
      return FailedPreconditionError("InferenceContext is not reset");
    }
    state_ = InferenceContextState::IN_PROGRESS;

    // Calculate expected number of batches and check that all external objects
    // match that number.
    int num_batches = 0;
    for (const auto& s : sizes_) {
      const ValueId id = s.first;
      const size_t byte_size = s.second;

      auto buffer = objects_->FindBuffer(id);
      if (!buffer) continue;

      if (buffer->bytes_size() % byte_size) {
        return InvalidArgumentError(absl::StrCat(
            "Object ", id, " does not match expected byte size: ", byte_size));
      }

      const size_t b = buffer->bytes_size() / byte_size;
      if (num_batches == 0) {
        num_batches = b;
      } else if (num_batches != b) {
        return InvalidArgumentError(absl::StrCat(
            "Object ", id, " size does not match expected batch size: ", b,
            " vs ", num_batches));
      }
    }

    for (size_t b = 0; b < num_batches; ++b) {
      // slice external objects by batch.
      for (const auto& s : sizes_) {
        const ValueId id = s.first;
        const size_t byte_size = s.second;
        auto buffer = objects_->FindBuffer(id);
        if (buffer) {
          auto ref = refs_->FindBuffer(id);
          if (!ref) {
            return InvalidArgumentError(
                absl::StrCat("Reference to ", id, " is not found"));
          }
          RETURN_IF_ERROR(buffer->MakeView(b * byte_size, byte_size, ref));
        }
      }
      RETURN_IF_ERROR(runtime_->Execute());
    }
    return OkStatus();
  }

  Status Reset() final {
    std::lock_guard<std::mutex> lock(guard_);
    state_ = InferenceContextState::NOT_STARTED;
    // TODO(akulik): should Reset not return Status?
    return OkStatus();
  }

  RuntimeStats stats() const final { return runtime_->stats(); }

 private:
  const ObjectsSizes sizes_;
  const ObjectManager* objects_;

  // view over external objects provided by a user.
  std::unique_ptr<ObjectManager> refs_;
  std::unique_ptr<Runtime> runtime_;

  mutable std::mutex guard_;
  InferenceContextState state_ = InferenceContextState::NOT_STARTED;
};

struct ProgramParameters {
  // A list of uniform parameters to be set.
  std::vector<Variable> parameters;

  // A list of objects to bind to opengl program.
  std::vector<Object> objects;

  uint3 workgroup_size;
  uint3 num_workgroups;

  size_t shader_idx;
};

std::string GetShaderHeader(uint3 localsize) {
  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

class CompiledModelImpl
#ifndef TFLITE_GPU_BINARY_RELEASE
    : public CompiledModel,
      public DeserializationHandler {
#else
    : public CompiledModel {
#endif  // TFLITE_GPU_BINARY_RELEASE
 public:
  explicit CompiledModelImpl(const GpuInfo& gpu_info) : gpu_info_(gpu_info) {}

  // Called while compiling shaders from scratch
  Status Add(const WorkgroupsCalculator& workgroup_calculator,
             ShaderCode code) {
    // Calculate workgroup size.
    uint3 workgroup_size = workgroup_calculator.Calculate(code);
    uint3 num_workgroups = IntegralDivideRoundUp(code.workload, workgroup_size);

    for (const auto& object : code.objects) {
      if (IsRef(object)) {
        object_sizes_[GetRef(object)] = ByteSizeOf(object);
      }
    }

    // Store full shader and compile it if necessary.
    size_t shader_idx;
    RETURN_IF_ERROR(
        AddFullShader(code.source_code, workgroup_size, &shader_idx));
    programs_.push_back({
        std::move(code.parameters),
        std::move(code.objects),
        workgroup_size,
        num_workgroups,
        shader_idx,
    });
    return OkStatus();
  }

  // Store full shader and compile it if necessary.
  // Returns full_shader_index
  Status AddFullShader(const std::string& partial_shader,
                       const uint3& workgroup_size, size_t* size) {
    std::string shader_src = GetShaderHeader(workgroup_size) + partial_shader;
    auto it = shader_to_index_.find(shader_src);
    if (it == shader_to_index_.end()) {
      GlShader shader;
      RETURN_IF_ERROR(
          GlShader::CompileShader(GL_COMPUTE_SHADER, shader_src, &shader));
      shaders_.push_back(std::move(shader));
      shader_to_index_.insert({shader_src, shader_to_index_.size()});
      *size = shader_to_index_.size() - 1;
    } else {
      *size = it->second;
    }
    return OkStatus();
  }

  Status NewRun(
      const RuntimeOptions& options, const ObjectManager* objects,
      CommandQueue* command_queue,
      std::unique_ptr<InferenceContext>* inference_context) const final {
    std::unique_ptr<ObjectManager> refs;
    if (dynamic_batch_) {
      // Runtime is using objects from refs that will point to provided objects.
      // At this point just create 0 batch slice references.
      refs = absl::make_unique<ObjectManager>();
      for (const auto& s : object_sizes_) {
        auto buffer = objects->FindBuffer(s.first);
        if (!buffer) continue;
        GlBuffer ref;
        RETURN_IF_ERROR(buffer->MakeView(0, s.second, &ref));
        RETURN_IF_ERROR(refs->RegisterBuffer(s.first, std::move(ref)));
      }
    }
    auto runtime = absl::make_unique<Runtime>(options, gpu_info_, command_queue,
                                              refs ? refs.get() : objects);
    for (auto& c : programs_) {
      RETURN_IF_ERROR(runtime->AddProgram(shaders_[c.shader_idx], c.parameters,
                                          c.objects, c.num_workgroups));
    }
    RETURN_IF_ERROR(runtime->PrepareForExecution());
    if (dynamic_batch_) {
      *inference_context = absl::make_unique<InferenceContextWithBatchImpl>(
          object_sizes_, objects, std::move(refs), std::move(runtime));
    } else {
      *inference_context =
          absl::make_unique<InferenceContextImpl>(std::move(runtime));
    }
    return OkStatus();
  }

#ifndef TFLITE_GPU_BINARY_RELEASE
  // Called on deserialization
  Status OnProgram(const std::vector<Variable>& parameters,
                   const std::vector<Object>& objects,
                   const uint3& workgroup_size, const uint3& num_workgroups,
                   size_t partial_shader_index) final {
    for (auto& object : objects) {
      if (IsRef(object)) {
        object_sizes_[GetRef(object)] = ByteSizeOf(object);
      }
    }

    size_t shader_idx;
    RETURN_IF_ERROR(AddFullShader(partial_shaders_[partial_shader_index],
                                  workgroup_size, &shader_idx));
    programs_.push_back({
        parameters,
        objects,
        workgroup_size,
        num_workgroups,
        shader_idx,
    });
    return OkStatus();
  }

  Status Serialize(
      std::vector<uint8_t>* serialized_compiled_model) const final {
    SerializedCompiledModelBuilder builder;

    // sort shaders first. They need to be serialized in order.
    std::vector<std::string> full_shaders(shaders_.size());
    for (const auto& shader : shader_to_index_) {
      full_shaders[shader.second] = shader.first;
    }

    std::unordered_map<std::string, size_t> partial_shader_to_index;
    std::vector<std::string> partial_shaders;
    for (const auto& program : programs_) {
      // Remove a header from a shader.
      std::string shader_without_header = full_shaders[program.shader_idx];
      shader_without_header.erase(0, shader_without_header.find("in;") + 3);

      // Insert shader into partial shaders array.
      auto it = partial_shader_to_index.find(shader_without_header);
      size_t shader_idx;
      if (it == partial_shader_to_index.end()) {
        shader_idx = partial_shaders.size();
        partial_shaders.push_back(shader_without_header);
        builder.AddShader(shader_without_header);
        partial_shader_to_index.insert({shader_without_header, shader_idx});
      } else {
        shader_idx = it->second;
      }
      builder.AddProgram(program.parameters, program.objects,
                         program.workgroup_size, program.num_workgroups,
                         shader_idx);
    }
    CompiledModelOptions options;
    options.dynamic_batch = dynamic_batch_;
    auto data = builder.Finalize(options);
    serialized_compiled_model->insert(serialized_compiled_model->end(),
                                      data.begin(), data.end());
    return OkStatus();
  }

  Status OnShader(absl::Span<const char> shader_src) final {
    std::string source(shader_src.data(), shader_src.size());
    partial_shaders_.push_back(source);
    return OkStatus();
  }

  void OnOptions(const CompiledModelOptions& options) final {
    dynamic_batch_ = options.dynamic_batch;
  }
#endif  // TFLITE_GPU_BINARY_RELEASE

  CompilerStats stats() const final { return stats_; }

  void set_dynamic_batch(bool dynamic_batch) { dynamic_batch_ = dynamic_batch; }

 private:
  const GpuInfo gpu_info_;
  bool dynamic_batch_ = false;

  std::vector<std::string> partial_shaders_;
  std::vector<GlShader> shaders_;

  // Shaders are serialized in order of their indices.
  std::unordered_map<std::string, size_t> shader_to_index_;
  std::deque<ProgramParameters> programs_;
  std::unordered_map<ValueId, size_t> object_sizes_;
  CompilerStats stats_;
};
}  // namespace

Status Compile(const CompilationOptions& options, const GraphFloat32& model,
               const std::unordered_set<int>& tflite_graph_io,
               const NodeShader& node_shader,
               const WorkgroupsCalculator& workgroup_calculator,
               std::unique_ptr<CompiledModel>* compiled_model) {
  if (!IsBatchMatchesForAllValues(model)) {
    return InvalidArgumentError("Only identical batch dimension is supported");
  }
  GpuInfo gpu_info;
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
  if (!IsOpenGl31OrAbove(gpu_info)) {
    return InternalError(
        "OpenGL ES 3.1 or above is required to use OpenGL inference.");
  }
  auto compiled_model_impl = absl::make_unique<CompiledModelImpl>(gpu_info);
  compiled_model_impl->set_dynamic_batch(options.dynamic_batch);
  auto compiler = NewCompiler(&node_shader, &gpu_info, options);
  RETURN_IF_ERROR(
      compiler->Compile(model, tflite_graph_io, [&](ShaderCode code) -> Status {
        return compiled_model_impl->Add(workgroup_calculator, std::move(code));
      }));
  *compiled_model = std::move(compiled_model_impl);
  return OkStatus();
}

#ifndef TFLITE_GPU_BINARY_RELEASE
Status ReadSerializedModel(const std::vector<uint8_t>& serialized_model,
                           std::unique_ptr<CompiledModel>* compiled_model) {
  GpuInfo gpu_info;
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
  if (!IsOpenGl31OrAbove(gpu_info)) {
    return InternalError(
        "OpenGL ES 3.1 or above is required to use OpenGL inference.");
  }
  auto compiled_model_impl = absl::make_unique<CompiledModelImpl>(gpu_info);
  RETURN_IF_ERROR(DeserializeCompiledModel(
      absl::MakeConstSpan(serialized_model), compiled_model_impl.get()));
  *compiled_model = std::move(compiled_model_impl);
  return OkStatus();
}
#endif  // TFLITE_GPU_BINARY_RELEASE

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
