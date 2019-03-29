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

#include "tensorflow/lite/delegates/gpu/gl/compiler.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inline.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.h"
#include "tensorflow/lite/delegates/gpu/gl/float16_conversions.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

struct ExceedSizeChecker {
  bool operator()(uint32_t v) const { return v > max_size; }

  bool operator()(const uint2& v) const {
    return v.x > max_size || v.y > max_size;
  }

  bool operator()(const uint3& v) const {
    return v.x > max_size || v.y > max_size || v.z > max_z_size;
  }

  int max_size;
  int max_z_size;
};

// Returns true if any size variable exceeds the given limit
bool ExceedsMaxSize(const Object& object, const GpuInfo& gpu_info) {
  return absl::visit(ExceedSizeChecker{gpu_info.max_texture_size,
                                       gpu_info.max_array_texture_layers},
                     object.size);
}

ObjectType ChooseFastestObjectType(const GpuInfo& gpu_info) {
  return gpu_info.type == GpuType::ADRENO ? ObjectType::TEXTURE
                                          : ObjectType::BUFFER;
}

ObjectType ChooseFastestRefObjectType(const GpuInfo& gpu_info,
                                      const CompilationOptions& options) {
  if (gpu_info.type != GpuType::ADRENO) {
    return ObjectType::BUFFER;
  }
  switch (gpu_info.gpu_model) {
    case GpuModel::ADRENO630:
      return ObjectType::TEXTURE;
    default:
      return options.allow_precision_loss ? ObjectType::TEXTURE
                                          : ObjectType::BUFFER;
  }
}

// Compiler executes the following steps:
//   1. Runs NodeShader for every node in the input graph.
//   2. Creates a compiled graph that mirrors the input graph and keeps
//      GeneratedCode in operation's attributes.
//   3. Fuses nodes in the compiled graph.
//   4. Generates the full shader code using the nodes in the compiled graph.
class CompilerImpl : public Compiler {
 public:
  // We use const GpuInfo* because it doesn't let you assign temporary object
  CompilerImpl(const NodeShader* node_shader, const GpuInfo* gpu_info,
               const CompilationOptions& options)
      : node_shader_(*node_shader), gpu_info_(*gpu_info), options_(options) {
    if (options_.preferred_obj_type == ObjectType::UNKNOWN) {
      options_.preferred_obj_type = ChooseFastestObjectType(*gpu_info);
    }
    if (options_.ref_obj_type == ObjectType::UNKNOWN) {
      options_.ref_obj_type = ChooseFastestRefObjectType(*gpu_info, options);
    }
  }

  Status Compile(const GraphFloat32& graph,
                 const ShaderCodeCallback& callback) final {
    // It is important to have ids in a compiled graph identical to the given
    // graph.
    RETURN_IF_ERROR(graph.MakeExactCopy(&compiled_graph_));

    // Clear out batch dimension for dynamic batch support.
    if (options_.dynamic_batch) {
      for (auto value : compiled_graph_.values()) {
        value->tensor.shape.b = 1;
      }
    }

    // Generate a shader for a node and all input/output objects.
    for (auto node : compiled_graph_.nodes()) {
      CompiledNodeAttributes attr;
      attr.node_indices.push_back(node->id);
      RETURN_IF_ERROR(node_shader_.GenerateCode(
          {&compiled_graph_, &gpu_info_, node, options_}, &attr.code));
      node->operation.attributes = std::move(attr);
    }

    ModelTransformer transformer(&compiled_graph_, nullptr);
    if (options_.fuse_operations) {
      FuseAutoOutputWithInline fuse_inline;
      if (!transformer.Apply("fuse_auto_with_inline", &fuse_inline)) {
        return InternalError("fuse_auto_with_inline failed");
      }
      FuseInplaceUpdate fuse_inplace;
      if (!transformer.Apply("fuse_inplace_update", &fuse_inplace)) {
        return InternalError("fuse_inplace failed");
      }
      if (options_.auto_input_fusion) {
        FuseAutoInput fuse_auto_input;
        if (!transformer.Apply("fuse_auto_input", &fuse_auto_input)) {
          return InternalError("fuse_auto_input failed");
        }
      }
    }
    RemoveUnusedInplaceUpdates remove_inplace_updates;
    if (!transformer.Apply("remove_inplace_updates", &remove_inplace_updates)) {
      return InternalError("remove_inplace_updates failed");
    }

    // Prepare internal objects.
    std::unordered_map<ValueId, Object> objects;
    for (auto value : compiled_graph_.values()) {
      Object object = MakePHWC4Ref(value->id, value->tensor.shape);
      object.data_type = value->tensor.type;
      // External references may not be upgraded to f16 nor be represented as
      // textures.
      bool is_external =
          graph.IsGraphOutput(value->id) || graph.IsGraphInput(value->id);
      if (is_external) {
        object.object_type = ObjectType::BUFFER;
      } else {
        if (options_.allow_precision_loss) {
          MaybeConvertToFloat16(&object);
        }
      }
      objects[value->id] = std::move(object);
    }

    // Prepare readonly objects and check whether object types are supported.
    for (auto node : compiled_graph_.nodes()) {
      auto& attr =
          absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);

      // Set workload explicitly.
      if (attr.code.workload == uint3()) {
        auto outputs = compiled_graph_.FindOutputs(node->id);
        auto shape = outputs[0]->tensor.shape;
        for (auto output : outputs) {
          if (shape != output->tensor.shape) {
            return FailedPreconditionError(
                "Workload uint3() requires all output sizes to match");
          }
        }
        attr.code.workload =
            uint3(shape.w, shape.h, IntegralDivideRoundUp(shape.c, 4));
      }

      int num_textures = 0;
      // Counts number of used textures and chooses ObjectType for an object.
      auto set_object_type = [&](Object* object) {
        if (object->object_type == ObjectType::BUFFER) {
          // Don't change from buffer once it is set.
          return;
        }
        bool is_ref = IsRef(*object);
        if (num_textures < gpu_info_.max_image_units &&
            !ExceedsMaxSize(*object, gpu_info_) &&
            (object->object_type == ObjectType::TEXTURE ||
             (is_ref && options_.ref_obj_type == ObjectType::TEXTURE) ||
             (!is_ref && options_.preferred_obj_type == ObjectType::TEXTURE))) {
          object->object_type = ObjectType::TEXTURE;
          num_textures++;
        } else {
          object->object_type = ObjectType::BUFFER;
        }
      };

      for (auto& object : attr.code.objects) {
        // Downgrade readonly objects to F16 is requested.
        if (options_.allow_precision_loss) {
          MaybeConvertToFloat16(&object.second);
        }
        set_object_type(&object.second);
      }

      for (auto ref : compiled_graph_.FindInputs(node->id)) {
        set_object_type(&objects[ref->id]);
      }
      for (auto ref : compiled_graph_.FindOutputs(node->id)) {
        set_object_type(&objects[ref->id]);
      }
    }

    // Generate shaders from the transformed graph.
    ShaderCodegen codegen(options_, gpu_info_);
    for (auto node : compiled_graph_.nodes()) {
      auto& attr =
          absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
      if (attr.code.source_code.empty()) {
        // noop. Skip this node.
        continue;
      }

      // Declare inputs and outputs explicitly.
      for (auto ref : compiled_graph_.FindInputs(node->id)) {
        auto object = objects[ref->id];
        object.access = AccessType::READ;
        attr.inputs.push_back(object);
      }
      for (auto ref : compiled_graph_.FindOutputs(node->id)) {
        auto object = objects[ref->id];
        object.access = AccessType::WRITE;
        attr.outputs.push_back(object);
      }

      // Allocate bindings. Textures must be bound first. max_image_units also
      // defines max binding number for a texture.
      uint32_t binding = 0;
      auto set_binding = [&](ObjectType type, Object& object) {
        if (object.object_type == type) {
          object.binding = binding++;
        }
      };
      for (auto& object : attr.inputs) {
        set_binding(ObjectType::TEXTURE, object);
      }
      for (auto& object : attr.outputs) {
        set_binding(ObjectType::TEXTURE, object);
      }
      for (auto& object : attr.code.objects) {
        set_binding(ObjectType::TEXTURE, object.second);
      }
      for (auto& object : attr.inputs) {
        set_binding(ObjectType::BUFFER, object);
      }
      for (auto& object : attr.outputs) {
        set_binding(ObjectType::BUFFER, object);
      }
      for (auto& object : attr.code.objects) {
        set_binding(ObjectType::BUFFER, object.second);
      }

      // Generate source code.
      ShaderCode shader_code;
      RETURN_IF_ERROR(codegen.Build(std::move(attr), &shader_code));
      RETURN_IF_ERROR(callback(std::move(shader_code)));
    }
    return OkStatus();
  }

 private:
  const NodeShader& node_shader_;
  const GpuInfo& gpu_info_;
  CompilationOptions options_;
  GraphFloat32 compiled_graph_;
};

}  // namespace

std::unique_ptr<Compiler> NewCompiler(const NodeShader* node_shader,
                                      const GpuInfo* gpu_info,
                                      const CompilationOptions& options) {
  return absl::make_unique<CompilerImpl>(node_shader, gpu_info, options);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
