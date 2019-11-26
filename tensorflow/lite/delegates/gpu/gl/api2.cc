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

#include "tensorflow/lite/delegates/gpu/gl/api2.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/converter.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/registry.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/runtime.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

std::string GetShaderHeader(uint3 localsize) {
  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

// Wraps given SSBO into GlBuffer object that does not have ownership.
Status WrapSSBO(OpenGlBuffer ssbo, GlBuffer* buffer) {
  int64_t size_bytes;
  RETURN_IF_ERROR(GetSSBOSize(ssbo.id, &size_bytes));
  *buffer = GlBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.id, size_bytes, 0, false);
  return OkStatus();
}

Status MaybeAllocateGlBuffer(const TensorObjectDef& def, GlBuffer* ssbo) {
  if (def.object_def.object_type != gpu::ObjectType::OPENGL_SSBO) {
    return InvalidArgumentError("Tensor object is not GL SSBO");
  }
  const uint32_t num_elements = NumElements(def);
  switch (def.object_def.data_type) {
    case DataType::FLOAT32:
      return CreateReadWriteShaderStorageBuffer<float>(num_elements, ssbo);
    case DataType::FLOAT16:
      return CreateReadWriteShaderStorageBuffer<uint16_t>(num_elements, ssbo);
    default:
      return InternalError(
          "Unable to create new GL SSBO. Unsupported data type.");
  }
  return OkStatus();
}

// Does one-step conversion between internal and external objects.
// It may also allocate external objects if requested.
class DefaultTensorTie : public TensorTie {
 public:
  DefaultTensorTie(const TensorTieDef& def, TensorObject internal_obj,
                   ObjectManager* objects)
      : TensorTie(def), objects_(objects), internal_obj_(internal_obj) {}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
    return converter_builder.IsSupported(def.internal_def, def.external_def) &&
           converter_builder.IsSupported(def.external_def, def.internal_def);
  }

  static Status New(const TensorTieDef& def,
                    TensorObjectConverterBuilder* converter_builder,
                    ObjectManager* objects, std::unique_ptr<TensorTie>* tie) {
    auto tie_impl =
        absl::make_unique<DefaultTensorTie>(def, TensorObject{}, objects);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder));
    *tie = std::move(tie_impl);
    return OkStatus();
  }

  static Status New(const TensorTieDef& def,
                    TensorObjectConverterBuilder* converter_builder,
                    TensorObject internal_object,
                    std::unique_ptr<TensorTie>* tie) {
    if (!IsValid(def.internal_def, internal_object)) {
      return InternalError("Internal object does not match definition.");
    }

    auto tie_impl =
        absl::make_unique<DefaultTensorTie>(def, internal_object, nullptr);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder));
    *tie = std::move(tie_impl);
    return OkStatus();
  }

  Status CopyToExternalObject() final {
    if (!converter_to_) {
      return OkStatus();
    }
    return converter_to_->Convert(internal_obj_, GetExternalObject());
  }

  Status CopyFromExternalObject() final {
    if (!converter_from_) {
      return OkStatus();
    }
    return converter_from_->Convert(GetExternalObject(), internal_obj_);
  }

  Status SetExternalObject(TensorObject obj) final {
    if (!def().external_def.object_def.user_provided) {
      return InvalidArgumentError("External object is read-only");
    }
    if (!IsValid(def().external_def, obj)) {
      return InvalidArgumentError("Given object is not valid");
    }
    // TODO(akulik): external object should propagate to internal.
    if (IsSameDef()) {
      return UnimplementedError("Not supported");
    }
    external_obj_ = obj;
    return OkStatus();
  }

  TensorObject GetExternalObject() final { return external_obj_; }

 private:
  bool IsSameDef() const {
    const auto& external_def = def().external_def.object_def;
    const auto& internal_def = def().internal_def.object_def;
    return (external_def.object_type == internal_def.object_type &&
            external_def.data_type == internal_def.data_type &&
            external_def.data_layout == internal_def.data_layout) ||
           // Check for equivalent layouts that have the same size.
           (external_def.object_type == internal_def.object_type &&
            external_def.data_type == internal_def.data_type &&
            external_def.data_layout == DataLayout::BHWC &&
            internal_def.data_layout == DataLayout::DHWC4 &&
            def().external_def.dimensions.c == 4);
  }
  Status Init(TensorObjectConverterBuilder* converter_builder) {
    // First check is an object is user provided.
    const auto& external_def = def().external_def.object_def;

    const bool is_same_def = IsSameDef();

    if (!is_same_def) {
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().internal_def, def().external_def, &converter_to_));
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().external_def, def().internal_def, &converter_from_));
    }

    if (external_def.user_provided) {
      if (is_same_def) {
        return OkStatus();
      }
      // Object is provided by a user, but runtime expects different object
      // type. Therefore, we have to allocate internal object and convert.
      return MaybeAllocateInternalObject();
    } else {
      RETURN_IF_ERROR(MaybeAllocateInternalObject());

      if (is_same_def) {
        // Object is NOT provided by a user, but it matches definition expected
        // by runtime. Conversion is not needed.
        external_obj_ = internal_obj_;
        return OkStatus();
      }

      // Object is NOT provided by a user.
      return MaybeAllocateExternalObject();
    }
    return OkStatus();
  }

  Status MaybeAllocateInternalObject() {
    const TensorObjectDef& d = def().internal_def;
    if (d.object_def.user_provided) {
      return OkStatus();
    }
    switch (d.object_def.object_type) {
      case gpu::ObjectType::OPENGL_SSBO: {
        GlBuffer ssbo;
        RETURN_IF_ERROR(MaybeAllocateGlBuffer(d, &ssbo));
        internal_obj_ = OpenGlBuffer{ssbo.id()};
        RETURN_IF_ERROR(objects_->RegisterBuffer(def().id, std::move(ssbo)));
        break;
      }
      // TODO(akulik): support textures as internal object when compiler permits
      default:
        return InternalError("Unexpected object type");
    }
    return OkStatus();
  }

  Status MaybeAllocateExternalObject() {
    const TensorObjectDef& d = def().external_def;
    switch (d.object_def.object_type) {
      case gpu::ObjectType::CPU_MEMORY: {
        size_t bytes_size = NumElements(d) * SizeOf(d.object_def.data_type);
        cpu_memory_.resize(bytes_size);
        external_obj_ = CpuMemory{cpu_memory_.data(), cpu_memory_.size()};
        break;
      }
      case gpu::ObjectType::OPENGL_SSBO: {
        RETURN_IF_ERROR(MaybeAllocateGlBuffer(d, &external_ssbo_));
        external_obj_ = OpenGlBuffer{external_ssbo_.id()};
        GlBuffer bbb;
        RETURN_IF_ERROR(WrapSSBO(OpenGlBuffer{external_ssbo_.id()}, &bbb));
        break;
      }
      default:
        return InternalError("Unexpected object type");
    }
    return OkStatus();
  }

  ObjectManager* objects_;

  // hold references to objects.
  TensorObject internal_obj_;
  TensorObject external_obj_;

  // Hold actual objects.
  GlBuffer external_ssbo_;
  std::vector<uint8_t> cpu_memory_;

  std::unique_ptr<TensorObjectConverter> converter_to_;
  std::unique_ptr<TensorObjectConverter> converter_from_;
};

// Copies data to intermediate OpenGL buffer and then does two step conversion.
// It drives the following cases were one-step conversion is not supported:
//   - CPU BHWC -> GL buffer BHWC -> GL texture DHWC4.
class TwoStepTensorTie : public TensorTie {
 public:
  explicit TwoStepTensorTie(const TensorTieDef& def) : TensorTie(def) {}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
    auto defs = MakeOuterInnerDefs(def);
    return DefaultTensorTie::IsSupported(defs.first, converter_builder) &&
           DefaultTensorTie::IsSupported(defs.second, converter_builder);
  }

  static Status New(const TensorTieDef& def,
                    TensorObjectConverterBuilder* converter_builder,
                    ObjectManager* objects, std::unique_ptr<TensorTie>* tie) {
    auto tie_impl = absl::make_unique<TwoStepTensorTie>(def);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, objects));
    *tie = std::move(tie_impl);
    return OkStatus();
  }

  Status CopyToExternalObject() final {
    RETURN_IF_ERROR(inner_tie_->CopyToExternalObject());
    return outer_tie_->CopyToExternalObject();
  }

  Status CopyFromExternalObject() final {
    RETURN_IF_ERROR(outer_tie_->CopyFromExternalObject());
    return inner_tie_->CopyFromExternalObject();
  }

  Status SetExternalObject(TensorObject obj) final {
    return outer_tie_->SetExternalObject(obj);
  }

  TensorObject GetExternalObject() final {
    return outer_tie_->GetExternalObject();
  }

 private:
  static std::pair<TensorTieDef, TensorTieDef> MakeOuterInnerDefs(
      const TensorTieDef& def) {
    TensorTieDef outer_def;
    outer_def.external_def = def.external_def;
    outer_def.internal_def = def.external_def;
    outer_def.internal_def.object_def.object_type =
        gpu::ObjectType::OPENGL_SSBO;
    // Will not allocate new SSBO
    outer_def.internal_def.object_def.user_provided = true;

    TensorTieDef inner_def;
    inner_def.id = def.id;
    inner_def.external_def = outer_def.internal_def;
    // Should not allocate external object.
    inner_def.external_def.object_def.user_provided = false;
    // Reflects what is actually supported by compiler.
    inner_def.internal_def.dimensions = inner_def.external_def.dimensions;
    inner_def.internal_def.object_def.data_type = DataType::FLOAT32;
    inner_def.internal_def.object_def.data_layout = DataLayout::DHWC4;
    inner_def.internal_def.object_def.object_type =
        gpu::ObjectType::OPENGL_SSBO;
    // It may allocate another internal object and should register it to
    // ObjectManager.
    inner_def.internal_def.object_def.user_provided = false;
    return std::make_pair(outer_def, inner_def);
  }

  Status Init(TensorObjectConverterBuilder* converter_builder,
              ObjectManager* objects) {
    auto defs = MakeOuterInnerDefs(def());
    RETURN_IF_ERROR(DefaultTensorTie::New(defs.second, converter_builder,
                                          objects, &inner_tie_));
    return DefaultTensorTie::New(defs.first, converter_builder,
                                 inner_tie_->GetExternalObject(), &outer_tie_);
  }

  std::unique_ptr<TensorTie> inner_tie_;
  std::unique_ptr<TensorTie> outer_tie_;
};

// Responsible for creating new tensor tie objects.
class TensorTieFactory {
 public:
  explicit TensorTieFactory(const InferenceEnvironmentOptions& env_options)
      : converter_builder_(NewConverterBuilder(env_options.queue)) {}

  bool IsSupported(const TensorTieDef& def) const {
    return IsValid(def.external_def.object_def) &&
           (DefaultTensorTie::IsSupported(def, *converter_builder_) ||
            TwoStepTensorTie::IsSupported(def, *converter_builder_));
  }

  Status NewTensorTie(const TensorTieDef& def, ObjectManager* objects,
                      std::unique_ptr<TensorTie>* tie) {
    auto converter = converter_builder_.get();
    if (DefaultTensorTie::IsSupported(def, *converter)) {
      return DefaultTensorTie::New(def, converter, objects, tie);
    }
    if (TwoStepTensorTie::IsSupported(def, *converter)) {
      return TwoStepTensorTie::New(def, converter, objects, tie);
    }
    return UnimplementedError("Unsupported tensor tie definition.");
  }

 private:
  std::unique_ptr<TensorObjectConverterBuilder> converter_builder_;
};

class InferenceRunnerImpl : public InferenceRunner {
 public:
  InferenceRunnerImpl(std::unique_ptr<Runtime> runtime,
                      std::unique_ptr<ObjectManager> objects)
      : runtime_(std::move(runtime)), objects_(std::move(objects)) {}

  Status Initialize(const std::vector<TensorTieDef>& inputs,
                    const std::vector<TensorTieDef>& outputs,
                    TensorTieFactory* tie_factory) {
    RETURN_IF_ERROR(LinkTensors(inputs, tie_factory, &inputs_));
    RETURN_IF_ERROR(LinkTensors(outputs, tie_factory, &outputs_));
    for (const auto& def : outputs) {
      output_to_cpu_ |= def.external_def.object_def.object_type ==
                        gpu::ObjectType::CPU_MEMORY;
    }
    return OkStatus();
  }

  std::vector<TensorObjectDef> inputs() const override {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const override {
    return GetExternalDefinitions(outputs_);
  }

  Status GetInputObject(int index, TensorObject* object) override {
    if (index < 0 || index >= inputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    *object = inputs_[index]->GetExternalObject();
    return OkStatus();
  }

  Status GetOutputObject(int index, TensorObject* object) override {
    if (index < 0 || index >= outputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    *object = outputs_[index]->GetExternalObject();
    return OkStatus();
  }

  Status SetInputObject(int index, TensorObject object) override {
    if (index < 0 || index >= inputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    return inputs_[index]->SetExternalObject(object);
  }

  Status SetOutputObject(int index, TensorObject object) override {
    if (index < 0 || index >= outputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    return outputs_[index]->SetExternalObject(object);
  }

  Status Run() override {
    for (auto& obj : inputs_) {
      RETURN_IF_ERROR(obj->CopyFromExternalObject());
    }
    RETURN_IF_ERROR(runtime_->Execute());
    for (auto& obj : outputs_) {
      RETURN_IF_ERROR(obj->CopyToExternalObject());
    }
    RETURN_IF_ERROR(runtime_->command_queue()->Flush());
    if (output_to_cpu_) {
      RETURN_IF_ERROR(runtime_->command_queue()->WaitForCompletion());
    }
    return OkStatus();
  }

 private:
  Status LinkTensors(const std::vector<TensorTieDef>& defs,
                     TensorTieFactory* tie_factory,
                     std::vector<std::unique_ptr<TensorTie>>* objects) {
    objects->reserve(defs.size());
    for (auto& def : defs) {
      std::unique_ptr<TensorTie> object;
      RETURN_IF_ERROR(tie_factory->NewTensorTie(def, objects_.get(), &object));
      objects->push_back(std::move(object));
    }
    return OkStatus();
  }

  static std::vector<TensorObjectDef> GetExternalDefinitions(
      const std::vector<std::unique_ptr<TensorTie>>& objects) {
    std::vector<TensorObjectDef> defs;
    defs.reserve(objects.size());
    for (auto& obj : objects) {
      defs.push_back(obj->def().external_def);
    }
    return defs;
  }

  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<ObjectManager> objects_;
  std::vector<std::unique_ptr<TensorTie>> inputs_;
  std::vector<std::unique_ptr<TensorTie>> outputs_;
  bool output_to_cpu_ = false;
};

class InferenceBuilderImpl : public InferenceBuilder {
 public:
  InferenceBuilderImpl(const InferenceEnvironmentOptions& env_options,
                       const InferenceOptions& options, GraphFloat32 graph,
                       const GpuInfo* gpu_info)
      : env_options_(env_options),
        options_(options),
        graph_(std::move(graph)),
        gpu_info_(gpu_info),
        tie_factory_(env_options_) {}

  Status Initialize() {
    inputs_ = LinkTensors(graph_.inputs());
    outputs_ = LinkTensors(graph_.outputs());
    return OkStatus();
  }

  std::vector<TensorObjectDef> inputs() const final {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const final {
    return GetExternalDefinitions(outputs_);
  }

  Status SetInputShape(int index, const Dimensions& dimensions) final {
    if (index < 0 || index >= inputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    return UnimplementedError("Changing input shapes is not supported");
  }

  Status SetInputObjectDef(int index, ObjectDef new_def) final {
    if (index < 0 || index >= inputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    auto def = inputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_.IsSupported(def)) {
      return InvalidArgumentError("New object definition is not supported.");
    }
    inputs_[index] = def;
    return OkStatus();
  }

  Status SetOutputObjectDef(int index, ObjectDef new_def) final {
    if (index < 0 || index >= outputs_.size()) {
      return OutOfRangeError("Index is out of range");
    }
    auto def = outputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_.IsSupported(def)) {
      return InvalidArgumentError("New object definition is not supported.");
    }
    outputs_[index] = def;
    return OkStatus();
  }

  Status Build(std::unique_ptr<InferenceRunner>* runner) final {
    auto kernels = NewNodeShaderRegistry();
    CompilationOptions compiler_options;
    compiler_options.allow_precision_loss =
        GetPosition(options_, InferencePriority::MAX_PRECISION) > 1;
    compiler_options.inline_parameters =
        options_.usage == InferenceUsage::SUSTAINED_SPEED &&
        GetPosition(options_, InferencePriority::MIN_LATENCY) == 1;
    if (GetRelativeImportance(options_, InferencePriority::MIN_MEMORY_USAGE,
                              InferencePriority::MIN_LATENCY) ==
        PriorityImportance::HIGHER) {
      // Buffers have far better memory utilization.
      compiler_options.preferred_obj_type = ObjectType::BUFFER;
      compiler_options.ref_obj_type = ObjectType::BUFFER;
    }

    auto compiler = NewCompiler(kernels.get(), gpu_info_, compiler_options);
    auto workgroup_calculator = NewDefaultWorkgroupsCalculator(*gpu_info_);
    auto external_objects = absl::make_unique<ObjectManager>();
    std::vector<GlShader> shaders;
    std::unordered_map<std::string, size_t> shader_to_index;
    RuntimeOptions runtime_options;
    auto runtime =
        absl::make_unique<Runtime>(runtime_options, *gpu_info_,
                                   env_options_.queue, external_objects.get());
    Runtime* runtime_ptr = runtime.get();
    auto runner_impl = absl::make_unique<InferenceRunnerImpl>(
        std::move(runtime), std::move(external_objects));
    RETURN_IF_ERROR(runner_impl->Initialize(inputs_, outputs_, &tie_factory_));
    RETURN_IF_ERROR(
        compiler->Compile(graph_, {}, [&](ShaderCode code) -> Status {
          auto workgroup = workgroup_calculator->Calculate(code);
          size_t shader_index;
          std::string shader_src =
              GetShaderHeader(workgroup) + code.source_code;
          // Check if a shader was already compiled.
          auto it = shader_to_index.find(shader_src);
          if (it == shader_to_index.end()) {
            GlShader shader;
            RETURN_IF_ERROR(GlShader::CompileShader(GL_COMPUTE_SHADER,
                                                    shader_src, &shader));
            shaders.push_back(std::move(shader));
            shader_to_index.insert({shader_src, shader_to_index.size()});
            shader_index = shader_to_index.size() - 1;
          } else {
            shader_index = it->second;
          }
          auto num_workgroups = IntegralDivideRoundUp(code.workload, workgroup);
          return runtime_ptr->AddProgram(shaders[shader_index], code.parameters,
                                         code.objects, num_workgroups);
        }));
    RETURN_IF_ERROR(runtime_ptr->PrepareForExecution());
    *runner = std::move(runner_impl);
    return OkStatus();
  }

 private:
  // Links internal tensors with external user-facing objects.
  std::vector<TensorTieDef> LinkTensors(
      const std::vector<Value<TensorRef<BHWC>>*>& values) {
    std::vector<TensorTieDef> links;
    links.reserve(values.size());
    for (const auto& value : values) {
      TensorObjectDef def;
      // So far the compiler always forces inputs and outputs to be in the fixed
      // format.
      const auto& shape = value->tensor.shape;
      def.dimensions = Dimensions(shape.b, shape.h, shape.w, shape.c);
      def.object_def.data_type = DataType::FLOAT32;
      def.object_def.data_layout = DataLayout::DHWC4;
      def.object_def.object_type = gpu::ObjectType::OPENGL_SSBO;
      def.object_def.user_provided = true;
      AccessType access =
          graph_.IsGraphInput(value->id) ? AccessType::READ : AccessType::WRITE;
      links.push_back({value->id, access, def, def});
    }
    return links;
  }

  static std::vector<TensorObjectDef> GetExternalDefinitions(
      const std::vector<TensorTieDef>& links) {
    std::vector<TensorObjectDef> defs;
    defs.reserve(links.size());
    for (auto& desc : links) {
      defs.push_back(desc.external_def);
    }
    return defs;
  }

  const InferenceEnvironmentOptions env_options_;
  const InferenceOptions options_;
  GraphFloat32 graph_;
  const GpuInfo* gpu_info_;
  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
  TensorTieFactory tie_factory_;
};

class InferenceEnvironmentImpl : public InferenceEnvironment {
 public:
  explicit InferenceEnvironmentImpl(const InferenceEnvironmentOptions& options)
      : env_options_(options) {}

  Status Init() {
    RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&egl_env_));

    RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
    properties_.is_opengl_available = IsOpenGl31OrAbove(gpu_info_);
    if (!properties_.is_opengl_available) {
      return InternalError(
          "OpenGL ES 3.1 or above is required to use OpenGL inference.");
    }
    if (!env_options_.queue) {
      queue_ = NewCommandQueue(gpu_info_);
      env_options_.queue = queue_.get();
    }
    return OkStatus();
  }

  Status NewInferenceBuilder(GraphFloat32&& model,
                             const InferenceOptions& options,
                             std::unique_ptr<InferenceBuilder>* builder) final {
    if (!IsValid(options)) {
      return InvalidArgumentError("InferenceOptions are invalid.");
    }
    InferenceOptions resolved_options = options;
    ResolveAutoPriority(&resolved_options);
    if (!IsBatchMatchesForAllValues(model)) {
      return InvalidArgumentError(
          "Only identical batch dimension is supported");
    }
    auto builder_impl = absl::make_unique<InferenceBuilderImpl>(
        env_options_, resolved_options, std::move(model), &gpu_info_);
    RETURN_IF_ERROR(builder_impl->Initialize());
    *builder = std::move(builder_impl);
    return OkStatus();
  }

  const InferenceEnvironmentProperties& properties() const {
    return properties_;
  }

 private:
  std::unique_ptr<EglEnvironment> egl_env_;
  std::unique_ptr<CommandQueue> queue_;
  InferenceEnvironmentOptions env_options_;
  GpuInfo gpu_info_;
  InferenceEnvironmentProperties properties_;
};

}  // namespace

Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties) {
  auto env_impl = absl::make_unique<InferenceEnvironmentImpl>(options);
  Status status = env_impl->Init();
  if (properties) {
    *properties = env_impl->properties();
  }
  RETURN_IF_ERROR(status);
  *environment = std::move(env_impl);
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
