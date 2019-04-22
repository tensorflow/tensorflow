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

#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/general_transformations.h"
#include "tensorflow/lite/delegates/gpu/gl/api.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/bhwc_to_phwc4.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/phwc4_to_bhwc.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/registry.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/best_effort_calculator.h"
#include "tensorflow/lite/minimal_logging.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "flatbuffers/flatbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/delegates/gpu/gl/metadata_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Forward declarations.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);
TfLiteStatus DelegateCopyFromBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle,  // ValueId
    TfLiteTensor* tensor);
TfLiteStatus DelegateCopyToBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle,  // ValueId
    TfLiteTensor* tensor);

inline bool IsPHWC4(const BHWC& shape) { return shape.c == 4; }

class Delegate {
  struct ValueRef {
    BHWC shape;
    int tensor_index;
  };

 public:
  explicit Delegate(const TfLiteGpuDelegateOptions* options) {
    if (options) {
      options_ = *options;
    } else {
      // Default options.
      options_.metadata = nullptr;
      options_.compile_options.precision_loss_allowed = 0;
      options_.compile_options.preferred_gl_object_type =
          TFLITE_GL_OBJECT_TYPE_FASTEST;
      options_.compile_options.dynamic_batch_enabled = 0;
    }
  }

  Status CopyFromBufferHandle(TfLiteBufferHandle handle, TfLiteTensor* tensor) {
    ValueRef ref;
    RETURN_IF_ERROR(FindObject(handle, &ref));
    auto buffer = phwc4_objects_.FindBuffer(handle);
    return buffer->MappedRead<float>([&](absl::Span<const float> data) {
      tensor->data_is_stale = false;
      return ConvertFromPHWC4(
          data, ref.shape,
          absl::MakeSpan(tensor->data.f, tensor->bytes / sizeof(float)));
    });
  }

  Status CopyToBufferHandle(TfLiteBufferHandle handle,
                            TfLiteTensor* tensor) const {
    ValueRef ref;
    RETURN_IF_ERROR(FindObject(handle, &ref));
    auto buffer = phwc4_objects_.FindBuffer(handle);
    return buffer->MappedWrite<float>([&](absl::Span<float> data) {
      return ConvertToPHWC4(
          absl::MakeConstSpan(tensor->data.f, tensor->bytes / sizeof(float)),
          ref.shape, data);
    });
  }

  Status BindBufferToTensor(GLuint ssbo, int tensor_index) {
    int64_t bytes_size;
    {
      gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, ssbo);
      RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetBufferParameteri64v,
                                         GL_SHADER_STORAGE_BUFFER,
                                         GL_BUFFER_SIZE, &bytes_size));
    }
    return bhwc_objects_.RegisterBuffer(
        tensor_index, GlBuffer(GL_SHADER_STORAGE_BUFFER, ssbo, bytes_size,
                               /* offset = */ 0,
                               /* has_ownership = */ false));
  }

  Status Prepare(TfLiteContext* context,
                 const TfLiteDelegateParams* delegate_params) {
    // Extract TFLite delegate execution plan from the context and convert it
    // into FlowGraph32.
    GraphFloat32 graph;
    RETURN_IF_ERROR(BuildModel(context, delegate_params, &graph));

    // Apply general transformations on the graph.
    NullTransformationReporter reporter;
    ModelTransformer transformer(&graph, &reporter);
    if (!ApplyGeneralTransformations(&transformer)) {
      return InternalError("Graph general transformations failed");
    }

    if (!env_) RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env_));

    // TODO(impjdi): Remove code duplication.
    auto values = graph.values();
    tensors_.reserve(values.back()->id + 1);
    for (auto value : values) {
      if (tensors_.size() <= value->id) {
        tensors_.resize(value->id + 1);
      }
      tensors_[value->id] = {value->tensor.shape, 0};
    }

    // Prepare graph inputs.
    {
      inputs_.reserve(delegate_params->input_tensors->size);
      for (auto input : graph.inputs()) {
        auto tensor_index = input->tensor.ref;
        auto& tensor = context->tensors[tensor_index];

        inputs_.push_back(input->id);
        tensor.buffer_handle = input->id;
        tensor.delegate = &delegate_;
        tensors_[input->id].tensor_index = tensor_index;

        // Create phwc4 input buffer.
        // Check whether there is externally provided object is already in
        // PHWC4. If yes, we may skip conversion step.
        // We need to keep same buffer in bhwc_objects_ to indicate there is
        // externally provided buffer.
        auto external_buffer = bhwc_objects_.FindBuffer(tensor_index);
        GlBuffer buffer;
        if (IsPHWC4(input->tensor.shape) && external_buffer) {
          buffer = external_buffer->MakeRef();
        } else {
          RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
              GetElementsSizeForPHWC4(input->tensor.shape), &buffer));
        }
        RETURN_IF_ERROR(
            phwc4_objects_.RegisterBuffer(input->id, std::move(buffer)));
      }
    }

    // Prepare graph outputs.
    {
      outputs_.reserve(delegate_params->output_tensors->size);
      for (auto output : graph.outputs()) {
        auto tensor_index = output->tensor.ref;
        auto& tensor = context->tensors[tensor_index];

        outputs_.push_back(output->id);
        tensor.buffer_handle = output->id;
        tensor.delegate = &delegate_;
        tensors_[output->id].tensor_index = tensor_index;

        // Create phwc4 output buffer.
        // Check whether there is externally provided object is already in
        // PHWC4. If yes, we may skip conversion step.
        auto external_buffer = bhwc_objects_.FindBuffer(tensor_index);
        GlBuffer buffer;
        if (IsPHWC4(output->tensor.shape) && external_buffer) {
          buffer = external_buffer->MakeRef();
        } else {
          RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
              GetElementsSizeForPHWC4(output->tensor.shape), &buffer));
        }
        RETURN_IF_ERROR(
            phwc4_objects_.RegisterBuffer(output->id, std::move(buffer)));
      }
    }

    // Create shaders to convert from/to phwc4.
    RETURN_IF_ERROR(ConverterBhwcToPhwc4::Create(&bhwc_to_phwc4_));
    RETURN_IF_ERROR(ConverterPhwc4ToBhwc::Create(&phwc4_to_bhwc_));

    // Compile model.
    CompilationOptions compile_options;
    compile_options.allow_precision_loss =
        static_cast<bool>(options_.compile_options.precision_loss_allowed);
    compile_options.preferred_obj_type = static_cast<ObjectType>(
        options_.compile_options.preferred_gl_object_type);
    compile_options.ref_obj_type = static_cast<ObjectType>(
        options_.compile_options.preferred_gl_object_type);
    compile_options.dynamic_batch =
        static_cast<bool>(options_.compile_options.dynamic_batch_enabled);
    auto shaders = NewNodeShaderRegistry();
    GpuInfo gpu_info;
    RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
    command_queue_ = NewCommandQueue(gpu_info);
    auto workgroups_calculator =
        BestEffortWorkgroupsCalculator(options_.metadata, gpu_info);
    std::unique_ptr<CompiledModel> compiled_model;
    RETURN_IF_ERROR(Compile(compile_options, graph, *shaders,
                            *workgroups_calculator, &compiled_model));

    // Create inference context.
    const RuntimeOptions runtime_options;
    RETURN_IF_ERROR(compiled_model->NewRun(runtime_options, &phwc4_objects_,
                                           command_queue_.get(),
                                           &inference_context_));
    return OkStatus();
  }

  Status Invoke(TfLiteContext* context) {
    const EGLContext egl_context_at_delegate_init = env_->context().context();
    const EGLContext egl_context_at_delegate_invoke = eglGetCurrentContext();
    if (egl_context_at_delegate_init != egl_context_at_delegate_invoke) {
      return FailedPreconditionError(
          "Delegate should run on the same thread where it was initialized.");
    }

    // Push input data from a tensor to GPU.
    for (ValueId id : inputs_) {
      const ValueRef& ref = tensors_[id];
      auto external_object = bhwc_objects_.FindBuffer(ref.tensor_index);
      if (external_object) {
        // Use input from GPU.
        // Conversion is needed only when external object is not phwc4.
        if (!IsPHWC4(tensors_[id].shape)) {
          RETURN_IF_ERROR(bhwc_to_phwc4_.Convert(
              ref.shape, *external_object, command_queue_.get(),
              phwc4_objects_.FindBuffer(id)));
        }
      } else {
        // Copy from CPU to GPU
        TfLiteTensor& tensor = context->tensors[ref.tensor_index];
        RETURN_IF_ERROR(CopyToBufferHandle(id, &tensor));
      }
    }

    // Run inference.
    RETURN_IF_ERROR(inference_context_->Reset());
    RETURN_IF_ERROR(inference_context_->Execute());

    // Push output data from GPU to a tensor.
    bool finished_gpu_processing = false;
    for (ValueId id : outputs_) {
      const ValueRef& ref = tensors_[id];
      auto external_object = bhwc_objects_.FindBuffer(ref.tensor_index);
      if (external_object) {
        // Convert data from PHWC4 to BHWC and leave it in GPU object.
        // Conversion is needed only when external object is not phwc4.
        if (!IsPHWC4(tensors_[id].shape)) {
          RETURN_IF_ERROR(
              phwc4_to_bhwc_.Convert(ref.shape, *phwc4_objects_.FindBuffer(id),
                                     command_queue_.get(), external_object));
        }
      } else {
        // Wait until all GPU command are completed. This call leads to a lower
        // processing latency because a buffer reading below will not stall if
        // data is not yet ready.
        if (!finished_gpu_processing) {
          RETURN_IF_ERROR(command_queue_->WaitForCompletion());
          finished_gpu_processing = true;
        }
        // Copy from GPU to CPU.
        TfLiteTensor& tensor = context->tensors[ref.tensor_index];
        RETURN_IF_ERROR(CopyFromBufferHandle(id, &tensor));
      }
    }
    return OkStatus();
  }

  TfLiteDelegate* tflite_delegate() { return &delegate_; }

 private:
  Status FindObject(ValueId id, ValueRef* ref) const {
    if (id >= tensors_.size()) {
      return InvalidArgumentError("Invalid buffer id");
    }
    *ref = tensors_[id];
    return OkStatus();
  }

  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      DelegateCopyFromBufferHandle,   // .CopyFromBufferHandle
      DelegateCopyToBufferHandle,     // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  TfLiteGpuDelegateOptions options_;

  std::unique_ptr<EglEnvironment> env_;
  std::vector<ValueRef> tensors_;  // indexed by ValueId
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
  ObjectManager phwc4_objects_;
  ObjectManager bhwc_objects_;  // key is tensor_index
  ConverterPhwc4ToBhwc phwc4_to_bhwc_;
  ConverterBhwcToPhwc4 bhwc_to_phwc4_;
  std::unique_ptr<CommandQueue> command_queue_;
  std::unique_ptr<InferenceContext> inference_context_;
};

// TODO(impjdi): Merge with MetalDelegate.
bool IsAllFloatTensors(const TfLiteContext* context,
                       const TfLiteIntArray* array) {
  for (int i = 0; i < array->size; ++i) {
    const TfLiteTensor* t = context->tensors + array->data[i];
    if (t->allocation_type == kTfLiteArenaRw && t->type != kTfLiteFloat32) {
      return false;
    }
  }
  return true;
}

inline Delegate* GetGpuDelegate(TfLiteNode* node) {
  return reinterpret_cast<Delegate*>(node->user_data);
}

inline Delegate* GetGpuDelegate(TfLiteDelegate* delegate) {
  return reinterpret_cast<Delegate*>(delegate->data_);
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const TfLiteRegistration kRegistration = {
      // .init
      [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* gpu_delegate = GetGpuDelegate(params->delegate);
        // Everything below should happen in prepare function call, but TFLite
        // for whatever reason forbids that.
        const auto status = gpu_delegate->Prepare(context, params);
        if (status.ok()) return gpu_delegate;
        context->ReportError(context, "TfLiteGpuDelegate Prepare: %s",
                             status.error_message().c_str());
        return nullptr;
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {},
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetGpuDelegate(node)->Invoke(context);
        if (status.ok()) return kTfLiteOk;
        context->ReportError(context, "TfLiteGpuDelegate Invoke: %s",
                             status.error_message().c_str());
        return kTfLiteError;
      },
      nullptr,              // .profiling_string
      0,                    // .builtin_code
      "TfLiteGpuDelegate",  // .custom_name
      1,                    // .version
  };
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

TfLiteStatus DelegateCopyFromBufferHandle(TfLiteContext* context,
                                          TfLiteDelegate* delegate,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteTensor* tensor) {
  auto* gpu_delegate = GetGpuDelegate(delegate);
  if (!gpu_delegate) return kTfLiteError;
  const auto status = gpu_delegate->CopyFromBufferHandle(buffer_handle, tensor);
  if (status.ok()) return kTfLiteOk;
  context->ReportError(context, "TfLiteGpuDelegate CopyFromBufferHandle: %s",
                       status.error_message().c_str());
  return kTfLiteError;
}

TfLiteStatus DelegateCopyToBufferHandle(TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) {
  auto* gpu_delegate = GetGpuDelegate(delegate);
  if (!gpu_delegate) return kTfLiteError;
  const auto status = gpu_delegate->CopyToBufferHandle(buffer_handle, tensor);
  if (status.ok()) return kTfLiteOk;
  context->ReportError(context, "TfLiteGpuDelegate CopyToBufferHandle: %s",
                       status.error_message().c_str());
  return kTfLiteError;
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

TfLiteDelegate* TfLiteGpuDelegateCreate(
    const TfLiteGpuDelegateOptions* options) {
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for GPU.");
  auto* gpu_delegate = new tflite::gpu::gl::Delegate(options);
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}

void TfLiteGpuDelegateDelete(TfLiteDelegate* delegate) {
  delete tflite::gpu::gl::GetGpuDelegate(delegate);
}

TfLiteStatus TfLiteGpuDelegateBindBufferToTensor(TfLiteDelegate* delegate,
                                                 GLuint buffer,
                                                 int tensor_index) {
  auto* gpu_delegate = tflite::gpu::gl::GetGpuDelegate(delegate);
  return gpu_delegate &&
                 gpu_delegate->BindBufferToTensor(buffer, tensor_index).ok()
             ? kTfLiteOk
             : kTfLiteError;
}

#ifndef TFLITE_GPU_BINARY_RELEASE
const uint8_t* TfLiteGpuDelegateGetModelMetadata(const void* tflite_model) {
  const auto* model = reinterpret_cast<const tflite::Model*>(tflite_model);
  if (!model || !model->metadata_buffer() || !model->buffers()) return nullptr;
  for (int32_t buffer_index : *model->metadata_buffer()) {
    if (buffer_index < 0 && buffer_index >= model->buffers()->size()) continue;
    const tflite::Buffer* buffer = model->buffers()->Get(buffer_index);
    if (!buffer) continue;
    const uint8_t* data = buffer->data()->data();
    if (!flatbuffers::BufferHasIdentifier(
            data, tflite::gpu::gl::data::FlowMetadataIdentifier())) {
      continue;
    }
    flatbuffers::Verifier verifier(data, buffer->data()->size());
    return tflite::gpu::gl::data::VerifyFlowMetadataBuffer(verifier) ? data
                                                                     : nullptr;
  }
  return nullptr;
}
#endif  // TFLITE_GPU_BINARY_RELEASE
