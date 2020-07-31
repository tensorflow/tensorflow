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

#include "tensorflow/lite/delegates/gpu/cl/gpu_api_delegate.h"

#include <cstdint>

#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/general_transformations.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Forward declarations.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

InferencePriority ToPriority(int32_t priority) {
  switch (priority) {
    case TfLiteGpuInferencePriority::
        TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:
      return InferencePriority::MAX_PRECISION;
    case TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:
      return InferencePriority::MIN_LATENCY;
  }
  return InferencePriority::MAX_PRECISION;
}

DataType ToDataType(TfLiteType data_type) {
  switch (data_type) {
    case kTfLiteFloat16:
      return DataType::FLOAT16;
    case kTfLiteFloat32:
      return DataType::FLOAT32;
    default:
      return DataType::UNKNOWN;
  }
}

DataLayout ToDataLayoutFromTFL(TfLiteGpuDataLayout data_layout) {
  switch (data_layout) {
    case TFLITE_GPU_DATA_LAYOUT_BHWC:
      return DataLayout::BHWC;
    case TFLITE_GPU_DATA_LAYOUT_DHWC4:
      return DataLayout::DHWC4;
    default:
      return DataLayout::UNKNOWN;
  }
}

class Delegate {
 public:
  explicit Delegate(const TfLiteGpuDelegateOptions_New* options) {
    if (options) {
      options_ = *options;
    } else {
      // Default options.
      options_.compile_options.precision_loss_allowed = 0;
      options_.compile_options.inference_priority = TfLiteGpuInferencePriority::
          TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
      options_.egl_display = EGL_NO_DISPLAY;
      options_.egl_context = EGL_NO_CONTEXT;
      options_.serialized_binary_cache_data = nullptr;
      options_.serialized_binary_cache_size = 0;
    }
  }

  absl::Status Prepare(TfLiteContext* context,
                       const TfLiteDelegateParams* delegate_params) {
    // Extract TFLite delegate execution plan from the context and convert it
    // into GraphFloat32.
    GraphFloat32 graph;
    RETURN_IF_ERROR(BuildModel(context, delegate_params, &graph));

    // Apply general transformations on the graph.
    NullTransformationReporter reporter;
    ModelTransformer transformer(&graph, &reporter);
    if (!ApplyGeneralTransformations(&transformer)) {
      return absl::InternalError("Graph general transformations failed");
    }

    InferenceEnvironmentOptions env_options;
    env_options.egl_context = options_.egl_context;
    env_options.egl_display = options_.egl_display;
    env_options.serialized_binary_cache = {
        options_.serialized_binary_cache_data,
        options_.serialized_binary_cache_size};
    InferenceEnvironmentProperties properties;
    absl::Status status =
        NewInferenceEnvironment(env_options, &environment_, &properties);
    if (!properties.is_opencl_available) {
      TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate: OpenCL is not available");
    }
    if (!properties.is_gl_sharing_supported) {
      TF_LITE_KERNEL_LOG(context,
                         "TfLiteGpuDelegate: GL sharing is not supported");
    }
    if (!properties.is_cl_to_gl_fast_sync_supported) {
      TF_LITE_KERNEL_LOG(
          context, "TfLiteGpuDelegate: fast CL to GL sync is not supported");
    }
    if (!properties.is_gl_to_cl_fast_sync_supported) {
      TF_LITE_KERNEL_LOG(
          context, "TfLiteGpuDelegate: fast GL to CL sync is not supported");
    }
    RETURN_IF_ERROR(status);

    std::vector<uint32_t> input_refs;
    {
      const auto& inputs = graph.inputs();
      input_refs.reserve(inputs.size());
      for (auto input : inputs) {
        input_refs.push_back(input->tensor.ref);
      }
    }
    std::vector<uint32_t> output_refs;
    {
      const auto& outputs = graph.outputs();
      output_refs.reserve(outputs.size());
      for (auto output : outputs) {
        output_refs.push_back(output->tensor.ref);
      }
    }

    InferenceOptions options;
    options.usage = InferenceUsage::FAST_SINGLE_ANSWER;
    if (options_.compile_options.precision_loss_allowed == 0) {
      options.priority1 = InferencePriority::MAX_PRECISION;
      switch (options_.compile_options.inference_priority) {
        case TfLiteGpuInferencePriority::
            TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:
          options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
          options.priority3 = InferencePriority::MIN_LATENCY;
          break;
        case TfLiteGpuInferencePriority::
            TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:
          options.priority2 = InferencePriority::MIN_LATENCY;
          options.priority3 = InferencePriority::MIN_MEMORY_USAGE;
          break;
        case TfLiteGpuInferencePriority::
            TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE:
          options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
          options.priority3 = InferencePriority::MIN_LATENCY;
          break;
      }
    } else {
      switch (options_.compile_options.inference_priority) {
        case TfLiteGpuInferencePriority::
            TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:
          options.priority1 = InferencePriority::MIN_LATENCY;
          options.priority2 = InferencePriority::MAX_PRECISION;
          options.priority3 = InferencePriority::MIN_MEMORY_USAGE;
          break;
        case TfLiteGpuInferencePriority::
            TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:
          options.priority1 = InferencePriority::MIN_LATENCY;
          options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
          options.priority3 = InferencePriority::MAX_PRECISION;
          break;
        case TfLiteGpuInferencePriority::
            TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE:
          options.priority1 = InferencePriority::MIN_MEMORY_USAGE;
          options.priority2 = InferencePriority::MIN_LATENCY;
          options.priority3 = InferencePriority::MAX_PRECISION;
          break;
      }
    }
    std::unique_ptr<InferenceBuilder> builder;
    RETURN_IF_ERROR(
        environment_->NewInferenceBuilder(options, std::move(graph), &builder));

    // At this point tflite didn't allocate tensors yet, therefore, collect
    // indices and set all input and output tensors from tflite later.
    input_indices_.reserve(input_refs.size());
    for (auto tensor_index : input_refs) {
      int object_index = input_indices_.size();
      input_indices_.push_back(tensor_index);
      RETURN_IF_ERROR(
          builder->SetInputObjectDef(object_index, GetObjectDef(tensor_index)));
    }
    output_indices_.reserve(output_refs.size());
    for (auto tensor_index : output_refs) {
      int object_index = output_indices_.size();
      output_indices_.push_back(tensor_index);
      RETURN_IF_ERROR(builder->SetOutputObjectDef(object_index,
                                                  GetObjectDef(tensor_index)));
    }

    return builder->Build(&runner_);
  }

  absl::Status SetInputsAndOutputs(TfLiteContext* context) {
    int i = 0;
    for (auto index : input_indices_) {
      RETURN_IF_ERROR(
          runner_->SetInputObject(i++, GetTensorObject(index, context)));
    }
    i = 0;
    for (auto index : output_indices_) {
      RETURN_IF_ERROR(
          runner_->SetOutputObject(i++, GetTensorObject(index, context)));
    }
    return absl::OkStatus();
  }

  absl::Status Invoke(TfLiteContext* context) {
    RETURN_IF_ERROR(SetInputsAndOutputs(context));
    return runner_->Run();
  }

  void BindGlBufferToTensor(GLuint buffer_id, int tensor_index,
                            DataType data_type, DataLayout data_layout) {
    // At this point the delegate haven't seen a model yet. Therefore, just
    // record what object gets assigned.
    if (tensor_index >= tensors_.size()) {
      tensors_.resize(tensor_index + 1);
    }
    TensorObjectDef def;
    def.object_def.data_type = data_type;
    def.object_def.data_layout = data_layout;
    def.object_def.object_type = ObjectType::OPENGL_SSBO;
    def.object_def.user_provided = true;
    def.dimensions = Dimensions(0, 0, 0, 0);
    OpenGlBuffer buffer;
    buffer.id = buffer_id;
    TensorObject obj = buffer;
    tensors_[tensor_index] = std::make_pair(obj, def);
  }

  ObjectDef GetObjectDef(int index) const {
    if (index < tensors_.size() && IsValid(tensors_[index].second)) {
      return tensors_[index].second.object_def;
    }
    ObjectDef default_object_def;
    default_object_def.data_type = DataType::FLOAT32;
    default_object_def.data_layout = DataLayout::BHWC;
    default_object_def.object_type = ObjectType::CPU_MEMORY;
    default_object_def.user_provided = true;
    return default_object_def;
  }

  TensorObject GetTensorObject(int index, TfLiteContext* context) const {
    if (index < tensors_.size() &&
        IsValid(tensors_[index].second, tensors_[index].first)) {
      return tensors_[index].first;
    }
    auto& tensor = context->tensors[index];
    return MakeCpuMemory(absl::MakeSpan(tensor.data.raw, tensor.bytes));
  }

  TfLiteDelegate* tflite_delegate() { return &delegate_; }

  bool SupportsGlObjects() const {
    return options_.egl_context != EGL_NO_CONTEXT &&
           options_.egl_display != EGL_NO_DISPLAY;
  }

  absl::Span<const uint8_t> GetSerializedBinaryCache() {
    binary_cache_ = environment_->GetSerializedBinaryCache();
    return binary_cache_;
  }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  TfLiteGpuDelegateOptions_New options_;
  std::unique_ptr<InferenceEnvironment> environment_;
  std::unique_ptr<InferenceRunner> runner_;
  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;
  std::vector<uint8_t> binary_cache_;
  std::vector<std::pair<TensorObject, TensorObjectDef>> tensors_;
};

inline Delegate* GetDelegate(TfLiteNode* node) {
  return reinterpret_cast<Delegate*>(node->user_data);
}

inline Delegate* GetDelegate(TfLiteDelegate* delegate) {
  return reinterpret_cast<Delegate*>(delegate->data_);
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const TfLiteRegistration kRegistration = {
      // .init
      [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* gpu_delegate = GetDelegate(params->delegate);
        // Everything below should happen in prepare function call, but TFLite
        // for whatever reason forbids that.
        const auto status = gpu_delegate->Prepare(context, params);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Init: %s",
                             std::string(status.message()).c_str());
          return nullptr;
        }
        return gpu_delegate;
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {},
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (!node->user_data) {
          TF_LITE_KERNEL_LOG(
              context,
              "TfLiteGpuDelegate Prepare: delegate is not initialized");
          return kTfLiteError;
        }
        // TODO(akulik): tflite tensors are not allocated here either. It would
        // be good to set inputs and outputs only once here instead of setting
        // them every time in .invoke.
        return kTfLiteOk;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetDelegate(node)->Invoke(context);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Invoke: %s",
                             std::string(status.message()).c_str());
          return kTfLiteError;
        }
        return kTfLiteOk;
      },
      nullptr,                  // .profiling_string
      0,                        // .builtin_code
      "TfLiteGpuDelegate_New",  // .custom_name
      1,                        // .version
  };
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

TfLiteDelegate* TfLiteGpuDelegateCreate_New(
    const TfLiteGpuDelegateOptions_New* options) {
  auto* gpu_delegate = new tflite::gpu::cl::Delegate(options);
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}

void TfLiteGpuDelegateDelete_New(TfLiteDelegate* delegate) {
  delete tflite::gpu::cl::GetDelegate(delegate);
}

TFL_CAPI_EXPORT TfLiteStatus TfLiteGpuDelegateBindGlBufferToTensor(
    TfLiteDelegate* delegate, GLuint buffer_id, int tensor_index,
    TfLiteType data_type, TfLiteGpuDataLayout data_layout) {
  auto* gpu_delegate = tflite::gpu::cl::GetDelegate(delegate);
  if (!gpu_delegate) {
    return kTfLiteError;
  }
  if (!gpu_delegate->SupportsGlObjects()) {
    return kTfLiteError;
  }
  auto type = tflite::gpu::cl::ToDataType(data_type);
  if (type == tflite::gpu::DataType::UNKNOWN) {
    return kTfLiteError;
  }
  auto layout = tflite::gpu::cl::ToDataLayoutFromTFL(data_layout);
  if (layout == tflite::gpu::DataLayout::UNKNOWN) {
    return kTfLiteError;
  }
  gpu_delegate->BindGlBufferToTensor(buffer_id, tensor_index, type, layout);
  return kTfLiteOk;
}

bool TfLiteGpuDelegateGetSerializedBinaryCache(TfLiteDelegate* delegate,
                                               size_t* size,
                                               const uint8_t** data) {
  *size = 0;
  auto* gpu_delegate = tflite::gpu::cl::GetDelegate(delegate);
  if (!gpu_delegate) {
    return false;
  }
  auto cache = gpu_delegate->GetSerializedBinaryCache();
  if (cache.empty()) {
    return false;
  }
  *size = cache.size();
  *data = cache.data();
  return true;
}
