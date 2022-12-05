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

#include "tensorflow/lite/delegates/gpu/delegate.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/quantization_util.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"

#ifndef CL_DELEGATE_NO_GL
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#endif

namespace tflite {
namespace gpu {
namespace {

using delegates::Serialization;
using delegates::SerializationParams;

constexpr char kSerializedDataPrefix[] = "gpuv2_data_";

InferencePriority ToPriority(int32_t priority) {
  switch (priority) {
    case TFLITE_GPU_INFERENCE_PRIORITY_AUTO:
      return InferencePriority::AUTO;
    case TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:
      return InferencePriority::MAX_PRECISION;
    case TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:
      return InferencePriority::MIN_LATENCY;
    case TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE:
      return InferencePriority::MIN_MEMORY_USAGE;
  }
  return InferencePriority::UNKNOWN;
}

InferenceUsage ToUsage(int32_t usage) {
  switch (usage) {
    case TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return InferenceUsage::FAST_SINGLE_ANSWER;
    case TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return InferenceUsage::SUSTAINED_SPEED;
    case TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED:
      return InferenceUsage::BALANCED;
  }
  return InferenceUsage::UNKNOWN;
}

// Forward declarations.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
 public:
  explicit Delegate(const TfLiteGpuDelegateOptionsV2* options)
      : num_delegate_kernels_(0) {
    delegate_.data_ = reinterpret_cast<void*>(this);
    delegate_.Prepare = DelegatePrepare;
    delegate_.CopyFromBufferHandle = nullptr;
    delegate_.CopyToBufferHandle = nullptr;
    delegate_.FreeBufferHandle = nullptr;
    delegate_.flags = kTfLiteDelegateFlagsNone;
    options_ = options ? *options : TfLiteGpuDelegateOptionsV2Default();
    if (options_.max_delegated_partitions <= 0) {
      options_.max_delegated_partitions = 1;
    }
    if (options_.experimental_flags &
            TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION &&
        options_.model_token && options_.serialization_dir) {
      SerializationParams params;
      params.model_token = options_.model_token;
      params.cache_dir = options_.serialization_dir;
      serialization_ = std::make_unique<Serialization>(params);
    }
  }

  TfLiteDelegate* tflite_delegate() { return &delegate_; }
  Serialization* serialization() { return serialization_.get(); }
  const TfLiteGpuDelegateOptionsV2& options() const { return options_; }

  bool IsQuantOpsAllowed() const {
    return options_.experimental_flags &
           TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  }
  int MaxDelegatedPartitions() const {
    return options_.max_delegated_partitions;
  }
  int num_delegate_kernels() const { return num_delegate_kernels_; }

 private:
  TfLiteDelegate delegate_;
  TfLiteGpuDelegateOptionsV2 options_;
  int num_delegate_kernels_ = 0;

  std::unique_ptr<Serialization> serialization_;

  friend class DelegateKernel;
};

// Represent the execution of a subset of nodes on GPU.
class DelegateKernel {
 public:
  explicit DelegateKernel(Delegate* delegate) : delegate_(delegate) {
    ++delegate_->num_delegate_kernels_;
  }
  ~DelegateKernel() { --delegate_->num_delegate_kernels_; }

  absl::Status Prepare(TfLiteContext* context,
                       const TfLiteDelegateParams* delegate_params) {
    thread_id_prepare_ = std::this_thread::get_id();

    // Extract TFLite delegate execution plan from the context and convert it
    // into GraphFloat32.
    GraphFloat32 graph;
    std::vector<uint32_t> input_refs;
    std::vector<uint32_t> output_refs;
    RETURN_IF_ERROR(InitializeGraph(context, delegate_params, &graph,
                                    &input_refs, &output_refs));

    std::unique_ptr<InferenceBuilder> builder;
    bool graph_is_destroyed;
    const int experimental_flags = delegate_->options().experimental_flags;
    if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY) {
      RETURN_IF_ERROR(InitializeOpenClApi(&graph, &builder, &graph_is_destroyed,
                                          context, delegate_params,
                                          delegate_->serialization()));
    } else if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY) {
      RETURN_IF_ERROR(InitializeOpenGlApi(&graph, &builder));
    } else {
      // By default, we try CL first & fall back to GL if that fails.
      absl::Status status =
          InitializeOpenClApi(&graph, &builder, &graph_is_destroyed, context,
                              delegate_params, delegate_->serialization());
      if (!status.ok()) {
        TF_LITE_KERNEL_LOG(context, std::string(status.message()).c_str());
        TF_LITE_KERNEL_LOG(context, "Falling back to OpenGL");

        // Graph needs to be re-created because it is moved above.
        GraphFloat32 graph2;
        if (graph_is_destroyed) {
          RETURN_IF_ERROR(InitializeGraph(context, delegate_params, &graph2,
                                          &input_refs, &output_refs));
        }
        RETURN_IF_ERROR(InitializeOpenGlApi(
            graph_is_destroyed ? &graph2 : &graph, &builder));
      }
    }

    // At this point, TFLite hasn't allocated tensors yet, therefore, collect
    // indices and set all input and output tensors from TFLite later.
    input_indices_.reserve(input_refs.size());
    for (uint32_t tensor_index : input_refs) {
      const int64_t object_index = input_indices_.size();
      input_indices_.push_back(tensor_index);
      const TfLiteTensor& tflite_tensor = context->tensors[tensor_index];
      const DataType data_type = ToDataType(tflite_tensor.type);
      RETURN_IF_ERROR(builder->SetInputObjectDef(
          object_index, GetObjectDef(tensor_index, data_type)));
    }
    output_indices_.reserve(output_refs.size());
    for (uint32_t tensor_index : output_refs) {
      const int64_t object_index = output_indices_.size();
      output_indices_.push_back(tensor_index);
      const TfLiteTensor& tflite_tensor = context->tensors[tensor_index];
      const DataType data_type = ToDataType(tflite_tensor.type);
      RETURN_IF_ERROR(builder->SetOutputObjectDef(
          object_index, GetObjectDef(tensor_index, data_type)));
    }

    return builder->Build(&runner_);
  }

  // This directs the runtime to allocate memory for input/output temporary
  // tensors that require dequantization/quantization.
  absl::Status GetRequiredTemporaries(TfLiteContext* context, TfLiteNode* node,
                                      TfLiteIntArray** temporaries_array_ptr) {
    if (quant_conversion_map_.empty()) return absl::OkStatus();

    std::vector<int> temporary_tensors;
    for (auto index : input_indices_) {
      if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
        temporary_tensors.push_back(index);
      }
    }
    for (auto index : output_indices_) {
      if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
        temporary_tensors.push_back(index);
      }
    }
    *temporaries_array_ptr = TfLiteIntArrayCreate(temporary_tensors.size());
    for (int i = 0; i < temporary_tensors.size(); ++i) {
      (*temporaries_array_ptr)->data[i] = temporary_tensors[i];
    }
    return absl::OkStatus();
  }

  absl::Status Invoke(TfLiteContext* context) {
    if (thread_id_prepare_ != std::this_thread::get_id()) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "GpuDelegate invoke thread != prepare thread");
      if (enforce_same_thread_) {
        return absl::FailedPreconditionError(
            "GpuDelegate must run on the same thread where it was "
            "initialized.");
      }
    }

    const bool is_dequant_required = !quant_conversion_map_.empty();
    if (is_dequant_required) {
      RETURN_IF_ERROR(
          DequantizeInputs(context, input_indices_, quant_conversion_map_));
    }
    RETURN_IF_ERROR(SetInputsAndOutputs(context));
    RETURN_IF_ERROR(runner_->Run());
    if (is_dequant_required) {
      RETURN_IF_ERROR(
          QuantizeOutputs(context, output_indices_, quant_conversion_map_));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status SetInputsAndOutputs(TfLiteContext* context) {
    for (int i = 0; i < input_indices_.size(); ++i) {
      RETURN_IF_ERROR(runner_->SetInputObject(
          i, GetTensorObject(input_indices_[i], context)));
    }
    for (int i = 0; i < output_indices_.size(); ++i) {
      RETURN_IF_ERROR(runner_->SetOutputObject(
          i, GetTensorObject(output_indices_[i], context)));
    }
    return absl::OkStatus();
  }

  ObjectDef GetObjectDef(int index,
                         DataType data_type = DataType::FLOAT32) const {
    ObjectDef default_object_def;
    default_object_def.data_type = data_type;
    default_object_def.data_layout = DataLayout::BHWC;
    default_object_def.object_type = ObjectType::CPU_MEMORY;
    default_object_def.user_provided = true;
    return default_object_def;
  }

  TensorObject GetTensorObject(int index, TfLiteContext* context) const {
    auto& tensor = context->tensors[index];
    return MakeCpuMemory(absl::MakeSpan(tensor.data.raw, tensor.bytes));
  }

 private:
  absl::Status InitializeGraph(TfLiteContext* context,
                               const TfLiteDelegateParams* delegate_params,
                               GraphFloat32* graph,
                               std::vector<uint32_t>* input_refs,
                               std::vector<uint32_t>* output_refs) {
    quant_conversion_map_.clear();
    if (delegate_->IsQuantOpsAllowed()) {
      RETURN_IF_ERROR(BuildFinalModel(context, delegate_params, graph,
                                      &quant_conversion_map_));
    } else {
      RETURN_IF_ERROR(BuildFinalModel(context, delegate_params, graph));
    }

    // TfLiteDelegateParams.input_tensors is an array of all input tensors
    // including static weights.  GraphFloat32.inputs() is an array of runtime
    // tensors that don't have a producer and the order may not be the same as
    // defined by TfLiteDelegateParams.input_tensors.  These two sets are not
    // the same, especially on a multi-partition delegation.  These are matched
    // by filtering TfLiteDelegateParams.input_tensors with
    // !tflite::IsConstantTensor() and then inserting them in the order
    // specified by TfLiteDelegateParams.input_tensors.  This logic is shared
    // with ModelBuilder::PrecreateIOTensors() which is eventually called with
    // BuildFinalModel() above.
    //
    // Similarly, TfLiteDelegateParams.output_tensors is an array of all output
    // tensors, and can contain static tensors with buggy conversion.
    // GraphFloat32.outputs() is an array of runtime tensors that don't have a
    // consumer (this is a bug in the assumption) and the order may not be the
    // same as defined by TfLiteDelegateParams.output_tensors.  Again, these two
    // sets are not the same, especially on a multi-partition delegation.  These
    // are matched by inserting the tensors by the order defined by
    // TfLiteDelegateParams.output_tensors.  Similarly, this logic is shared
    // with ModelBuilder::PrecreateIOTensors() which is eventually called with
    // BuildFinalModel() above.
    //
    // The aforementioned matching in BuildFinalModel() is ported here to match
    // input/output_refs.
    // TODO(b/211393366): Fix this at GraphFloat32.inputs/outputs() level.
    const std::vector<Value*> inputs = graph->inputs();
    input_refs->clear();
    input_refs->reserve(delegate_params->input_tensors->size);
    for (int i = 0, j = 0; i < delegate_params->input_tensors->size; ++i) {
      const TfLiteTensor* tensor =
          context->tensors + delegate_params->input_tensors->data[i];
      if (tflite::IsConstantTensor(tensor)) continue;
      input_refs->push_back(inputs[j]->tensor.ref);
      ++j;
    }
    const std::vector<Value*> outputs = graph->outputs();
    output_refs->clear();
    const int output_size = std::min(static_cast<int>(graph->outputs().size()),
                                     delegate_params->output_tensors->size);
    output_refs->reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
      output_refs->push_back(outputs[i]->tensor.ref);
    }

    return absl::OkStatus();
  }

  absl::Status InitializeOpenClApi(GraphFloat32* graph,
                                   std::unique_ptr<InferenceBuilder>* builder,
                                   bool* graph_is_destroyed,
                                   TfLiteContext* context,
                                   const TfLiteDelegateParams* delegate_params,
                                   Serialization* serialization = nullptr) {
    *graph_is_destroyed = false;
    cl::InferenceEnvironmentOptions env_options;
    cl::InferenceEnvironmentProperties properties;

    // OpenCL initialization is parameterized by these InferenceOptions.
    auto delegate_options = delegate_->options();
    cl::InferenceOptions options;
    // If is_precision_loss_allowed == -1, then just use priorities instead
    // of paying attention to is_precision_loss_allowed value.
    if (delegate_options.is_precision_loss_allowed == -1) {
      options.priority1 = ToPriority(delegate_options.inference_priority1);
      options.priority2 = ToPriority(delegate_options.inference_priority2);
      options.priority3 = ToPriority(delegate_options.inference_priority3);
    } else {
      // Users set is_precision_loss_allowed explicitly, thus use it explicitly.
      if (delegate_options.is_precision_loss_allowed == 0) {
        options.priority1 = InferencePriority::MAX_PRECISION;
      } else {
        options.priority1 = InferencePriority::MIN_LATENCY;
      }
    }
    options.usage = ToUsage(delegate_options.inference_preference);

    if (!serialization) {
      // This path is faster when there is no serialization involved.
      RETURN_IF_ERROR(cl::NewInferenceEnvironment(env_options, &cl_environment_,
                                                  &properties));
      *graph_is_destroyed = true;
      RETURN_IF_ERROR(cl_environment_->NewInferenceBuilder(
          options, std::move(*graph), builder));
    } else {
      // If serialization data is found, initialize CL from it & return early.
      if (MaybeInitializeSerializedOpenCL(context, delegate_params, builder,
                                          &options, &env_options, &properties,
                                          serialization)
              .ok()) {
        return absl::OkStatus();
      }

      RETURN_IF_ERROR(cl::NewInferenceEnvironment(env_options, &cl_environment_,
                                                  &properties));
      *graph_is_destroyed = true;
      std::vector<uint8_t> serialized_model;
      RETURN_IF_ERROR(cl_environment_->BuildSerializedModel(
          options, std::move(*graph), &serialized_model));
      RETURN_IF_ERROR(
          cl_environment_->NewInferenceBuilder(serialized_model, builder));

      RETURN_IF_ERROR(SaveSerializedOpenCL(context, delegate_params, &options,
                                           serialization, serialized_model));
    }

    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Initialized OpenCL-based API.");
    return absl::OkStatus();
  }

  // Returns Ok only if serialized data is successsfully found.
  absl::Status MaybeInitializeSerializedOpenCL(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      std::unique_ptr<InferenceBuilder>* builder, cl::InferenceOptions* options,
      cl::InferenceEnvironmentOptions* env_options,
      cl::InferenceEnvironmentProperties* properties,
      Serialization* serialization) {
    if (!serialization) return absl::InvalidArgumentError("No serialization");
    // We use a fingerprint of the options to ensure compatibility.
    std::string options_fingerprint =
        delegates::StrFingerprint(options, sizeof(cl::InferenceOptions));
    auto data_key = serialization->GetEntryForKernel(
        std::string(kSerializedDataPrefix) + options_fingerprint, context,
        delegate_params);

    std::string model_data;
    auto model_data_status = data_key.GetData(context, &model_data);
    if (model_data_status == kTfLiteOk) {
      absl::Span<const uint8_t> model_span = absl::Span<const uint8_t>{
          reinterpret_cast<const uint8_t*>(model_data.data()),
          model_data.size()};
      RETURN_IF_ERROR(cl::NewInferenceEnvironment(
          *env_options, &cl_environment_, properties));
      RETURN_IF_ERROR(
          cl_environment_->NewInferenceBuilder(model_span, builder));
      TFLITE_LOG_PROD_ONCE(
          tflite::TFLITE_LOG_INFO,
          "Initialized OpenCL-based API from serialized data.");
      return absl::OkStatus();
    }

    return absl::NotFoundError("Serialization data not found");
  }

  // Returns Ok only if serialization happens successfully.
  absl::Status SaveSerializedOpenCL(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      cl::InferenceOptions* options, Serialization* serialization,
      const std::vector<uint8_t>& serialized_model) {
    if (!serialization) return absl::InvalidArgumentError("No serialization");
    // We use a fingerprint of the options to ensure compatibility.
    std::string options_fingerprint =
        delegates::StrFingerprint(options, sizeof(cl::InferenceOptions));

    // Save data.
    auto data_key = serialization->GetEntryForKernel(
        std::string(kSerializedDataPrefix) + options_fingerprint, context,
        delegate_params);
    auto save_status = data_key.SetData(
        context, reinterpret_cast<const char*>(serialized_model.data()),
        serialized_model.size());
    if (save_status != kTfLiteOk) {
      return absl::InvalidArgumentError("Failed to save serialized data");
    }
    return absl::OkStatus();
  }

  absl::Status InitializeOpenGlApi(GraphFloat32* graph,
                                   std::unique_ptr<InferenceBuilder>* builder) {
#ifndef CL_DELEGATE_NO_GL
    gl::InferenceEnvironmentOptions env_options;
    gl::InferenceEnvironmentProperties properties;
    RETURN_IF_ERROR(
        NewInferenceEnvironment(env_options, &gl_environment_, &properties));
    auto delegate_options = delegate_->options();
    gl::InferenceOptions options;
    options.usage = ToUsage(delegate_options.inference_preference);
    options.priority1 = ToPriority(delegate_options.inference_priority1);
    options.priority2 = ToPriority(delegate_options.inference_priority2);
    options.priority3 = ToPriority(delegate_options.inference_priority3);
    RETURN_IF_ERROR(gl_environment_->NewInferenceBuilder(std::move(*graph),
                                                         options, builder));
    enforce_same_thread_ = true;
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Initialized OpenGL-based API.");
    return absl::OkStatus();
#else
    return absl::UnavailableError("OpenGL-based API disabled");
#endif
  }

  // The Delegate instance that's shared across all DelegateKernel instances.
  Delegate* const delegate_;  // doesn't own the memory.
  std::unique_ptr<cl::InferenceEnvironment> cl_environment_;
#ifndef CL_DELEGATE_NO_GL
  std::unique_ptr<gl::InferenceEnvironment> gl_environment_;
#endif
  std::unique_ptr<InferenceRunner> runner_;
  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;
  // Whenever quantized inference is enabled, this maps the tensor index of each
  // originally quantized (8-bit) tensor to its float version added in
  // model_builder - and vice versa.
  absl::flat_hash_map<int, int> quant_conversion_map_;
  std::thread::id thread_id_prepare_;  // thread id used for Prapare()
  bool enforce_same_thread_ = false;   // flag to enforce same thread for Invoke
};

inline DelegateKernel* GetDelegateKernel(TfLiteNode* node) {
  return reinterpret_cast<DelegateKernel*>(node->user_data);
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
        auto gpu_delegate_kernel =
            std::make_unique<DelegateKernel>(gpu_delegate);
        const auto status = gpu_delegate_kernel->Prepare(context, params);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Init: %s",
                             std::string(status.message()).c_str());
          return nullptr;
        }
        return gpu_delegate_kernel.release();
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {
        delete reinterpret_cast<DelegateKernel*>(buffer);
      },
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (!node->user_data) {
          TF_LITE_KERNEL_LOG(
              context,
              "TfLiteGpuDelegate Prepare: delegate is not initialized");
          return kTfLiteError;
        }
        auto* gpu_delegate_kernel = GetDelegateKernel(node);
        const auto status = gpu_delegate_kernel->GetRequiredTemporaries(
            context, node, &node->temporaries);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Prepare: %s",
                             std::string(status.message()).c_str());
          return kTfLiteError;
        }
        // TODO(akulik): tflite tensors are not allocated here either. It would
        // be good to set inputs and outputs only once here instead of setting
        // them every time in .invoke.
        return kTfLiteOk;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetDelegateKernel(node)->Invoke(context);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Invoke: %s",
                             std::string(status.message()).c_str());
          return kTfLiteError;
        }
        return kTfLiteOk;
      },
      nullptr,                // .profiling_string
      0,                      // .builtin_code
      "TfLiteGpuDelegateV2",  // .custom_name
      1,                      // .version
  };

  auto* gpu_delegate = GetDelegate(delegate);
  absl::flat_hash_set<TfLiteBuiltinOperator> excluded_ops;
  if (!cl::OpenCLSupported()) {
    excluded_ops.insert(kTfLiteBuiltinSplit);
    excluded_ops.insert(kTfLiteBuiltinSplitV);
  }
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, gpu_delegate->IsQuantOpsAllowed(),
                      gpu_delegate->MaxDelegatedPartitions(), &excluded_ops);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Created %d GPU delegate kernels.",
                  gpu_delegate->num_delegate_kernels());
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace gpu
}  // namespace tflite

TfLiteDelegate* TfLiteGpuDelegateV2Create(
    const TfLiteGpuDelegateOptionsV2* options) {
  auto* gpu_delegate = new tflite::gpu::Delegate(options);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for GPU.");
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}

void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate) {
  delete tflite::gpu::GetDelegate(delegate);
}
