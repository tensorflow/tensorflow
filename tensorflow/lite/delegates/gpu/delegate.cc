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

// TODO(b/245966018): Consider refactoring this code to separate Android
// specific functionality from general functionality, and/or to separate sync
// kernel support from async kernel support, as discussed here:
// https://b/245966018#comment5.

#include "tensorflow/lite/delegates/gpu/delegate.h"

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/core/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/core/async/interop/c/constants.h"
#endif

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/quantization_util.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/delegates/serialization.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"
#include "tensorflow/lite/delegates/utils/ret_macros.h"
#include "tensorflow/lite/delegates/utils/sync_fence.h"
#include "tensorflow/lite/delegates/utils/utils.h"
#endif

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/profiling/telemetry/c/telemetry_setting_internal.h"
#include "tensorflow/lite/profiling/telemetry/telemetry.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_status.h"

#ifndef CL_DELEGATE_NO_GL
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#endif

#if defined(__ANDROID__)
using tflite::delegates::utils::BufferAttributes;
using tflite::delegates::utils::BufferType;
using tflite::delegates::utils::ConvertToTfLiteStatus;
using tflite::delegates::utils::IsPowerOfTwo;
using tflite::delegates::utils::ReadBufferAttrs;
using tflite::delegates::utils::ReadSyncAttrs;
using tflite::delegates::utils::SyncAttributes;
using tflite::delegates::utils::SyncType;
using tflite::delegates::utils::WaitForAllFds;
using tflite::delegates::utils::WriteBufferAttrs;
using tflite::delegates::utils::WriteSyncAttrs;
#endif

#define TFLITE_RETURN_IF_ABSL_ERROR(expr)             \
  do {                                                \
    if (const absl::Status val = (expr); !val.ok()) { \
      return ConvertToTfLiteStatus(val);              \
    }                                                 \
  } while (false)

#define TFLITE_RETURN_IF_ERROR(expr)                         \
  do {                                                       \
    if (const TfLiteStatus val = (expr); val != kTfLiteOk) { \
      return val;                                            \
    }                                                        \
  } while (false)

// This idiom allows selecting alternate code paths depending on whether or not
// AHWB is available.  However, it's still necessary to directly guard calls to
// AHardwareBuffer_* functions with "if (__builtin_available(android 26, *))" to
// avoid compiler errors.
#define TFLITE_AHWB_AVAILABLE()               \
  [] {                                        \
    if (__builtin_available(android 26, *)) { \
      return true;                            \
    } else {                                  \
      return false;                           \
    }                                         \
  }()

namespace tflite {
namespace gpu {
namespace {

using delegates::Serialization;
using delegates::SerializationParams;

constexpr char kSerializedDataPrefix[] = "gpuv2_data_";

#if defined(__ANDROID__)
// Xeno API does not impose alignment or padding requirements.
constexpr size_t kRequiredByteAlignment = 1;
constexpr size_t kRequiredBytePadding = 1;
#endif

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

#if defined(__ANDROID__)
class DelegateAsyncKernel;
#endif

class Delegate {
 public:
  explicit Delegate(const TfLiteGpuDelegateOptionsV2* options, bool async)
      : async_(async) {
    telemetry_settings_ =
        std::make_unique<TfLiteTelemetryGpuDelegateSettings>();
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
      telemetry_settings_ =
          std::make_unique<TfLiteTelemetryGpuDelegateSettings>();
    }
  }

  TfLiteDelegate* tflite_delegate() { return &delegate_; }
  Serialization* serialization() { return serialization_.get(); }
  const TfLiteGpuDelegateOptionsV2& options() const { return options_; }
  bool async() const { return async_; }

  bool IsQuantOpsAllowed() const {
    return options_.experimental_flags &
           TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  }
  int MaxDelegatedPartitions() const {
    return options_.max_delegated_partitions;
  }
  int num_delegate_kernels() const { return num_delegate_kernels_; }
  TfLiteTelemetryGpuDelegateSettings* telemetry_settings() {
    return telemetry_settings_.get();
  }

 private:
  TfLiteDelegate delegate_;
  TfLiteGpuDelegateOptionsV2 options_;
  std::atomic<int> num_delegate_kernels_ = 0;

  std::unique_ptr<Serialization> serialization_;

  std::unique_ptr<TfLiteTelemetryGpuDelegateSettings> telemetry_settings_;

  bool async_;

#if defined(__ANDROID__)
  // TODO(b/245966018): While tflite async is in experimental stage, there's no
  // clean way to deallocate a kernel via TfLiteRegistration::free. For now, we
  // allow kernels to live until the Delegate is destroyed.
  mutable absl::Mutex async_kernels_mutex_;
  std::vector<std::unique_ptr<DelegateAsyncKernel>> async_kernels_
      ABSL_GUARDED_BY(async_kernels_mutex_);
#endif

  friend class DelegateKernelCore;
#if defined(__ANDROID__)
  friend TfLiteRegistration CreateAsyncRegistration();
#endif
};

// Utility class to assist DelegateKernel and DelegateKernelAsync.
//
// A single DelegateKernelCore cannot be used for multiple concurrent
// executions, because it owns an InferenceRunner, which cannot be used for
// multiple concurrent executions.
//
// TODO(b/245966018): Consider factoring out DelegateKernelCore and adding unit
// tests for it, as discussed here: http://b/245966018#comment4.
class DelegateKernelCore {
 public:
  explicit DelegateKernelCore(Delegate* delegate) : delegate_(delegate) {
    ++delegate_->num_delegate_kernels_;
    telemetry_settings_ =
        std::make_unique<TfLiteTelemetryGpuDelegateSettings>();
  }
  ~DelegateKernelCore() { --delegate_->num_delegate_kernels_; }

  bool enforce_same_thread() const { return enforce_same_thread_; }
  const std::vector<int64_t>& input_indices() const { return input_indices_; }
  const std::vector<int64_t>& output_indices() const { return output_indices_; }
  const absl::flat_hash_map<int, int>& quant_conversion_map() const {
    return quant_conversion_map_;
  }
  const std::unique_ptr<InferenceRunner>& runner() const { return runner_; }

  absl::Status Setup(TfLiteContext* context,
                     const TfLiteDelegateParams* delegate_params);

 private:
  ObjectDef GetObjectDef(int index,
                         DataType data_type = DataType::FLOAT32) const;

  absl::Status InitializeGraph(TfLiteContext* context,
                               const TfLiteDelegateParams* delegate_params,
                               GraphFloat32* graph,
                               std::vector<uint32_t>* input_refs,
                               std::vector<uint32_t>* output_refs);

  absl::Status InitializeOpenClApi(GraphFloat32* graph,
                                   std::unique_ptr<InferenceBuilder>* builder,
                                   bool* graph_is_destroyed,
                                   TfLiteContext* context,
                                   const TfLiteDelegateParams* delegate_params,
                                   Serialization* serialization);

  absl::Status InitializeOpenGlApi(GraphFloat32* graph,
                                   std::unique_ptr<InferenceBuilder>* builder);

  absl::Status MaybeInitializeSerializedOpenCL(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      std::unique_ptr<InferenceBuilder>* builder, cl::InferenceOptions* options,
      cl::InferenceEnvironmentOptions* env_options,
      cl::InferenceEnvironmentProperties* properties,
      Serialization* serialization);

  absl::Status SaveSerializedOpenCL(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      cl::InferenceOptions* options, Serialization* serialization,
      const std::vector<uint8_t>& serialized_model);

  // The Delegate instance that's shared across all DelegateKernel instances.
  Delegate* const delegate_;  // doesn't own the memory.

  std::unique_ptr<cl::InferenceEnvironment> cl_environment_;
#ifndef CL_DELEGATE_NO_GL
  std::unique_ptr<gl::InferenceEnvironment> gl_environment_;
#endif

  // Note that a single InferenceRunner cannot be used for multiple concurrent
  // executions.
  std::unique_ptr<InferenceRunner> runner_;

  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;

  // Whenever quantized inference is enabled, this maps the tensor index of each
  // originally quantized (8-bit) tensor to its float version added in
  // model_builder - and vice versa.
  absl::flat_hash_map<int, int> quant_conversion_map_;

  bool enforce_same_thread_ = false;  // flag to enforce same thread for Invoke

  std::unique_ptr<TfLiteTelemetryGpuDelegateSettings> telemetry_settings_;
};

ObjectDef DelegateKernelCore::GetObjectDef(int index,
                                           DataType data_type) const {
  ObjectDef default_object_def;
  default_object_def.data_type = data_type;
  default_object_def.data_layout = DataLayout::BHWC;
  default_object_def.object_type = ObjectType::CPU_MEMORY;
  default_object_def.user_provided = true;
  return default_object_def;
}

absl::Status DelegateKernelCore::InitializeGraph(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    GraphFloat32* graph, std::vector<uint32_t>* input_refs,
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

absl::Status DelegateKernelCore::Setup(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params) {
  // Extract TFLite delegate execution plan from the context and convert it
  // into GraphFloat32.
  GraphFloat32 graph;
  std::vector<uint32_t> input_refs;
  std::vector<uint32_t> output_refs;
  RETURN_IF_ERROR(InitializeGraph(context, delegate_params, &graph, &input_refs,
                                  &output_refs));

  std::unique_ptr<InferenceBuilder> builder;
  bool graph_is_destroyed;
  bool backend_opencl = false;
  const int experimental_flags = delegate_->options().experimental_flags;
  if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY) {
    RETURN_IF_ERROR(InitializeOpenClApi(&graph, &builder, &graph_is_destroyed,
                                        context, delegate_params,
                                        delegate_->serialization()));
    backend_opencl = true;
  } else if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY) {
    if (delegate_->async()) {
      // TODO(b/245966018): Add support for this. Probably straightforward, as
      // Xeno API hides most of the CL/GL distinction.
      return absl::UnimplementedError("async API does not support OpenGL");
    }
    RETURN_IF_ERROR(InitializeOpenGlApi(&graph, &builder));
  } else {
    // By default, we try CL first & fall back to GL if that fails.
    absl::Status status =
        InitializeOpenClApi(&graph, &builder, &graph_is_destroyed, context,
                            delegate_params, delegate_->serialization());
    if (!status.ok()) {
      TF_LITE_KERNEL_LOG(context, std::string(status.message()).c_str());
      TF_LITE_KERNEL_LOG(context, "Falling back to OpenGL");
      if (delegate_->async()) {
        // TODO(b/245966018): Add support for this. Probably straightforward, as
        // Xeno API hides most of the CL/GL distinction.
        return absl::UnimplementedError("async API does not support OpenGL");
      }

      // Graph needs to be re-created because it is moved above.
      GraphFloat32 graph2;
      if (graph_is_destroyed) {
        RETURN_IF_ERROR(InitializeGraph(context, delegate_params, &graph2,
                                        &input_refs, &output_refs));
      }
      RETURN_IF_ERROR(
          InitializeOpenGlApi(graph_is_destroyed ? &graph2 : &graph, &builder));
    } else {
      backend_opencl = true;
    }
  }

  telemetry_settings_->backend =
      backend_opencl ? TfLiteTelemetryGpuDelegateSettings::OPENCL
                     : TfLiteTelemetryGpuDelegateSettings::OPENGL;
  telemetry::TelemetryReportDelegateSettings(
      context, "GpuDelegateKernel::Prepare",
      telemetry::TelemetrySource::TFLITE_GPU, telemetry_settings_.get());

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

absl::Status DelegateKernelCore::InitializeOpenClApi(
    GraphFloat32* graph, std::unique_ptr<InferenceBuilder>* builder,
    bool* graph_is_destroyed, TfLiteContext* context,
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

// Returns Ok only if serialized data is successfully found.
absl::Status DelegateKernelCore::InitializeOpenGlApi(
    GraphFloat32* graph, std::unique_ptr<InferenceBuilder>* builder) {
#ifndef CL_DELEGATE_NO_GL
  if (delegate_->async()) {
    // TODO(b/245966018): Add support for this. Probably straightforward, as
    // Xeno API hides most of the CL/GL distinction.
    return absl::UnimplementedError("async API does not support OpenGL");
  }

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

// Returns Ok only if serialized data is successfully found.
absl::Status DelegateKernelCore::MaybeInitializeSerializedOpenCL(
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
        reinterpret_cast<const uint8_t*>(model_data.data()), model_data.size()};
    RETURN_IF_ERROR(cl::NewInferenceEnvironment(*env_options, &cl_environment_,
                                                properties));
    RETURN_IF_ERROR(cl_environment_->NewInferenceBuilder(model_span, builder));
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Initialized OpenCL-based API from serialized data.");
    return absl::OkStatus();
  }

  return absl::NotFoundError("Serialization data not found");
}

// Returns Ok only if serialization happens successfully.
absl::Status DelegateKernelCore::SaveSerializedOpenCL(
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

// Represent the execution of a subset of nodes on GPU.
class DelegateKernel {
 public:
  explicit DelegateKernel(Delegate* delegate) : core_(delegate) {}
  ~DelegateKernel() = default;

  absl::Status Prepare(TfLiteContext* context,
                       const TfLiteDelegateParams* delegate_params) {
    thread_id_prepare_ = std::this_thread::get_id();

    return core_.Setup(context, delegate_params);
  }

  // This directs the runtime to allocate memory for input/output temporary
  // tensors that require dequantization/quantization.  This is ordinary
  // CPU memory.
  absl::Status GetRequiredTemporaries(TfLiteContext* context, TfLiteNode* node,
                                      TfLiteIntArray** temporaries_array_ptr) {
    if (core_.quant_conversion_map().empty()) return absl::OkStatus();

    std::vector<int> temporary_tensors;
    for (auto index : core_.input_indices()) {
      if (core_.quant_conversion_map().find(index) !=
          core_.quant_conversion_map().end()) {
        temporary_tensors.push_back(index);
      }
    }
    for (auto index : core_.output_indices()) {
      if (core_.quant_conversion_map().find(index) !=
          core_.quant_conversion_map().end()) {
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
      if (core_.enforce_same_thread()) {
        return absl::FailedPreconditionError(
            "GpuDelegate must run on the same thread where it was "
            "initialized.");
      }
    }

    const bool is_dequant_required = !core_.quant_conversion_map().empty();
    if (is_dequant_required) {
      RETURN_IF_ERROR(DequantizeInputs(context, core_.input_indices(),
                                       core_.quant_conversion_map()));
    }
    RETURN_IF_ERROR(SetInputsAndOutputs(context));
    RETURN_IF_ERROR(core_.runner()->Run());
    if (is_dequant_required) {
      RETURN_IF_ERROR(QuantizeOutputs(context, core_.output_indices(),
                                      core_.quant_conversion_map()));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status SetInputsAndOutputs(TfLiteContext* context) {
    for (int i = 0; i < core_.input_indices().size(); ++i) {
      RETURN_IF_ERROR(core_.runner()->SetInputObject(
          i, GetTensorObject(core_.input_indices()[i], context)));
    }
    for (int i = 0; i < core_.output_indices().size(); ++i) {
      RETURN_IF_ERROR(core_.runner()->SetOutputObject(
          i, GetTensorObject(core_.output_indices()[i], context)));
    }
    return absl::OkStatus();
  }

  TensorObject GetTensorObject(int index, TfLiteContext* context) const {
    auto& tensor = context->tensors[index];
    return MakeCpuMemory(absl::MakeSpan(tensor.data.raw, tensor.bytes));
  }

 private:
  DelegateKernelCore core_;
  std::thread::id thread_id_prepare_;  // thread id used for Prepare()
};

#if defined(__ANDROID__)
using BackendAsyncKernelInterface =
    ::tflite::delegates::BackendAsyncKernelInterface;

// Represent the execution of a subset of nodes on GPU, for use with async API.
class DelegateAsyncKernel : public BackendAsyncKernelInterface {
 public:
  explicit DelegateAsyncKernel(Delegate* delegate) : core_(delegate) {}
  ~DelegateAsyncKernel() override = default;

  absl::Status Init(TfLiteContext* context, const TfLiteDelegateParams* params);

  // Buffer operations
  TfLiteStatus RegisterBuffer(TfLiteOpaqueContext* opaque_context,
                              TfLiteIoType io_type,
                              const TfLiteBackendBuffer* buffer,
                              const TfLiteAttributeMap* attrs,
                              TfLiteBufferHandle handle) override;
  TfLiteStatus RegisterBufferSlice(TfLiteOpaqueContext* context,
                                   TfLiteBufferHandle buffer_pool,
                                   const TfLiteAttributeMap* attrs,
                                   TfLiteBufferHandle handle) override {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                    "DelegateAsyncKernel::RegisterBufferSlice unimplemented");
    return kTfLiteError;
  }
  TfLiteStatus UnregisterBuffer(TfLiteOpaqueContext* opaque_context,
                                TfLiteBufferHandle handle) override;

  // Reconciliations
  const std::vector<const char*>& SupportedBufferTypes(
      TfLiteIoType io_type) const override {
    return supported_buffer_types_;
  }
  const std::vector<const char*>& SupportedSynchronizations(
      TfLiteIoType io_type) const override {
    return supported_synchronizations_;
  }
  bool ReconcileRestrictions(TfLiteOpaqueContext* opaque_context,
                             TfLiteOpaqueNode* opaque_node, int tensor_index,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const override;
  TfLiteStatus SetAttributes(TfLiteOpaqueContext* context,
                             TfLiteOpaqueNode* node, int tensor_index,
                             const TfLiteAttributeMap* attrs) override;
  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  // Execution methods
  TfLiteStatus Eval(TfLiteOpaqueContext* opaque_context,
                    TfLiteOpaqueNode* opaque_node,
                    TfLiteExecutionTask* task) override;
  TfLiteStatus Wait(TfLiteOpaqueContext* opaque_context,
                    TfLiteExecutionTask* task) override {
    // Implementation is synchronous, so Wait is a no-op.
    return kTfLiteOk;
  }
  TfLiteStatus Finish(TfLiteOpaqueContext* opaque_context,
                      TfLiteExecutionTask* task) override {
    // Implementation is synchronous, so Finish is a no-op.
    return kTfLiteOk;
  }

 private:
  DelegateKernelCore core_;

  // Similar to the corresponding BackendAsyncKernelInterface method,
  // but accepts non-opaque input types.
  // TODO(b/241109768): Instead of reinterpret_cast, switch to use the stable
  // APIs with opaque types when those are ready.
  TfLiteStatus RegisterBufferImpl(TfLiteContext* context, TfLiteIoType io_type,
                                  const TfLiteBackendBuffer* buffer,
                                  const TfLiteAttributeMap* attrs,
                                  TfLiteBufferHandle handle);
  TfLiteStatus UnregisterBufferImpl(TfLiteContext* context,
                                    TfLiteBufferHandle handle);
  TfLiteStatus SetAttributesImpl(TfLiteContext* context, TfLiteNode* node,
                                 int tensor_index,
                                 const TfLiteAttributeMap* attrs);
  TfLiteStatus PrepareImpl(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                        TfLiteExecutionTask* task);

  using UniquePtrAHardwareBuffer =
      std::unique_ptr<AHardwareBuffer, void (*)(AHardwareBuffer*)>;
  static UniquePtrAHardwareBuffer Acquire(AHardwareBuffer* ahwb) {
    if (__builtin_available(android 26, *)) {
      AHardwareBuffer_acquire(ahwb);
    } else {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "attempting AHardwareBuffer_acquire on a device without "
                      "AHardwareBuffer support");
    }
    return UniquePtrAHardwareBuffer(ahwb, [](AHardwareBuffer* b) {
      if (__builtin_available(android 26, *)) {
        AHardwareBuffer_release(b);
      }
    });
  }
  static AHardwareBuffer_Desc Describe(
      const UniquePtrAHardwareBuffer& uptr_ahwb) {
    AHardwareBuffer_Desc desc_ahwb = {};
    if (__builtin_available(android 26, *)) {
      AHardwareBuffer_describe(uptr_ahwb.get(), &desc_ahwb);
    } else {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "attempting AHardwareBuffer_describe on a device without "
                      "AHardwareBuffer support");
    }
    return desc_ahwb;
  }

  // For SupportedBufferTypes and SupportedSynchronizations
  const std::vector<const char*> supported_buffer_types_ = {
      ::tflite::delegates::utils::kBufferTypeAHardwareBufferBlob};
  const std::vector<const char*> supported_synchronizations_ = {
      kTfLiteSyncTypeNoSyncObj,
      ::tflite::delegates::utils::kSyncTypeSyncFenceFd};

  mutable absl::Mutex mutex_;

  absl::flat_hash_map<TfLiteBufferHandle, UniquePtrAHardwareBuffer>
      buffer_by_handle_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<int, SyncType> sync_type_by_tensor_index_
      ABSL_GUARDED_BY(mutex_);
  std::vector<SyncType> input_sync_types_ ABSL_GUARDED_BY(mutex_);
  std::vector<SyncType> output_sync_types_ ABSL_GUARDED_BY(mutex_);

  // Whether 'Prepare' is called or not.
  bool prepared_ ABSL_GUARDED_BY(mutex_) = false;
};

absl::Status DelegateAsyncKernel::Init(TfLiteContext* context,
                                       const TfLiteDelegateParams* params) {
  return core_.Setup(context, params);
}

namespace {

bool ReconcileBufferRestrictions(TfLiteContext* context, int tensor_index,
                                 const BufferAttributes& user,
                                 BufferAttributes& merged,
                                 BufferAttributes& conflict) {
  auto buffer_type =
      user.buffer_type.value_or(BufferType::kAHardwareBufferBlob);
  if (buffer_type != BufferType::kAHardwareBufferBlob) {
    conflict.buffer_type = BufferType::kAHardwareBufferBlob;
    return false;
  }
  merged.buffer_type = buffer_type;

  if (user.alignment.has_value()) {
    size_t alignment = user.alignment.value();
    if (IsPowerOfTwo(alignment)) {
      merged.alignment = std::max(alignment, kRequiredByteAlignment);
    } else {
      conflict.alignment = kRequiredByteAlignment;
      return false;
    }
  }

  size_t merged_padding_value = kRequiredBytePadding;
  if (user.padding.has_value()) {
    size_t padding = user.padding.value();
    if (IsPowerOfTwo(padding)) {
      merged.padding = merged_padding_value =
          std::max(padding, kRequiredBytePadding);
    } else {
      conflict.padding = kRequiredBytePadding;
      return false;
    }
  }

  size_t required_size = tflite::delegates::utils::RoundUp(
      context->tensors[tensor_index].bytes, merged_padding_value);
  merged.size = std::max(user.size.value_or(0), required_size);
  return true;
}

bool ReconcileSyncRestrictions(TfLiteContext* context, int tensor_index,
                               const SyncAttributes& user,
                               SyncAttributes& merged,
                               SyncAttributes& conflict) {
  auto sync_type = user.sync_type.value_or(SyncType::kNoSyncObj);
  if (sync_type == SyncType::kUnknown) {
    conflict.sync_type = SyncType::kNoSyncObj;
    return false;
  }
  merged.sync_type = sync_type;
  return true;
}

}  // namespace

bool DelegateAsyncKernel::ReconcileRestrictions(
    TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* opaque_node,
    int tensor_index, const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  TFLITE_ABORT_CHECK(opaque_context != nullptr, "");            // Crash OK
  TFLITE_ABORT_CHECK(user_provided_attributes != nullptr, "");  // Crash OK
  TFLITE_ABORT_CHECK(merged != nullptr, "");                    // Crash OK
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  if (TfLiteAttributeMapIsBufferAttributeMap(user_provided_attributes)) {
    if (!TfLiteAttributeMapIsBufferAttributeMap(merged)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "'merged' have a different attribute map type than "
                      "'user_provided_attributes'");
      return false;
    }
    if (conflict != nullptr &&
        !TfLiteAttributeMapIsBufferAttributeMap(conflict)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "'conflict' have a different attribute map type than "
                      "'user_provided_attributes'");
      return false;
    }
    BufferAttributes merged_attrs{};
    BufferAttributes conflict_attrs{};
    bool ok = ReconcileBufferRestrictions(
        context, tensor_index, ReadBufferAttrs(user_provided_attributes),
        merged_attrs, conflict_attrs);
    WriteBufferAttrs(merged_attrs, merged);
    if (conflict != nullptr) {
      WriteBufferAttrs(conflict_attrs, conflict);
    }
    return ok;
  }
  if (TfLiteAttributeMapIsSyncAttributeMap(user_provided_attributes)) {
    if (!TfLiteAttributeMapIsSyncAttributeMap(merged)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "'merged' have a different attribute map type than "
                      "'user_provided_attributes'");
      return false;
    }
    if (conflict != nullptr &&
        !TfLiteAttributeMapIsSyncAttributeMap(conflict)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "'conflict' have a different attribute map type than "
                      "'user_provided_attributes'");
      return false;
    }
    SyncAttributes merged_attrs{};
    SyncAttributes conflict_attrs{};
    bool ok = ReconcileSyncRestrictions(context, tensor_index,
                                        ReadSyncAttrs(user_provided_attributes),
                                        merged_attrs, conflict_attrs);
    WriteSyncAttrs(merged_attrs, merged);
    if (conflict != nullptr) {
      WriteSyncAttrs(conflict_attrs, conflict);
    }
    return ok;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "unknown type of user_provided_attributes");
  return false;
}

TfLiteStatus DelegateAsyncKernel::SetAttributes(
    TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* opaque_node,
    int tensor_index, const TfLiteAttributeMap* attrs) {
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  auto* node = reinterpret_cast<TfLiteNode*>(opaque_node);
  return SetAttributesImpl(context, node, tensor_index, attrs);
}

TfLiteStatus DelegateAsyncKernel::SetAttributesImpl(
    TfLiteContext* context, TfLiteNode* node, int tensor_index,
    const TfLiteAttributeMap* attrs) {
  // Currently we are only able to handle sync attributes.
  TFLITE_RET_CHECK_STATUS(
      TfLiteAttributeMapIsSyncAttributeMap(attrs),
      "calling SetAttributes with an invalid attribute map type");

  // Validate sync attributes.
  auto sync_attrs = ReadSyncAttrs(attrs);
  TFLITE_RET_CHECK_STATUS(
      sync_attrs.sync_type.has_value(),
      "calling SetAttributes with sync object type name unspecified");
  TFLITE_RET_CHECK_STATUS(
      sync_attrs.sync_type.value() != SyncType::kUnknown,
      "calling SetAttributes with unknown sync object type name");

  // Record the attributes.
  absl::MutexLock lock(&mutex_);
  TFLITE_RET_CHECK_STATUS(!prepared_,
                          "SetAttributes must be called before Prepare");
  sync_type_by_tensor_index_[tensor_index] = sync_attrs.sync_type.value();

  return kTfLiteOk;
}

TfLiteStatus DelegateAsyncKernel::Prepare(TfLiteOpaqueContext* opaque_context,
                                          TfLiteOpaqueNode* opaque_node) {
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  auto* node = reinterpret_cast<TfLiteNode*>(opaque_node);
  return PrepareImpl(context, node);
}

TfLiteStatus DelegateAsyncKernel::PrepareImpl(TfLiteContext* context,
                                              TfLiteNode* node) {
  absl::MutexLock lock(&mutex_);
  TFLITE_RET_CHECK_STATUS(!prepared_, "Prepare must be called at most once");

  input_sync_types_.resize(node->inputs->size, SyncType::kNoSyncObj);
  output_sync_types_.resize(node->outputs->size, SyncType::kNoSyncObj);
  for (size_t i = 0; i < node->inputs->size; ++i) {
    auto it = sync_type_by_tensor_index_.find(node->inputs->data[i]);
    if (it != sync_type_by_tensor_index_.end()) {
      TFLITE_ABORT_CHECK(it->second != SyncType::kUnknown, "");  // Crash OK
      input_sync_types_[i] = it->second;
    }
  }
  for (size_t i = 0; i < node->outputs->size; ++i) {
    auto it = sync_type_by_tensor_index_.find(node->outputs->data[i]);
    if (it != sync_type_by_tensor_index_.end()) {
      TFLITE_ABORT_CHECK(it->second != SyncType::kUnknown, "");  // Crash OK
      output_sync_types_[i] = it->second;
    }
  }

  prepared_ = true;
  return kTfLiteOk;
}

TfLiteStatus DelegateAsyncKernel::RegisterBuffer(
    TfLiteOpaqueContext* opaque_context, TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle handle) {
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  return RegisterBufferImpl(context, io_type, buffer, attrs, handle);
}

TfLiteStatus DelegateAsyncKernel::RegisterBufferImpl(
    TfLiteContext* context, TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle handle) {
  TFLITE_ABORT_CHECK(buffer != nullptr, "");                  // Crash OK
  TFLITE_ABORT_CHECK(attrs != nullptr, "");                   // Crash OK
  TFLITE_ABORT_CHECK(handle != kTfLiteNullBufferHandle, "");  // Crash OK

  // Validate buffer attributes.
  TFLITE_RET_CHECK_STATUS(
      TfLiteAttributeMapIsBufferAttributeMap(attrs),
      "calling RegisterBuffer with invalid attribute map type");
  auto buffer_attrs = ReadBufferAttrs(attrs);
  TFLITE_RET_CHECK_STATUS(
      buffer_attrs.buffer_type.has_value(),
      "calling RegisterBuffer with buffer resource type name unspecified");
  TFLITE_RET_CHECK_STATUS(
      buffer_attrs.buffer_type.value() != BufferType::kUnknown,
      "calling RegisterBuffer with unknown buffer resource type");
  size_t alignment = buffer_attrs.alignment.value_or(kRequiredByteAlignment);
  TFLITE_RET_CHECK_STATUS(
      alignment % kRequiredByteAlignment == 0,
      "calling RegisterBuffer with invalid buffer alignment");
  size_t padding = buffer_attrs.padding.value_or(kRequiredBytePadding);
  TFLITE_RET_CHECK_STATUS(padding % kRequiredBytePadding == 0,
                          "calling RegisterBuffer with invalid buffer padding");
  size_t offset = buffer_attrs.offset.value_or(0);
  TFLITE_RET_CHECK_STATUS(offset == 0,
                          "calling RegisterBuffer with non-zero offset");

  // We depend on the availability of AHardwareBuffer.
  TFLITE_RET_CHECK_STATUS(
      TFLITE_AHWB_AVAILABLE(),
      "calling tflite::gpu::DelegateAsyncKernel::RegisterBuffer on device "
      "without AHardwareBuffer support");

  // Retrieve and validate the buffer.
  auto* ahwb =
      reinterpret_cast<AHardwareBuffer*>(TfLiteBackendBufferGetPtr(buffer));
  TFLITE_RET_CHECK_STATUS(ahwb != nullptr,
                          "calling RegisterBuffer with nullptr buffer");
  UniquePtrAHardwareBuffer uptr_ahwb = Acquire(ahwb);
  const AHardwareBuffer_Desc desc_ahwb = Describe(uptr_ahwb);
  TFLITE_RET_CHECK_STATUS(
      desc_ahwb.format == AHARDWAREBUFFER_FORMAT_BLOB,
      "calling RegisterBuffer with an AHardwareBuffer of format other than "
      "BLOB is not supported");
  size_t size = buffer_attrs.size.value_or(desc_ahwb.width);
  TFLITE_RET_CHECK_STATUS(
      size <= desc_ahwb.width,
      "calling RegisterBuffer with buffer size larger than the actual "
      "AHardwareBuffer size");

  // Register the buffer.
  absl::MutexLock lock(&mutex_);
  auto [it, did_something] =
      buffer_by_handle_.try_emplace(handle, std::move(uptr_ahwb));
  TFLITE_RET_CHECK_STATUS(did_something,
                          "RegisterBuffer called with duplicate handle");
  return kTfLiteOk;
}

TfLiteStatus DelegateAsyncKernel::UnregisterBuffer(
    TfLiteOpaqueContext* opaque_context, TfLiteBufferHandle handle) {
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  return UnregisterBufferImpl(context, handle);
}

TfLiteStatus DelegateAsyncKernel::UnregisterBufferImpl(
    TfLiteContext* context, TfLiteBufferHandle handle) {
  absl::MutexLock lock(&mutex_);
  auto it = buffer_by_handle_.find(handle);
  TFLITE_RET_CHECK_STATUS(it != buffer_by_handle_.end(),
                          "UnregisterBuffer called with unknown handle");
  buffer_by_handle_.erase(it);
  return kTfLiteOk;
}

TfLiteStatus DelegateAsyncKernel::Eval(TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueNode* opaque_node,
                                       TfLiteExecutionTask* task) {
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  auto* node = reinterpret_cast<TfLiteNode*>(opaque_node);
  return EvalImpl(context, node, task);
}

TfLiteStatus DelegateAsyncKernel::EvalImpl(TfLiteContext* context,
                                           TfLiteNode* node,
                                           TfLiteExecutionTask* task) {
  // For now we implement synchronous (rather than asynchronous) inference,
  // taking the following approach:
  // - explicitly wait on sync objects prior to commencing inference
  // - on input and output, use AHWB functions to access memory as CPU memory,
  // to use with Xeno
  // - explicitly signal sync objects after completing inference

  // We depend on the availability of AHardwareBuffer.
  TFLITE_RET_CHECK_STATUS(TFLITE_AHWB_AVAILABLE(),
                          "calling tflite::gpu::DelegateAsyncKernel::Eval on "
                          "device without AHardwareBuffer support");

  // Utility class to assist in locking and unlocking buffers.  Allows us to
  // automatically unlock buffers when we leave the scope of the class instance
  // (e.g., early exit from EvalImpl()).
  class LockedAHWBs {
   public:
    enum class Usage { READ, WRITE };

    explicit LockedAHWBs(Usage usage)
        : usage_(usage == Usage::READ
                     ? AHARDWAREBUFFER_USAGE_CPU_READ_RARELY
                     : AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY) {}
    ~LockedAHWBs() { unlock(); }

    // See AHardwareBuffer_lock() but also tracks the lock so that we can unlock
    // later. If we successfully lock a buffer, and then try again without
    // unlocking, we'll return the same outVirtualAddress and not try to re-lock
    // the buffer.
    int lock(AHardwareBuffer* _Nonnull buffer,
             void* _Nullable* _Nonnull outVirtualAddress) {
      if (auto it = buffers_to_addresses_.find(buffer);
          it != buffers_to_addresses_.end()) {
        *outVirtualAddress = it->second;
        return 0;
      }

      int lock_error = [buffer, outVirtualAddress, this] {
        if (__builtin_available(android 26, *)) {
          return AHardwareBuffer_lock(buffer, this->usage_, -1 /* fence */,
                                      nullptr /* rect */, outVirtualAddress);
        } else {
          return EINVAL;
        }
      }();
      if (!lock_error) {
        auto [it, inserted] =
            buffers_to_addresses_.insert({buffer, *outVirtualAddress});
        if (inserted == false) {
          TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "LockedAHWBs::lock inconsistency");
          return -EINVAL;
        }
      }
      return lock_error;
    }

    // Performs AHardwareBuffer_unlock() on all buffers we have locked.
    // Attempts to unlock all buffers, even if unlock of one or more bufers
    // fails. Returns the error from the first failure. After this call, we are
    // no longer tracking any buffers for unlocking.
    int unlock() {
      int fn_error = 0;

      for (auto [buffer, address] : buffers_to_addresses_) {
        if (int unlock_error = [buffer = buffer] {
              if (__builtin_available(android 26, *)) {
                return AHardwareBuffer_unlock(buffer, nullptr /* fence */);
              } else {
                return EINVAL;
              }
            }()) {
          if (!fn_error) {
            fn_error = unlock_error;
          }
        }
      }

      buffers_to_addresses_.clear();
      if (fn_error) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "LockedAHWBs::unlock error %d",
                        fn_error);
      }
      return fn_error;
    }

   private:
    const int
        usage_;  // Marked const because this is frozen at construction time and
                 // because we want the class instance to be non copyable.
    absl::flat_hash_map<AHardwareBuffer*, void*> buffers_to_addresses_;
  };

  absl::MutexLock lock(&mutex_);
  TFLITE_RET_CHECK_STATUS(prepared_, "Eval must be called after Prepare");

  // TODO(b/245966018): Need to call GetRequiredTemporaries? Today the data
  // lives on TfLiteNode -- is that what we want?
  TFLITE_RET_CHECK_STATUS(core_.quant_conversion_map().empty(),
                          "Quantized kernels are not supported");

  // The code below needs to do work corresponding to DelegateKernel::Invoke:

  // Wait on input sync objects.
  //   Get unique input sync fences.
  absl::flat_hash_set<int> unique_input_sync_fds_set;
  for (size_t i = 0; i < node->inputs->size; ++i) {
    if (input_sync_types_[i] == SyncType::kNoSyncObj) continue;
    TfLiteSynchronization* sync =
        TfLiteExecutionTaskGetSyncByIndex(task, node->inputs->data[i]);
    if (sync == nullptr) continue;
    void* sync_obj = TfLiteSynchronizationGetPtr(sync);
    if (sync_obj == nullptr) continue;
    int sync_fence_fd = *(reinterpret_cast<int*>(sync_obj));
    if (sync_fence_fd == -1) continue;
    unique_input_sync_fds_set.insert(sync_fence_fd);
  }
  //  Wait for all input sync fences to be signalled.
  std::vector<int> unique_input_sync_fds_vec;
  unique_input_sync_fds_vec.reserve(unique_input_sync_fds_set.size());
  for (auto fd : unique_input_sync_fds_set) {
    unique_input_sync_fds_vec.push_back(fd);
  }
  const auto waitfor = WaitForAllFds(unique_input_sync_fds_vec);
  TFLITE_RET_CHECK_STATUS(waitfor.has_value(), "wait for input fds");

  // Iterate over inputs or outputs, and for each one do AHardwareBuffer_lock()
  // and InferenceRunner::Set*Object(). Compare to
  // DelegateKernel::SetInputsAndOutputs().
  auto SetInputsOrOutputs =
      [this, task](LockedAHWBs* locks,
                   const std::vector<int64_t>& in_or_out_indices,
                   absl::Status (InferenceRunner::*set_object)(
                       int index, TensorObject object))
          ABSL_SHARED_LOCKS_REQUIRED(mutex_) -> TfLiteStatus {
    for (int i = 0; i < in_or_out_indices.size(); ++i) {
      int context_tensor_index = in_or_out_indices[i];
      TfLiteBufferHandle handle =
          TfLiteExecutionTaskGetBufferByIndex(task, context_tensor_index);
      TFLITE_RET_CHECK_STATUS(handle >= 0, "bad handle");
      const UniquePtrAHardwareBuffer& uptr_ahwb = buffer_by_handle_.at(handle);
      void* virtualAddress = nullptr;
      TFLITE_RET_CHECK_STATUS(
          0 == locks->lock(uptr_ahwb.get(), &virtualAddress), "lock() failed");
      TFLITE_RETURN_IF_ABSL_ERROR((core_.runner().get()->*set_object)(
          i,
          MakeCpuMemory(absl::MakeSpan(reinterpret_cast<char*>(virtualAddress),
                                       Describe(uptr_ahwb).width))));
    }
    return kTfLiteOk;
  };

  LockedAHWBs read_locks(LockedAHWBs::Usage::READ);
  TFLITE_RETURN_IF_ERROR(SetInputsOrOutputs(&read_locks, core_.input_indices(),
                                            &InferenceRunner::SetInputObject));

  // TODO(b/245966018): Dequantize inputs, if needed (see
  // core_.quant_conversion_map())

  LockedAHWBs write_locks(LockedAHWBs::Usage::WRITE);
  TFLITE_RETURN_IF_ERROR(SetInputsOrOutputs(
      &write_locks, core_.output_indices(), &InferenceRunner::SetOutputObject));

  TFLITE_RETURN_IF_ABSL_ERROR(core_.runner()->Run());

  int unlock_status = read_locks.unlock();

  // TODO(b/245966018): Quantize outputs, if needed (see
  // core_.quant_conversion_map())

  if (int write_unlock_status = write_locks.unlock()) {
    if (!unlock_status) {
      unlock_status = write_unlock_status;
    }
  }

  // Signal sync objects by indicating that inference has completed and there
  // are no output sync fences.
  for (size_t i = 0; i < node->outputs->size; ++i) {
    if (output_sync_types_[i] == SyncType::kNoSyncObj) continue;
    TfLiteSynchronization* sync =
        TfLiteExecutionTaskGetSyncByIndex(task, node->outputs->data[i]);
    if (sync == nullptr) continue;
    TfLiteSynchronizationSetPtr(sync, new int{-1});
  }

  TFLITE_RET_CHECK_STATUS(unlock_status == 0, "unlock() failed");
  return kTfLiteOk;
}

#endif  // defined(__ANDROID__)

inline DelegateKernel* GetDelegateKernel(TfLiteNode* node) {
  return reinterpret_cast<DelegateKernel*>(node->user_data);
}

inline Delegate* GetDelegate(TfLiteDelegate* delegate) {
  return reinterpret_cast<Delegate*>(delegate->data_);
}

const char kRegistrationCustomName[] = "TfLiteGpuDelegateV2";

TfLiteRegistration CreateRegistration() {
  return TfLiteRegistration{
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
      nullptr,                  // .profiling_string
      0,                        // .builtin_code
      kRegistrationCustomName,  // .custom_name
      1,                        // .version
  };
}

#if defined(__ANDROID__)
TfLiteRegistration CreateAsyncRegistration() {
  return TfLiteRegistration{
      // .init
      [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* gpu_delegate = GetDelegate(params->delegate);
        // Everything below should happen in prepare function call, but TFLite
        // for whatever reason forbids that.
        auto gpu_delegate_kernel =
            std::make_unique<DelegateAsyncKernel>(gpu_delegate);
        const auto status = gpu_delegate_kernel->Init(context, params);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteGpuDelegate Init (async): %s",
                             std::string(status.message()).c_str());
          return nullptr;
        }
        TfLiteAsyncKernel* tflite_async_kernel =
            gpu_delegate_kernel.get()->kernel();
        absl::MutexLock lock(&gpu_delegate->async_kernels_mutex_);
        gpu_delegate->async_kernels_.emplace_back(
            std::move(gpu_delegate_kernel));
        return tflite_async_kernel;
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {
        // Do nothing: See Delegate::async_kernels_.
      },
      // ,prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (!node->user_data) {
          TF_LITE_KERNEL_LOG(
              context,
              "TfLiteGpuDelegate Prepare (async): delegate is not initialized");
          return kTfLiteError;
        }
        return kTfLiteOk;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "Should not call TfLiteRegistration::invoke when using "
                        "async API");
        return kTfLiteError;
      },
      nullptr,                  // .profiling_string
      0,                        // .builtin_code
      kRegistrationCustomName,  // .custom_name
      1,                        // .version
  };
}
#endif  // defined(__ANDROID__)

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  auto* gpu_delegate = GetDelegate(delegate);

  const TfLiteRegistration kRegistration =
#if defined(__ANDROID__)
      gpu_delegate->async() ? CreateAsyncRegistration() : CreateRegistration();
#else
      CreateRegistration();
#endif

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
  auto* delegate_setting = gpu_delegate->telemetry_settings();
  delegate_setting->num_nodes_delegated = ops_to_replace->size;
  TfLiteIntArrayFree(ops_to_replace);
  telemetry::TelemetryReportDelegateSettings(
      context, "GpuDelegate::DelegatePrepare",
      telemetry::TelemetrySource::TFLITE_GPU, delegate_setting);
  return status;
}

}  // namespace
}  // namespace gpu
}  // namespace tflite

TfLiteDelegate* TfLiteGpuDelegateV2Create(
    const TfLiteGpuDelegateOptionsV2* options) {
  auto* gpu_delegate = new tflite::gpu::Delegate(options, /*async=*/false);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for GPU.");
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}

#if defined(__ANDROID__)
TfLiteDelegate* TfLiteGpuDelegateV2CreateAsync(
    const TfLiteGpuDelegateOptionsV2* options) {
  // We depend on the availability of AHardwareBuffer.
  if (!TFLITE_AHWB_AVAILABLE()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "calling TfLiteGpuDelegateV2CreateAsync on device without "
                    "AHardwareBuffer support");
    return nullptr;
  }

  auto* gpu_delegate = new tflite::gpu::Delegate(options, /*async=*/true);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for GPU (async).");
  return gpu_delegate ? gpu_delegate->tflite_delegate() : nullptr;
}
#endif  // defined(__ANDROID__)

void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate) {
  delete tflite::gpu::GetDelegate(delegate);
}
