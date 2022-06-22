/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/interpreter.h"

#include <stddef.h>
#include <stdlib.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ruy/denormal.h"  // from @ruy
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/util.h"

// TODO(b/139446230): Move to portable platform header.
#if defined(__ANDROID__)
#define TFLITE_IS_MOBILE_PLATFORM
#endif  // defined(__ANDROID__)

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define TFLITE_IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define TFLITE_IS_MOBILE_PLATFORM
#endif
#endif  // defined(__APPLE__)

// TODO(b/132087118): move static_assert to c_api_internal when compiled with
// C++.
static_assert(sizeof(TfLiteFloat16) == sizeof(uint16_t),
              "Float 16 type must be 16 bits.");

namespace tflite {

namespace {

// Gets the current TfLiteQuantization from the legacy TfLiteQuantizationParams.
TfLiteQuantization GetQuantizationFromLegacy(
    const TfLiteQuantizationParams& legacy_quantization) {
  TfLiteQuantization quantization;
  quantization.type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(1);
  affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  affine_quantization->scale->data[0] = legacy_quantization.scale;
  affine_quantization->zero_point->data[0] = legacy_quantization.zero_point;
  quantization.params = affine_quantization;

  return quantization;
}

// TODO(b/153131797): We have put 'delegate_status' to 0 in the following macro
// temporarily because delegate-specific error codes are either not retrievable
// at the moment, which we will add later.
#define TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(runtime_event, a) \
  do {                                                                      \
    TfLiteStatus status = (a);                                              \
    runtime_event.set_runtime_status(/*delegate_status=*/0,                 \
                                     static_cast<int64_t>(status));         \
    TF_LITE_ENSURE_STATUS(status);                                          \
  } while (0)

}  // namespace

Interpreter::Interpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  // TODO(b/128420794): Include the TFLite runtime version in the log.
  // Prod logging is useful for mobile platforms where scraping console logs is
  // critical for debugging.
#if defined(TFLITE_IS_MOBILE_PLATFORM)
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#else
  TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#endif

  // There's always at least 1 subgraph which is the primary subgraph.
  AddSubgraphs(1);
  context_ = primary_subgraph().context();

  // Reserve some space for the tensors to avoid excessive resizing.
  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  // This operation is cheap because we allocate the CPU context resources (i.e.
  // threads) lazily.
  own_external_cpu_backend_context_.reset(new ExternalCpuBackendContext());
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();
}

Interpreter::~Interpreter() {
  // The owned external Cpu Backend Context will go out of scope with this
  // interpreter. If we have an external backend context that is not
  // owned, we need to clear the cache for other interpreters that may
  // use the context.
  if (external_contexts_[kTfLiteCpuBackendContext] &&
      (external_contexts_[kTfLiteCpuBackendContext] !=
       own_external_cpu_backend_context_.get())) {
    ExternalCpuBackendContext* external_context =
        static_cast<ExternalCpuBackendContext*>(
            external_contexts_[kTfLiteCpuBackendContext]);
    TfLiteInternalBackendContext* internal_context =
        external_context->internal_backend_context();
    if (internal_context) {
      // This call may have negative performance impacts on the next inference
      // for any interpreter using this context. The cache will be refreshed
      // by the next inference.
      internal_context->ClearCaches();
    }
  }
}

void Interpreter::SetExternalContext(TfLiteExternalContextType type,
                                     TfLiteExternalContext* ctx) {
  if (ctx == own_external_cpu_backend_context_.get()) {
    error_reporter_->Report(
        "WARNING: The passed external context is identical to the internally "
        "owned one.");
    return;
  }

  // We have an internally owned external context of kTfLiteCpuBackendContext.
  // If it's overwritten here, we will release the resource of the internally
  // owned external context.
  // Note: the 'max thread count' info associated with the overwritten context
  // will be lost here, and such info is now determined by the new context, thus
  // affecting how much parallelism a TFLite op would have.
  if (kTfLiteCpuBackendContext == type &&
      external_contexts_[kTfLiteCpuBackendContext] ==
          own_external_cpu_backend_context_.get()) {
    own_external_cpu_backend_context_.reset();
  }

  // This essentially changes the "external_contexts_[type]".
  primary_subgraph().SetExternalContext(type, ctx);
}

TfLiteStatus Interpreter::SetInputs(std::vector<int> inputs) {
  return primary_subgraph().SetInputs(std::move(inputs));
}

TfLiteStatus Interpreter::SetOutputs(std::vector<int> outputs) {
  return primary_subgraph().SetOutputs(std::move(outputs));
}

TfLiteStatus Interpreter::SetVariables(std::vector<int> variables) {
  return primary_subgraph().SetVariables(std::move(variables));
}

TfLiteStatus Interpreter::AllocateTensors() {
  // Apply the default delegate that TFLite will enable at this point to allow
  // other user-level delegates to be applied first. Only returns error when
  // the status is kTfLiteError. For other statuses, it will fall back to the
  // default implementation.
  if (ApplyLazyDelegateProviders() == kTfLiteError) return kTfLiteError;

  return primary_subgraph().AllocateTensors();
}

void Interpreter::AddSubgraphs(int subgraphs_to_add,
                               int* first_new_subgraph_index) {
  const size_t base_index = subgraphs_.size();
  if (first_new_subgraph_index) *first_new_subgraph_index = base_index;

  subgraphs_.reserve(base_index + subgraphs_to_add);
  for (int i = 0; i < subgraphs_to_add; ++i) {
    Subgraph* subgraph =
        new Subgraph(error_reporter_, external_contexts_, &subgraphs_,
                     &resources_, &resource_ids_, &initialization_status_map_);
    subgraphs_.emplace_back(subgraph);
  }
}

TfLiteStatus Interpreter::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  return primary_subgraph().AddNodeWithParameters(
      inputs, outputs, {}, init_data, init_data_size, builtin_data,
      registration, node_index);
}

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Interpreter::ResizeInputTensorStrict(
    int tensor_index, const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensorStrict(tensor_index, dims);
}

TfLiteStatus Interpreter::Invoke() {
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(root_profiler_.get(),
                                                           "invoke");

  // Denormal floating point numbers could cause significant slowdown on
  // platforms like x86, therefore, we suppress denormals here to prevent this
  // from happening.
  ruy::ScopedSuppressDenormals suppress_denormals;

  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
      scoped_runtime_event, primary_subgraph().Invoke());

  if (!allow_buffer_handle_output_) {
    for (int tensor_index : outputs()) {
      TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
          scoped_runtime_event,
          primary_subgraph().EnsureTensorDataIsReadable(tensor_index));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddTensors(int tensors_to_add,
                                     int* first_new_tensor_index) {
  return primary_subgraph().AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, dims.size(), dims.data(), quantization, buffer,
      bytes, allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    bool is_variable) {
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, dims.size(), dims.data(), quantization,
      is_variable);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, const char* buffer,
    size_t bytes, const Allocation* allocation) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, rank, dims, new_quantization, buffer, bytes,
      allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, rank, dims, new_quantization, is_variable,
      rank_dims_signature, dims_signature);
}

TfLiteStatus Interpreter::SetExecutionPlan(const std::vector<int>& new_plan) {
  return primary_subgraph().SetExecutionPlan(new_plan);
}

TfLiteStatus Interpreter::SetNumThreads(int num_threads) {
  if (num_threads < -1) {
    context_->ReportError(context_,
                          "num_threads should be >=0 or just -1 to let TFLite "
                          "runtime set the value.");
    return kTfLiteError;
  }

  // num_threads == 0 has the same effect as num_threads == 1.
  num_threads = num_threads == 0 ? 1 : num_threads;
  for (auto& subgraph : subgraphs_) {
    subgraph->context()->recommended_num_threads = num_threads;
  }

  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    auto* c = external_contexts_[i];
    if (c && c->Refresh) {
      c->Refresh(context_);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::ApplyLazyDelegateProviders() {
  if (lazy_delegate_providers_.empty() || IsFullyDelegated()) return kTfLiteOk;

  // We only apply lazy delegate providers once.
  TfLiteDelegateCreators delegate_providers;
  delegate_providers.swap(lazy_delegate_providers_);

  TFLITE_LOG(TFLITE_LOG_INFO,
             "Applying %zu TensorFlow Lite delegate(s) lazily.",
             delegate_providers.size());
  // At the momement, XNNPACK delegate is the only one that might be applied
  // by default, in which case, the execution will fall back to default
  // implementation if the XNNPACK delegate fails to be applied.
  for (size_t i = 0; i < delegate_providers.size(); ++i) {
    auto delegate_ptr =
        delegate_providers[i](context_->recommended_num_threads);
    // Note when XNNPACK-by-default is disabled, the corresponding creator (i.e.
    // tflite::MaybeCreateXNNPACKDelegate(...)) will return a nullptr.
    // Therefore, we simply continue with the next one.
    if (delegate_ptr == nullptr) continue;
    auto status = ModifyGraphWithDelegateImpl(std::move(delegate_ptr));
    switch (status) {
      case kTfLiteOk:
        TFLITE_LOG(
            TFLITE_LOG_INFO,
            "Successfully applied the default TensorFlow Lite "
            "delegate indexed at %zu.\n *NOTE*: because a delegate has been "
            "applied, the precision of computations should be unchanged, but "
            "the exact output tensor values may have changed. If such output "
            "values are checked in your code, like in your tests etc., please "
            "consider increasing error tolerance for the check.",
            i);
        break;
      case kTfLiteError:
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Failed to apply the default TensorFlow Lite "
                             "delegate indexed at %zu.",
                             i);
        return kTfLiteError;
      case kTfLiteDelegateError:
        TFLITE_LOG(
            TFLITE_LOG_INFO,
            "Error in applying the default TensorFlow Lite delegate indexed "
            "at %zu, and all previously applied delegates are reverted.",
            i);
        return kTfLiteDelegateError;
      case kTfLiteApplicationError:
        TFLITE_LOG(
            TFLITE_LOG_INFO,
            "Failed to apply the default TensorFlow Lite delegate indexed at "
            "%zu because of incompatibility between runtime and delegate. "
            "Ignoring the error, and continuing anyway.",
            i);
        return kTfLiteApplicationError;
      case kTfLiteUnresolvedOps:
        TFLITE_LOG(
            TFLITE_LOG_INFO,
            "Failed to apply the default TensorFlow Lite delegate indexed at "
            "%zu because of unresolved ops (which could be resolved by "
            "another delegate). Ignoring the error, and continuing anyway.",
            i);
        return kTfLiteUnresolvedOps;
      default:
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Unknown status (%d) after applying the default "
                             "TensorFlow Lite delegate indexed at %zu.",
                             status, i);
        return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::ModifyGraphWithDelegateImpl(
    TfLiteDelegate* delegate) {
  TfLiteStatus status = kTfLiteOk;
  for (auto& subgraph : subgraphs_) {
    if (IsValidationSubgraph(subgraph->GetName().c_str())) {
      continue;
    }
    status = subgraph->ModifyGraphWithDelegate(delegate);
    if (status != kTfLiteOk) {
      break;
    }
  }
  // Delegate-specific errors can be recovered from by restoring Interpreter to
  // its original state.
  if (status == kTfLiteDelegateError) {
    TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
  }
  return status;
}

TfLiteStatus Interpreter::RemoveAllDelegates() {
  for (auto& subgraph : subgraphs_) {
    TF_LITE_ENSURE_STATUS(subgraph->RemoveAllDelegates());
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::SetMetadata(
    const std::map<std::string, std::string>& metadata) {
  metadata_ = metadata;
  for (int subgraph_index = 0; subgraph_index < subgraphs_.size();
       ++subgraph_index) {
    TF_LITE_ENSURE_STATUS(subgraphs_[subgraph_index]->SetMetadata(&metadata_));
  }
  return kTfLiteOk;
}

bool Interpreter::IsFullyDelegated() const {
  return primary_subgraph().IsFullyDelegated();
}

void Interpreter::SetProfilerImpl(std::unique_ptr<Profiler> profiler) {
  if (profiler == nullptr) {
    root_profiler_ = nullptr;
    return;
  }
  if (root_profiler_ == nullptr) {
    root_profiler_ = std::make_unique<profiling::RootProfiler>();
  } else {
    // Removes all previously registered profilers.
    root_profiler_->RemoveChildProfilers();
  }
  root_profiler_->AddProfiler(std::move(profiler));
  SetSubgraphProfiler();
}

void Interpreter::SetSubgraphProfiler() {
  for (int subgraph_index = 0; subgraph_index < subgraphs_.size();
       ++subgraph_index) {
    subgraphs_[subgraph_index]->SetProfiler(root_profiler_.get(),
                                            subgraph_index);
  }
}

TfLiteStatus Interpreter::ApplyOptionsImpl(InterpreterOptions* options) {
  if (options == nullptr) {
    return kTfLiteOk;
  }
  options_ = std::make_unique<InterpreterOptions>(*options);

  // Set InterpreterOptions object to SubGraph.
  for (auto& subgraph : subgraphs_) {
    subgraph->SetOptions(options_.get());
  }

  // Handle `experimental_dynamic_allocation_for_large_tensors_`.
  if (options->GetDynamicAllocationForLargeTensors() > 0) {
    for (auto& subgraph : subgraphs_) {
      subgraph->OptimizeMemoryForLargeTensors(
          options->GetDynamicAllocationForLargeTensors());
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite
