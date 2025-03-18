// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/build_stamp.h"
#include "tensorflow/lite/experimental/litert/runtime/dispatch/dispatch_delegate_kernel.h"
#include "tensorflow/lite/experimental/litert/runtime/dispatch/dispatch_delegate_options.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

namespace {

using ::litert::internal::kLiteRtDispatchOpCustomCode;

// A TFL Delegate that can recognize subgraphs that run on Dispatch API capable
// accelerators, e.g. TPU, DSP, ... It replaces such subgraphs and offloads
// their work through the Dispatch API.
class DispatchDelegate : public tflite::SimpleOpaqueDelegateInterface {
 public:
  static TfLiteOpaqueDelegate* Create(LiteRtDispatchDelegateOptions* options_) {
    litert::DispatchDelegateOptionsPtr options(
        options_, LiteRtDestroyDispatchDelegateOptions);
    if (!options) {
      LITERT_LOG(LITERT_ERROR, "Null input");
      return nullptr;
    }

    std::unique_ptr<DispatchDelegate> managed_sb_delegate(
        new DispatchDelegate(std::move(options)));
    return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
        std::move(managed_sb_delegate),
        kTfLiteDelegateFlagsAllowDynamicTensors);
  }

  bool IsNodeSupportedByDelegate(const TfLiteOperator* op,
                                 const TfLiteOpaqueNode* node,
                                 TfLiteOpaqueContext* context) const override;

  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override;

  const char* Name() const override;

  std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override;

  TfLiteStatus StartMetricsCollection(int detail_level);

  TfLiteStatus StopMetricsCollection(LiteRtDispatchDelegateMetricsT& metrics);

 private:
  static constexpr absl::string_view kDelegateName = "DispatchDelegate";

  explicit DispatchDelegate(litert::DispatchDelegateOptionsPtr&& options)
      : options_(std::move(options)) {}

  litert::DispatchDelegateOptionsPtr options_;
  int dispatch_graph_name_id_ = 0;
  std::vector<litert::internal::DispatchDelegateKernel*> kernels_;
};

bool DispatchDelegate::IsNodeSupportedByDelegate(
    const TfLiteOperator* op, const TfLiteOpaqueNode* node,
    TfLiteOpaqueContext* context) const {
  auto custom_code = absl::string_view(TfLiteOperatorGetCustomName(op));
  return custom_code == kLiteRtDispatchOpCustomCode;
}

TfLiteStatus DispatchDelegate::Initialize(TfLiteOpaqueContext* context) {
  return kTfLiteOk;
}

const char* DispatchDelegate::Name() const { return kDelegateName.data(); }

std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>
DispatchDelegate::CreateDelegateKernelInterface() {
  std::string dispatch_graph_name =
      absl::StrFormat("DispatchGraph_%d", dispatch_graph_name_id_++);

  auto kernel = litert::internal::DispatchDelegateKernel::Create(
      std::move(dispatch_graph_name), *options_);
  if (kernel) {
    auto* kernel_ptr =
        dynamic_cast<typename litert::internal::DispatchDelegateKernel*>(
            kernel->get());
    kernels_.push_back(kernel_ptr);
    return std::move(*kernel);
  } else {
    LITERT_FATAL("Failed to create a dispatch delegate kernel: %s",
                 kernel.Error().Message().c_str());
    return nullptr;
  }
}

TfLiteStatus DispatchDelegate::StartMetricsCollection(int detail_level) {
  for (auto* kernel : kernels_) {
    if (auto status = kernel->StartMetricsCollection(detail_level);
        status != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to start metrics collection: %d",
                 status);
      return status;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus DispatchDelegate::StopMetricsCollection(
    LiteRtDispatchDelegateMetricsT& metrics) {
  // TODO: b/393453378 - Combine metrics of same type from different kernels.
  for (auto* kernel : kernels_) {
    if (auto status = kernel->StopMetricsCollection(metrics);
        status != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to stop metrics collection: %d", status);
      return status;
    }
  }
  return kTfLiteOk;
}

}  // namespace

LiteRtDispatchDelegateOptions* LiteRtCreateDefaultDispatchDelegateOptions(
    LiteRtEnvironmentOptions environment_options) {
  return new LiteRtDispatchDelegateOptions(environment_options);
}

TfLiteStatus LiteRtAddDispatchDelegateOption(
    LiteRtDispatchDelegateOptions* options, LiteRtDispatchOption option) {
  if (!options) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kTfLiteError;
  }

  options->AddOption(option);
  return kTfLiteOk;
}

TfLiteStatus LiteRtDispatchDelegateAddAllocBaseOption(
    LiteRtDispatchDelegateOptions* options, const void* alloc_base) {
  AddAllocBaseOption(alloc_base, *options);
  return kTfLiteOk;
}

TfLiteStatus LiteRtDispatchDelegateAddAllocFdOption(
    LiteRtDispatchDelegateOptions* options, int alloc_fd) {
  AddAllocFdOption(alloc_fd, *options);
  return kTfLiteOk;
}

void LiteRtDestroyDispatchDelegateOptions(
    LiteRtDispatchDelegateOptions* options) {
  delete options;
}

TfLiteOpaqueDelegate* LiteRtCreateDispatchDelegate(
    LiteRtEnvironmentOptions environment_options,
    LiteRtDispatchDelegateOptions* options) {
  if (!options) {
    options = LiteRtCreateDefaultDispatchDelegateOptions(environment_options);
  }
  return DispatchDelegate::Create(options);
}

void LiteRtDestroyDispatchDelegate(TfLiteOpaqueDelegate* delegate) {
  tflite::TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(delegate);
}

TfLiteStatus LiteRtDispatchDelegateStartMetricsCollection(
    TfLiteOpaqueDelegate* delegate, int detail_level) {
  if (!delegate) return kTfLiteError;
  auto* dispatch_delegate = reinterpret_cast<DispatchDelegate*>(
      TfLiteOpaqueDelegateGetData(delegate));
  return dispatch_delegate->StartMetricsCollection(detail_level);
}

TfLiteStatus LiteRtDispatchDelegateStopMetricsCollection(
    TfLiteOpaqueDelegate* delegate, LiteRtDispatchDelegateMetrics* metrics) {
  if (!delegate) return kTfLiteError;
  auto* dispatch_delegate = reinterpret_cast<DispatchDelegate*>(
      TfLiteOpaqueDelegateGetData(delegate));
  auto dispatch_delegate_metrics =
      std::make_unique<LiteRtDispatchDelegateMetricsT>();
  auto status =
      dispatch_delegate->StopMetricsCollection(*dispatch_delegate_metrics);
  if (status != kTfLiteOk) {
    return status;
  }
  *metrics = dispatch_delegate_metrics.release();
  return kTfLiteOk;
}

TfLiteStatus LiteRtDispatchDelegateGetNumMetrics(
    LiteRtDispatchDelegateMetrics metrics, int* num_metrics) {
  if (!metrics || !num_metrics) {
    return kTfLiteError;
  }
  *num_metrics = metrics->metrics.size();
  return kTfLiteOk;
}

TfLiteStatus LiteRtDispatchDelegateGetMetric(
    LiteRtDispatchDelegateMetrics metrics, int metric_index,
    LiteRtMetric* metric) {
  if (!metrics || !metric) {
    return kTfLiteError;
  }
  if (metric_index < 0 || metric_index >= metrics->metrics.size()) {
    return kTfLiteError;
  }
  auto& dispatch_metric = metrics->metrics[metric_index];
  *metric = {.name = dispatch_metric.name.c_str(),
             .value = dispatch_metric.value};
  return kTfLiteOk;
}

void LiteRtDispatchDelegateDestroyMetrics(
    LiteRtDispatchDelegateMetrics metrics) {
  delete metrics;
}

namespace litert {

DispatchDelegateOptionsPtr CreateDispatchDelegateOptionsPtr(
    LiteRtEnvironmentOptions environment_options) {
  return {LiteRtCreateDefaultDispatchDelegateOptions(environment_options),
          LiteRtDestroyDispatchDelegateOptions};
}

DispatchDelegatePtr CreateDispatchDelegatePtr(
    LiteRtEnvironmentOptions environment_options,
    DispatchDelegateOptionsPtr&& options) {
  return DispatchDelegatePtr(
      LiteRtCreateDispatchDelegate(environment_options, options.release()),
      LiteRtDestroyDispatchDelegate);
}

Expected<void> StartDispatchDelegateMetricsCollection(
    DispatchDelegatePtr& delegate, int detail_level) {
  if (auto status = LiteRtDispatchDelegateStartMetricsCollection(delegate.get(),
                                                                 detail_level);
      status != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to start metrics collection");
  }
  return {};
}

Expected<DispatchDelegateMetricsPtr> StopDispatchDelegateMetricsCollection(
    DispatchDelegatePtr& delegate) {
  LiteRtDispatchDelegateMetricsT* dispatch_delegate_metrics = nullptr;
  if (auto status = LiteRtDispatchDelegateStopMetricsCollection(
          delegate.get(), &dispatch_delegate_metrics);
      status != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to stop metrics collection");
  }
  return DispatchDelegateMetricsPtr(dispatch_delegate_metrics,
                                    LiteRtDispatchDelegateDestroyMetrics);
}

Expected<int> DispatchDelegateGetNumMetrics(
    DispatchDelegateMetricsPtr& metrics) {
  int num_metrics = 0;
  if (auto status =
          LiteRtDispatchDelegateGetNumMetrics(metrics.get(), &num_metrics);
      status != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get number of metrics");
  }
  return num_metrics;
}

Expected<LiteRtMetric> DispatchDelegateGetMetric(
    DispatchDelegateMetricsPtr& metrics, int metric_index) {
  LiteRtMetric metric;
  if (auto status =
          LiteRtDispatchDelegateGetMetric(metrics.get(), metric_index, &metric);
      status != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get metric");
  }
  return metric;
}

}  // namespace litert
