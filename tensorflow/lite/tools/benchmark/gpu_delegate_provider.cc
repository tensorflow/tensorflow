/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/delegate_provider.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR
// Only enable metal delegate when using a real iPhone device.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
#endif

namespace tflite {
namespace benchmark {

class GpuDelegateProvider : public DelegateProvider {
 public:
  std::vector<Flag> CreateFlags(BenchmarkParams* params) const final;

  void AddParams(BenchmarkParams* params) const final;

  void LogParams(const BenchmarkParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(
      const BenchmarkParams& params) const final;

  std::string GetName() const final { return "GPU"; }
};
REGISTER_DELEGATE_PROVIDER(GpuDelegateProvider);

std::vector<Flag> GpuDelegateProvider::CreateFlags(
    BenchmarkParams* params) const {
  std::vector<Flag> flags = {
    CreateFlag<bool>("use_gpu", params, "use gpu"),
#if defined(__ANDROID__) || defined(REAL_IPHONE_DEVICE)
    CreateFlag<bool>("gpu_precision_loss_allowed", params,
                     "Allow to process computation in lower precision than "
                     "FP32 in GPU. By default, it's enabled."),
#endif
#if defined(REAL_IPHONE_DEVICE)
    CreateFlag<std::string>(
        "gpu_wait_type", params,
        "GPU wait type. Should be one of the following: passive, active, "
        "do_not_wait, aggressive"),
#endif
  };
  return flags;
}

void GpuDelegateProvider::AddParams(BenchmarkParams* params) const {
  params->AddParam("use_gpu", BenchmarkParam::Create<bool>(false));
#if defined(__ANDROID__) || defined(REAL_IPHONE_DEVICE)
  params->AddParam("gpu_precision_loss_allowed",
                   BenchmarkParam::Create<bool>(true));
#endif
#if defined(REAL_IPHONE_DEVICE)
  params->AddParam("gpu_wait_type", BenchmarkParam::Create<std::string>(""));
#endif
}

void GpuDelegateProvider::LogParams(const BenchmarkParams& params) const {
  TFLITE_LOG(INFO) << "Use gpu : [" << params.Get<bool>("use_gpu") << "]";
#if defined(__ANDROID__) || defined(REAL_IPHONE_DEVICE)
  TFLITE_LOG(INFO) << "Allow lower precision in gpu : ["
                   << params.Get<bool>("gpu_precision_loss_allowed") << "]";
#endif
#if defined(REAL_IPHONE_DEVICE)
  TFLITE_LOG(INFO) << "GPU delegate wait type : ["
                   << params.Get<std::string>("gpu_wait_type") << "]";
#endif
}

TfLiteDelegatePtr GpuDelegateProvider::CreateTfLiteDelegate(
    const BenchmarkParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});

  if (params.Get<bool>("use_gpu")) {
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    if (params.Get<bool>("gpu_precision_loss_allowed")) {
      gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      gpu_opts.inference_priority2 =
          TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      gpu_opts.inference_priority3 =
          TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }
    delegate = evaluation::CreateGPUDelegate(&gpu_opts);
#elif defined(REAL_IPHONE_DEVICE)
    TFLGpuDelegateOptions gpu_opts = {0};
    gpu_opts.allow_precision_loss =
        params.Get<bool>("gpu_precision_loss_allowed");

    std::string string_gpu_wait_type = params.Get<std::string>("gpu_wait_type");
    if (!string_gpu_wait_type.empty()) {
      TFLGpuDelegateWaitType wait_type = TFLGpuDelegateWaitTypePassive;
      if (string_gpu_wait_type == "passive") {
        wait_type = TFLGpuDelegateWaitTypePassive;
      } else if (string_gpu_wait_type == "active") {
        wait_type = TFLGpuDelegateWaitTypeActive;
      } else if (string_gpu_wait_type == "do_not_wait") {
        wait_type = TFLGpuDelegateWaitTypeDoNotWait;
      } else if (string_gpu_wait_type == "aggressive") {
        wait_type = TFLGpuDelegateWaitTypeAggressive;
      }
      gpu_opts.wait_type = wait_type;
    }
    delegate = TfLiteDelegatePtr(TFLGpuDelegateCreate(&gpu_opts),
                                 &TFLGpuDelegateDelete);
#else
    TFLITE_LOG(WARN) << "The GPU delegate compile options are only supported "
                        "to be benchmarked on Android or iOS platforms.";
    delegate = evaluation::CreateGPUDelegate();
#endif

    if (!delegate.get()) {
      TFLITE_LOG(WARN) << "GPU acceleration is unsupported on this platform.";
    }
  }
  return delegate;
}

}  // namespace benchmark
}  // namespace tflite
