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
#include <utility>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if TFLITE_SUPPORTS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if (TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR) || \
    (TARGET_OS_OSX && TARGET_CPU_ARM64)
// Only enable metal delegate when using a real iPhone device or Apple Silicon.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
#endif

namespace tflite {
namespace tools {

class GpuDelegateProvider : public DelegateProvider {
 public:
  GpuDelegateProvider() {
    default_params_.AddParam("use_gpu", ToolParam::Create<bool>(false));
#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)
    default_params_.AddParam("gpu_precision_loss_allowed",
                             ToolParam::Create<bool>(true));
    default_params_.AddParam("gpu_experimental_enable_quant",
                             ToolParam::Create<bool>(true));
#endif
#if TFLITE_SUPPORTS_GPU_DELEGATE
    default_params_.AddParam("gpu_inference_for_sustained_speed",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("gpu_backend", ToolParam::Create<std::string>(""));
#endif
#if defined(REAL_IPHONE_DEVICE)
    default_params_.AddParam("gpu_wait_type",
                             ToolParam::Create<std::string>(""));
#endif
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "GPU"; }
};
REGISTER_DELEGATE_PROVIDER(GpuDelegateProvider);

std::vector<Flag> GpuDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {
    CreateFlag<bool>("use_gpu", params, "use gpu"),
#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)
    CreateFlag<bool>("gpu_precision_loss_allowed", params,
                     "Allow to process computation in lower precision than "
                     "FP32 in GPU. By default, it's enabled."),
    CreateFlag<bool>("gpu_experimental_enable_quant", params,
                     "Whether to enable the GPU delegate to run quantized "
                     "models or not. By default, it's enabled."),
#endif
#if TFLITE_SUPPORTS_GPU_DELEGATE
    CreateFlag<bool>("gpu_inference_for_sustained_speed", params,
                     "Whether to prefer maximizing the throughput. This mode "
                     "will help when the same delegate will be used repeatedly "
                     "on multiple inputs. This is supported on non-iOS "
                     "platforms. By default, it's disabled."),
    CreateFlag<std::string>(
        "gpu_backend", params,
        "Force the GPU delegate to use a particular backend for execution, and "
        "fail if unsuccessful. Should be one of: cl, gl"),
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

void GpuDelegateProvider::LogParams(const ToolParams& params,
                                    bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_gpu", "Use gpu", verbose);
#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)
  LOG_TOOL_PARAM(params, bool, "gpu_precision_loss_allowed",
                 "Allow lower precision in gpu", verbose);
  LOG_TOOL_PARAM(params, bool, "gpu_experimental_enable_quant",
                 "Enable running quant models in gpu", verbose);
#endif
#if TFLITE_SUPPORTS_GPU_DELEGATE
  LOG_TOOL_PARAM(params, bool, "gpu_inference_for_sustained_speed",
                 "Prefer maximizing the throughput in gpu", verbose);
  LOG_TOOL_PARAM(params, std::string, "gpu_backend", "GPU backend", verbose);
#endif
#if defined(REAL_IPHONE_DEVICE)
  LOG_TOOL_PARAM(params, std::string, "gpu_wait_type", "GPU delegate wait type",
                 verbose);
#endif
}

TfLiteDelegatePtr GpuDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});

  if (params.Get<bool>("use_gpu")) {
#if TFLITE_SUPPORTS_GPU_DELEGATE
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    if (params.Get<bool>("gpu_precision_loss_allowed")) {
      gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      gpu_opts.inference_priority2 =
          TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      gpu_opts.inference_priority3 =
          TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }
    if (params.Get<bool>("gpu_experimental_enable_quant")) {
      gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
    }
    if (params.Get<bool>("gpu_inference_for_sustained_speed")) {
      gpu_opts.inference_preference =
          TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    }
    std::string gpu_backend = params.Get<std::string>("gpu_backend");
    if (!gpu_backend.empty()) {
      if (gpu_backend == "cl") {
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
      } else if (gpu_backend == "gl") {
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
      }
    }
    gpu_opts.max_delegated_partitions =
        params.Get<int>("max_delegated_partitions");

    // Serialization.
    std::string serialize_dir =
        params.Get<std::string>("delegate_serialize_dir");
    std::string serialize_token =
        params.Get<std::string>("delegate_serialize_token");
    if (!serialize_dir.empty() && !serialize_token.empty()) {
      gpu_opts.experimental_flags =
          gpu_opts.experimental_flags |
          TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
      gpu_opts.serialization_dir = serialize_dir.c_str();
      gpu_opts.model_token = serialize_token.c_str();
    }

    delegate = evaluation::CreateGPUDelegate(&gpu_opts);
#elif defined(REAL_IPHONE_DEVICE)
    TFLGpuDelegateOptions gpu_opts = {0};
    gpu_opts.allow_precision_loss =
        params.Get<bool>("gpu_precision_loss_allowed");
    gpu_opts.enable_quantization =
        params.Get<bool>("gpu_experimental_enable_quant");

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
                        "on Android or iOS platforms or when the tool was "
                        "built with -DCL_DELEGATE_NO_GL.";
    delegate = evaluation::CreateGPUDelegate();
#endif

    if (!delegate.get()) {
      TFLITE_LOG(WARN) << "GPU acceleration is unsupported on this platform.";
    }
  }
  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
GpuDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<bool>("use_gpu"));
}
}  // namespace tools
}  // namespace tflite
