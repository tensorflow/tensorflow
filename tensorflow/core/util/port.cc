/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/port.h"

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

bool IsGoogleCudaEnabled() {
#if GOOGLE_CUDA
  return true;
#else
  return false;
#endif
}

bool IsBuiltWithROCm() {
#if TENSORFLOW_USE_ROCM
  return true;
#else
  return false;
#endif
}

bool IsBuiltWithXLA() {
#if TENSORFLOW_USE_XLA
  return true;
#else
  return false;
#endif
}

bool IsBuiltWithNvcc() {
#if TENSORFLOW_USE_NVCC
  return true;
#else
  return false;
#endif
}

bool IsAArch32Available() {
#if TF_LLVM_AARCH32_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool IsAArch64Available() {
#if TF_LLVM_AARCH64_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool IsPowerPCAvailable() {
#if TF_LLVM_POWERPC_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool IsSystemZAvailable() {
#if TF_LLVM_S390X_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool IsX86Available() {
#if TF_LLVM_X86_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool GpuSupportsHalfMatMulAndConv() {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  return true;
#else
  return false;
#endif
}

// Returns whether oneDNN should be enabled or disabled by default.
//   - Linux: Enabled by default for CPUs with neural network features.
//   - Windows: Disabled by default.
inline bool DefaultOneDnnPolicy() {
#if !defined(INTEL_MKL)
  return false;
#elif defined(PLATFORM_GOOGLE)
  return true;
#elif defined(PLATFORM_WINDOWS) && defined(PLATFORM_IS_X86)
  return true;
#elif defined(__linux__)
  return port::TestCPUFeature(port::CPUFeature::AVX512_VNNI) ||
         port::TestCPUFeature(port::CPUFeature::AVX512_BF16) ||
         port::TestCPUFeature(port::CPUFeature::AVX_VNNI) ||
         port::TestCPUFeature(port::CPUFeature::AMX_TILE) ||
         port::TestCPUFeature(port::CPUFeature::AMX_INT8) ||
         port::TestCPUFeature(port::CPUFeature::AMX_BF16) ||
         port::TestAarch64CPU(
             port::Aarch64CPU::ARM_NEOVERSE_V1);  // ARM NEOVERSE V1
#else
  return false;
#endif  // !defined(INTEL_MKL)
}

bool IsMklEnabled() {
#ifndef INTEL_MKL
  return false;
#endif
  static absl::once_flag once;  // NOLINT(clang-diagnostic-unreachable-code)
#ifdef ENABLE_MKL
  // Keeping TF_DISABLE_MKL env variable for legacy reasons.
  static bool oneDNN_disabled = false;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_DISABLE_MKL", false, &oneDNN_disabled));
    if (oneDNN_disabled) VLOG(2) << "TF-MKL: Disabling oneDNN";
  });
  return (!oneDNN_disabled);
#else
  static bool oneDNN_enabled = DefaultOneDnnPolicy();
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar("TF_ENABLE_ONEDNN_OPTS", oneDNN_enabled,
                                     &oneDNN_enabled);
    if (!status.ok()) {
      LOG(WARNING) << "TF_ENABLE_ONEDNN_OPTS is not set to either '0', 'false',"
                   << " '1', or 'true'. Using the default setting: "
                   << oneDNN_enabled;
    }
    if (oneDNN_enabled) {
      LOG(INFO) << "oneDNN custom operations are on. "
                << "You may see slightly different numerical results due to "
                << "floating-point round-off errors from different computation "
                << "orders. To turn them off, set the environment variable "
                << "`TF_ENABLE_ONEDNN_OPTS=0`.";
    }
  });
  return oneDNN_enabled;
#endif  // ENABLE_MKL
}

bool IsZenDnnEnabled() {
#ifndef AMD_ZENDNN
  return false;
#else
  static absl::once_flag once;
  static bool ZenDNN_enabled = false;
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar("TF_ENABLE_ZENDNN_OPTS", ZenDNN_enabled,
                                     &ZenDNN_enabled);

    if (!status.ok()) {
      LOG(WARNING) << "TF_ENABLE_ZENDNN_OPTS is not set to either '0', 'false',"
                   << " '1', or 'true'. Using the default setting: "
                   << ZenDNN_enabled;
    }
    if (ZenDNN_enabled) {
      LOG(INFO) << "ZenDNN custom operations are on. "
                << "You may see slightly different numerical results due to "
                << "floating-point round-off errors from different computation "
                << "orders. To turn them off, set the environment variable "
                << "`TF_ENABLE_ZENDNN_OPTS=0`.";
    }
  });
  return ZenDNN_enabled;
#endif  // !AMD_ZENDNN
}

}  // namespace tensorflow
