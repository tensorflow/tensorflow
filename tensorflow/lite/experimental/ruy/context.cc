/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/context.h"

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/detect_arm.h"
#include "tensorflow/lite/experimental/ruy/detect_x86.h"
#include "tensorflow/lite/experimental/ruy/have_built_path_for.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {

void Context::SetRuntimeEnabledPaths(Path paths) {
  runtime_enabled_paths_ = paths;
}

Path Context::GetRuntimeEnabledPaths() {
  // This function should always return the same value on a given machine.
  // When runtime_enabled_paths_ has its initial value kNone, it performs
  // some platform detection to resolve it to specific Path values.

  // Fast path: already resolved.
  if (runtime_enabled_paths_ != Path::kNone) {
    return runtime_enabled_paths_;
  }

  // Need to resolve now. Start by considering all paths enabled.
  runtime_enabled_paths_ = kAllPaths;

  // This mechanism is intended to be used for testing and benchmarking. For
  // example, one can set RUY_FORCE_DISABLE_PATHS to Path::kAvx512 in order to
  // evaluate AVX2 performance on an AVX-512 machine.
#ifdef RUY_FORCE_DISABLE_PATHS
  runtime_enabled_paths_ = runtime_enabled_paths_ & ~(RUY_FORCE_DISABLE_PATHS);
#endif

#if RUY_PLATFORM(ARM)
  // Now selectively disable paths that aren't supported on this machine.
  if ((runtime_enabled_paths_ & Path::kNeonDotprod) != Path::kNone) {
    if (!DetectDotprod()) {
      runtime_enabled_paths_ = runtime_enabled_paths_ & ~Path::kNeonDotprod;
      // Sanity check.
      RUY_DCHECK((runtime_enabled_paths_ & Path::kNeonDotprod) == Path::kNone);
    }
  }
#endif  // RUY_PLATFORM(ARM)

#if RUY_PLATFORM(X86)
  if ((runtime_enabled_paths_ & Path::kAvx2) != Path::kNone) {
    if (!(HaveBuiltPathForAvx2() && DetectCpuAvx2())) {
      runtime_enabled_paths_ = runtime_enabled_paths_ & ~Path::kAvx2;
      // Sanity check.
      RUY_DCHECK((runtime_enabled_paths_ & Path::kAvx2) == Path::kNone);
    }
  }

  if ((runtime_enabled_paths_ & Path::kAvx512) != Path::kNone) {
    if (!(HaveBuiltPathForAvx512() && DetectCpuAvx512())) {
      runtime_enabled_paths_ = runtime_enabled_paths_ & ~Path::kAvx512;
      // Sanity check.
      RUY_DCHECK((runtime_enabled_paths_ & Path::kAvx512) == Path::kNone);
    }
  }
#endif  // RUY_PLATFORM(X86)

  // Sanity check. We can't possibly have disabled all paths, as some paths
  // are universally available (kReference, kStandardCpp).
  RUY_DCHECK_NE(runtime_enabled_paths_, Path::kNone);
  return runtime_enabled_paths_;
}

}  // namespace ruy
