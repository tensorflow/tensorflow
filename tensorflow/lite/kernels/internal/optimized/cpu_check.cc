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

#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"

#if defined __linux__ && defined __aarch64__
#include <sys/auxv.h>
#endif

namespace tflite {

namespace {

// The implementation of dotprod detection is copied from ruy's internal
// function DetectDotprod().
// At the moment it's only implemented on Linux ARM64. Consider syncing again
// with ruy in the future to share improvements.
#if defined __linux__ && defined __aarch64__
bool DetectDotprodByLinuxAuxvMethod() {
  // This is the value of HWCAP_ASIMDDP in sufficiently recent Linux headers,
  // however we need to support building against older headers for the time
  // being.
  const int kLocalHwcapAsimddp = 1 << 20;
  return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
}
#endif

}  // namespace

bool DetectArmNeonDotprod() {
#if defined __linux__ && defined __aarch64__
  return DetectDotprodByLinuxAuxvMethod();
#endif

  return false;
}

}  // namespace tflite
