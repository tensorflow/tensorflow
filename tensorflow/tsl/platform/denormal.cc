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

#include "tensorflow/tsl/platform/denormal.h"

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/tsl/platform/platform.h"

// If we're on gcc 4.8 or older, there's a known bug that prevents the use of
// intrinsics when the architecture is not defined in the flags. See
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57202
#if !defined(__SSE3__) && !defined(__clang__) && \
    (defined(__GNUC__) && (__GNUC__ < 4) ||      \
     ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9)))
#define GCC_WITHOUT_INTRINSICS
#endif
// Only try to use SSE3 instructions if we're on an x86 platform, and it's not
// mobile, and we're not on a known bad gcc version.
#if defined(PLATFORM_IS_X86) && !defined(IS_MOBILE_PLATFORM) && \
    !defined(GCC_WITHOUT_INTRINSICS)
#define X86_DENORM_USE_INTRINSICS
#endif

#ifdef X86_DENORM_USE_INTRINSICS
#include <pmmintrin.h>
#endif

// If on ARM, only access the control register if hardware floating-point
// support is available.
#if defined(PLATFORM_IS_ARM) && defined(__ARM_FP) && (__ARM_FP > 0)
#define ARM_DENORM_AVAILABLE
// Flush-to-zero bit on the ARM floating-point control register.
#define ARM_FPCR_FZ (1 << 24)
#endif

namespace tsl {
namespace port {

bool DenormalState::operator==(const DenormalState& other) const {
  return flush_to_zero() == other.flush_to_zero() &&
         denormals_are_zero() == other.denormals_are_zero();
}

bool DenormalState::operator!=(const DenormalState& other) const {
  return !(this->operator==(other));
}

#ifdef ARM_DENORM_AVAILABLE
// Although the ARM ACLE does have a specification for __arm_rsr/__arm_wsr
// for reading and writing to the status registers, they are not implemented
// by GCC, so we need to resort to inline assembly.
static inline void ArmSetFloatingPointControlRegister(uint32_t fpcr) {
#ifdef PLATFORM_IS_ARM64
  __asm__ __volatile__("msr fpcr, %[fpcr]"
                       :
                       : [fpcr] "r"(static_cast<uint64_t>(fpcr)));
#else
  __asm__ __volatile__("vmsr fpscr, %[fpcr]" : : [fpcr] "r"(fpcr));
#endif
}

static inline uint32_t ArmGetFloatingPointControlRegister() {
  uint32_t fpcr;
#ifdef PLATFORM_IS_ARM64
  uint64_t fpcr64;
  __asm__ __volatile__("mrs %[fpcr], fpcr" : [fpcr] "=r"(fpcr64));
  fpcr = static_cast<uint32_t>(fpcr64);
#else
  __asm__ __volatile__("vmrs %[fpcr], fpscr" : [fpcr] "=r"(fpcr));
#endif
  return fpcr;
}
#endif  // ARM_DENORM_AVAILABLE

bool SetDenormalState(const DenormalState& state) {
  // For now, we flush denormals only on SSE 3 and ARM.  Other architectures
  // can be added as needed.

#ifdef X86_DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    // Restore flags
    _MM_SET_FLUSH_ZERO_MODE(state.flush_to_zero() ? _MM_FLUSH_ZERO_ON
                                                  : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(state.denormals_are_zero()
                                    ? _MM_DENORMALS_ZERO_ON
                                    : _MM_DENORMALS_ZERO_OFF);
    return true;
  }
#endif

#ifdef ARM_DENORM_AVAILABLE
  // ARM only has one setting controlling both denormal inputs and outputs.
  if (state.flush_to_zero() == state.denormals_are_zero()) {
    uint32_t fpcr = ArmGetFloatingPointControlRegister();
    if (state.flush_to_zero()) {
      fpcr |= ARM_FPCR_FZ;
    } else {
      fpcr &= ~ARM_FPCR_FZ;
    }
    ArmSetFloatingPointControlRegister(fpcr);
    return true;
  }
#endif

  // Setting denormal handling to the provided state is not supported.
  return false;
}

DenormalState GetDenormalState() {
#ifdef X86_DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    // Save existing flags
    bool flush_zero_mode = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
    bool denormals_zero_mode =
        _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;
    return DenormalState(flush_zero_mode, denormals_zero_mode);
  }
#endif

#ifdef ARM_DENORM_AVAILABLE
  uint32_t fpcr = ArmGetFloatingPointControlRegister();
  if ((fpcr & ARM_FPCR_FZ) != 0) {
    return DenormalState(true, true);
  }
#endif

  return DenormalState(false, false);
}

ScopedRestoreFlushDenormalState::ScopedRestoreFlushDenormalState()
    : denormal_state_(GetDenormalState()) {}

ScopedRestoreFlushDenormalState::~ScopedRestoreFlushDenormalState() {
  SetDenormalState(denormal_state_);
}

ScopedFlushDenormal::ScopedFlushDenormal() {
  SetDenormalState(
      DenormalState(/*flush_to_zero=*/true, /*denormals_are_zero=*/true));
}

ScopedDontFlushDenormal::ScopedDontFlushDenormal() {
  SetDenormalState(
      DenormalState(/*flush_to_zero=*/false, /*denormals_are_zero=*/false));
}

}  // namespace port
}  // namespace tsl
