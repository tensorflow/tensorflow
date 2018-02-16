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

#include "tensorflow/core/platform/denormal.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
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
#define DENORM_USE_INTRINSICS
#endif

#ifdef DENORM_USE_INTRINSICS
#include <pmmintrin.h>
#endif

namespace tensorflow {
namespace port {

ScopedFlushDenormal::ScopedFlushDenormal() {
  // For now, we flush denormals only on SSE 3.  Other architectures such as ARM
  // can be added as needed.

#ifdef DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    // Save existing flags
    flush_zero_mode_ = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
    denormals_zero_mode_ =
        _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;

    // Flush denormals to zero (the FTZ flag).
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // Interpret denormal inputs as zero (the DAZ flag).
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
#endif
}

ScopedFlushDenormal::~ScopedFlushDenormal() {
#ifdef DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    // Restore flags
    _MM_SET_FLUSH_ZERO_MODE(flush_zero_mode_ ? _MM_FLUSH_ZERO_ON
                                             : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(denormals_zero_mode_ ? _MM_DENORMALS_ZERO_ON
                                                     : _MM_DENORMALS_ZERO_OFF);
  }
#endif
}

}  // namespace port
}  // namespace tensorflow
