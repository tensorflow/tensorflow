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
// Check EIGEN_VECTORIZE_SSE3 since Windows doesn't define __SSE3__ properly
#ifdef EIGEN_VECTORIZE_SSE3
#include <pmmintrin.h>
#endif

namespace tensorflow {
namespace port {

ScopedFlushDenormal::ScopedFlushDenormal() {
// For now, we flush denormals only on SSE 3.  Other architectures such as ARM
// can be added as needed.
#ifdef EIGEN_VECTORIZE_SSE3
  // Save existing flags
  flush_zero_mode_ = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
  denormals_zero_mode_ = _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;

  // Flush denormals to zero (the FTZ flag).
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  // Interpret denormal inputs as zero (the DAZ flag).
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

ScopedFlushDenormal::~ScopedFlushDenormal() {
#ifdef EIGEN_VECTORIZE_SSE3
  // Restore flags
  _MM_SET_FLUSH_ZERO_MODE(flush_zero_mode_ ? _MM_FLUSH_ZERO_ON
                                           : _MM_FLUSH_ZERO_OFF);
  _MM_SET_DENORMALS_ZERO_MODE(denormals_zero_mode_ ? _MM_DENORMALS_ZERO_ON
                                                   : _MM_DENORMALS_ZERO_OFF);
#endif
}

}  // namespace port
}  // namespace tensorflow
