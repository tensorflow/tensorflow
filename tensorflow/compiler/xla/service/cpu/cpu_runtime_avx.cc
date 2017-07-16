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

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_avx.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"

namespace xla {
namespace cpu {
namespace runtime {

#ifdef __AVX__
V8F32 ExpV8F32(V8F32 x) { return Eigen::internal::pexp(x); }

V8F32 LogV8F32(V8F32 x) { return Eigen::internal::plog(x); }

V8F32 TanhV8F32(V8F32 x) { return Eigen::internal::ptanh(x); }
#endif  // __AVX__

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
