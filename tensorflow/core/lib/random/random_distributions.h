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

#ifndef TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
#define TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/lib/random/random_distributions.h"

namespace tensorflow {
namespace random {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::random::BoxMullerDouble;
using tsl::random::NormalDistribution;
using tsl::random::SignedAdd;
using tsl::random::SingleSampleAdapter;
using tsl::random::TruncatedNormalDistribution;
using tsl::random::Uint16ToGfloat16;
using tsl::random::Uint16ToHalf;
using tsl::random::UniformDistribution;
using tsl::random::UniformFullIntDistribution;
// NOLINTEND(misc-unused-using-decls)
}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
