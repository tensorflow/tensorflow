/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/random.h"

#include <cmath>
#include <limits>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {

xla::XlaOp TruncatedNormal(xla::XlaOp uniform) {
  auto normal_cdf = [](double x) {
    return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
  };

  const double kA = -2.0;
  const double kB = 2.0;
  const double kMu = 0.0;
  const double kSigma = 1.0;
  const double kAlpha = (kA - kMu) / kSigma;
  const double kBeta = (kB - kMu) / kSigma;
  const double kAlphaNormalCdf = normal_cdf(kAlpha);
  const double kBetaNormalCdf = normal_cdf(kBeta);
  const double kZ = kBetaNormalCdf - kAlphaNormalCdf;

  xla::XlaOp one = xla::ScalarLike(uniform, 1.0);
  xla::XlaOp two = xla::ScalarLike(uniform, 2.0);
  xla::XlaOp sqrt_2 = xla::ScalarLike(uniform, std::sqrt(2.0));
  xla::XlaOp z = xla::ScalarLike(uniform, kZ);
  xla::XlaOp alpha_normal_cdf = xla::ScalarLike(uniform, kAlphaNormalCdf);

  auto p = alpha_normal_cdf + z * uniform;
  // probit(p) = sqrt(2) * erfinv(2*p-1)
  return sqrt_2 * xla::ErfInv(two * p - one);
}

}  // namespace tensorflow
