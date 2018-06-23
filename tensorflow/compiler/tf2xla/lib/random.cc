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
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {
xla::StatusOr<xla::XlaOp> TruncatedNormal(const DataType dtype,
                                          const xla::XlaOp& uniform,
                                          xla::XlaBuilder* builder) {
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

  xla::XlaOp one = XlaHelpers::FloatLiteral(builder, dtype, 1.0);
  xla::XlaOp two = XlaHelpers::FloatLiteral(builder, dtype, 2.0);
  xla::XlaOp sqrt_2 = XlaHelpers::FloatLiteral(builder, dtype, std::sqrt(2.0));

  xla::XlaOp z = XlaHelpers::FloatLiteral(builder, dtype, kZ);
  xla::XlaOp alpha_normal_cdf =
      XlaHelpers::FloatLiteral(builder, dtype, kAlphaNormalCdf);

  // probit(p) = sqrt(2) * erfinv(2*p-1)
  auto p = builder->Add(alpha_normal_cdf, builder->Mul(z, uniform));
  auto erfinv_input = builder->Sub(builder->Mul(p, two), one);
  TF_ASSIGN_OR_RETURN(auto erfinv_or_status, ErfInv(builder, erfinv_input));
  return builder->Mul(sqrt_2, erfinv_or_status);
}
}  // namespace tensorflow
