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

#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {

xla::XlaOp TruncatedNormal(xla::XlaOp uniform) {
  const double kA = -2.0;
  const double kB = 2.0;
  const double kMu = 0.0;
  const double kSigma = 1.0;
  return ParameterizedTruncatedNormal(
      uniform, xla::ScalarLike(uniform, kMu), xla::ScalarLike(uniform, kSigma),
      xla::ScalarLike(uniform, kA), xla::ScalarLike(uniform, kB));
}

// Implements the sampling of truncated normal distribution using the
// inversed cumulative distribution function (CDF) method as described in
// https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf.
xla::XlaOp ParameterizedTruncatedNormal(xla::XlaOp uniform, xla::XlaOp mu,
                                        xla::XlaOp sigma, xla::XlaOp a,
                                        xla::XlaOp b) {
  xla::XlaOp one = xla::ScalarLike(uniform, 1.0);
  xla::XlaOp two = xla::ScalarLike(uniform, 2.0);
  xla::XlaOp sqrt_2 = xla::ScalarLike(uniform, std::sqrt(2.0));

  auto normal_cdf = [&](xla::XlaOp x) {
    return (one + xla::Erf(x / sqrt_2)) / two;
  };

  // Calculate the cumulative probabilities for the lower and upper bound, a and
  // b.
  xla::XlaOp alpha = (a - mu) / sigma;
  xla::XlaOp beta = (b - mu) / sigma;
  xla::XlaOp alpha_normal_cdf = normal_cdf(alpha);
  xla::XlaOp beta_normal_cdf = normal_cdf(beta);

  // Convert the random uniform value in range (0, 1) (uniform) to a value in
  // range (alpha_normal_cdf, beta_normal_cdf) that represents the cumulative
  // probability (p) of a value (x) in the truncated normal distribution.
  xla::XlaOp p =
      alpha_normal_cdf + (beta_normal_cdf - alpha_normal_cdf) * uniform;

  // Calculate x using the inversed cumulative distribution function:
  //   x = inversed_cdf(mu, sigma; p) = mu + sigma * sqrt(2) * erfinv(2*p-1)
  // Clamp the input of erfinv to (-1, 1) because 2*p-1 may produce +/-1 due to
  // computation precision.
  xla::XlaOp v = two * p - one;
  xla::PrimitiveType primitive_type =
      uniform.builder()->GetShape(uniform).value().element_type();
  xla::XlaOp epsilon = xla::Epsilon(uniform.builder(), primitive_type);
  v = xla::Clamp(-one + epsilon, v, one - epsilon);
  xla::XlaOp x = mu + sigma * sqrt_2 * xla::ErfInv(v);

  // The value of x may be out of the range of (a, b), this typically happens
  // when the region of the truncated normal has a very small probability.
  x = xla::Clamp(a, x, b);

  return x;
}

}  // namespace tensorflow
