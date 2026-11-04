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

  // When (alpha, beta) lies on one side of zero, the CDF values above can
  // round to 0 or 1 and the samples collapse onto a bound. Sample those
  // regions from the upper tail (mirrored for the lower side) using log tail
  // probabilities, which keep their precision far from the mean. F16 lacks
  // the range for these probabilities, so the tail is computed in F32.
  xla::PrimitiveType type =
      uniform.builder()->GetShape(uniform).value().element_type();
  xla::PrimitiveType tail_type = type == xla::F16 ? xla::F32 : type;
  xla::XlaOp u = xla::ConvertElementType(uniform, tail_type);
  xla::XlaOp tail_alpha = xla::ConvertElementType(alpha, tail_type);
  xla::XlaOp tail_beta = xla::ConvertElementType(beta, tail_type);
  one = xla::ScalarLike(u, 1.0);
  two = xla::ScalarLike(u, 2.0);
  sqrt_2 = xla::ScalarLike(u, std::sqrt(2.0));
  xla::XlaOp zero = xla::ScalarLike(u, 0.0);
  xla::XlaOp half = xla::ScalarLike(u, 0.5);
  xla::XlaOp log_sqrt_2pi = xla::ScalarLike(u, 0.5 * std::log(2 * M_PI));
  xla::XlaOp series_start = xla::ScalarLike(u, 8.0);
  // log(P(Z > t)) for t >= 0. erfc underflows for large t, so switch to the
  // asymptotic expansion of the Mills ratio there.
  auto log_upper_tail = [&](xla::XlaOp t) {
    xla::XlaOp direct = xla::Log(xla::Erfc(t / sqrt_2) / two);
    xla::XlaOp ts = xla::Max(t, series_start);
    xla::XlaOp z = one / (ts * ts);
    xla::XlaOp series = xla::ScalarLike(u, 10395.0);
    for (double c : {-945.0, 105.0, -15.0, 3.0, -1.0, 1.0}) {
      series = xla::ScalarLike(u, c) + z * series;
    }
    xla::XlaOp asymptotic =
        -half * ts * ts - log_sqrt_2pi - xla::Log(ts) + xla::Log(series);
    return xla::Select(xla::Lt(t, series_start), direct, asymptotic);
  };
  xla::XlaOp upper = xla::Gt(tail_alpha, zero);
  xla::XlaOp lo = xla::Select(upper, tail_alpha, -tail_beta);
  xla::XlaOp hi = xla::Select(upper, tail_beta, -tail_alpha);
  xla::XlaOp log_q_lo = log_upper_tail(lo);
  xla::XlaOp log_q_hi = log_upper_tail(hi);
  xla::XlaOp log_q =
      log_q_lo +
      xla::Log(u + (one - u) * xla::Exp(log_q_hi - log_q_lo));
  xla::XlaOp t = xla::Sqrt(lo * lo - two * (log_q - log_q_lo));
  for (int i = 0; i < 3; ++i) {
    // Newton step on log(P(Z > t)) = log_q.
    xla::XlaOp log_q_t = log_upper_tail(t);
    xla::XlaOp slope = -xla::Exp(-half * t * t - log_sqrt_2pi - log_q_t);
    t = t - (log_q_t - log_q) / slope;
  }
  t = xla::Clamp(lo, t, hi);
  xla::XlaOp tail_x = mu + sigma * xla::ConvertElementType(
                                  xla::Select(upper, t, -t), type);
  xla::XlaOp one_sided = xla::Or(upper, xla::Lt(tail_beta, zero));
  x = xla::Select(one_sided, xla::Clamp(a, tail_x, b), x);

  return x;
}

}  // namespace tensorflow
