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
#include <initializer_list>

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
// https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf,
// and log-tail-probability Newton inversion when both bounds lie on one
// side of the mean.
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
  // regions from the upper tail (mirrored for the lower side) using log
  // tail probabilities, which keep their precision far from the mean. F16
  // and BF16 lack the range and precision for these probabilities, so the
  // tail is computed in F32.
  xla::PrimitiveType tail_type =
      (primitive_type == xla::F16 || primitive_type == xla::BF16)
          ? xla::F32
          : primitive_type;
  xla::XlaOp u = xla::ConvertElementType(uniform, tail_type);
  xla::XlaOp tail_mu = xla::ConvertElementType(mu, tail_type);
  xla::XlaOp tail_sigma = xla::ConvertElementType(sigma, tail_type);
  xla::XlaOp tail_a = xla::ConvertElementType(a, tail_type);
  xla::XlaOp tail_b = xla::ConvertElementType(b, tail_type);
  xla::XlaOp tail_alpha = (tail_a - tail_mu) / tail_sigma;
  xla::XlaOp tail_beta = (tail_b - tail_mu) / tail_sigma;
  xla::XlaOp tail_zero = xla::ScalarLike(u, 0.0);
  xla::XlaOp tail_half = xla::ScalarLike(u, 0.5);
  xla::XlaOp tail_one = xla::ScalarLike(u, 1.0);
  xla::XlaOp tail_two = xla::ScalarLike(u, 2.0);
  xla::XlaOp tail_sqrt_2 = xla::ScalarLike(u, std::sqrt(2.0));
  xla::XlaOp log_sqrt_2pi = xla::ScalarLike(u, 0.5 * std::log(2 * M_PI));
  xla::XlaOp series_start =
      xla::ScalarLike(u, tail_type == xla::F64 ? 26.0 : 8.0);
  // log(P(Z > t)) + 0.5 * t^2 + 0.5 * log(2 * pi) for t >= 0. Factoring
  // out -0.5 * t^2 avoids catastrophic cancellation for large t; erfc
  // underflows for large t, so switch to the Mills-ratio asymptotic
  // expansion at series_start.
  auto log_scaled_upper_tail = [&](xla::XlaOp t) {
    xla::XlaOp td = xla::Min(t, series_start);
    xla::XlaOp direct = xla::Log(xla::Erfc(td / tail_sqrt_2) / tail_two) +
                        tail_half * td * td + log_sqrt_2pi;
    xla::XlaOp ts = xla::Max(t, series_start);
    xla::XlaOp inv_ts = tail_one / ts;
    xla::XlaOp z = inv_ts * inv_ts;
    xla::XlaOp series = xla::ScalarLike(u, 10395.0);
    for (double c : {-945.0, 105.0, -15.0, 3.0, -1.0, 1.0}) {
      series = xla::ScalarLike(u, c) + z * series;
    }
    xla::XlaOp asymptotic = xla::Log(series) - xla::Log(ts);
    return xla::Select(xla::Lt(t, series_start), direct, asymptotic);
  };
  xla::XlaOp upper = xla::Gt(tail_alpha, tail_zero);
  xla::XlaOp lo = xla::Select(upper, tail_alpha, -tail_beta);
  xla::XlaOp hi = xla::Select(upper, tail_beta, -tail_alpha);
  xla::XlaOp rest_lo = log_scaled_upper_tail(lo);
  xla::XlaOp rest_hi = log_scaled_upper_tail(hi);
  xla::XlaOp log_q_diff =
      (rest_hi - rest_lo) - (hi - lo) * (tail_half * hi + tail_half * lo);
  xla::XlaOp delta = xla::Log(u + (tail_one - u) * xla::Exp(log_q_diff));
  xla::XlaOp t_small = xla::Sqrt(lo * lo - tail_two * delta);
  xla::XlaOp t_large = lo * xla::Sqrt(tail_one - tail_two * (delta / lo) / lo);
  xla::XlaOp t = xla::Select(xla::Lt(lo, series_start), t_small, t_large);
  for (int i = 0; i < 3; ++i) {
    // Newton step on log(P(Z > t)) = log(P(Z > lo)) + delta.
    xla::XlaOp rest_t = log_scaled_upper_tail(t);
    xla::XlaOp err = (rest_t - rest_lo - delta) -
                     (t - lo) * (tail_half * t + tail_half * lo);
    xla::XlaOp slope = -xla::Exp(-rest_t);
    t = xla::Clamp(lo, t - err / slope, hi);
  }
  xla::XlaOp tail_x = tail_mu + tail_sigma * xla::Select(upper, t, -t);
  xla::XlaOp tail_clamped = xla::ConvertElementType(
      xla::Clamp(tail_a, tail_x, tail_b), primitive_type);
  xla::XlaOp one_sided = xla::And(xla::Or(upper, xla::Lt(tail_beta, tail_zero)),
                                  xla::IsFinite(tail_x));
  x = xla::Select(one_sided, tail_clamped, x);

  return x;
}

}  // namespace tensorflow
