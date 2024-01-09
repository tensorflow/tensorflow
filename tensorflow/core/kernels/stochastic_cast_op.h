/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_STOCHASTIC_CAST_OP_H_
#define TENSORFLOW_CORE_KERNELS_STOCHASTIC_CAST_OP_H_

#include <limits>
#include <type_traits>

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace internal {

// Base class that dispatches random algorithm, key and counter for
// StochasticCast ops.
class StochasticCastOpBase : public OpKernel {
 public:
  explicit StochasticCastOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

 protected:
  // Subclasses can implement this rounding kernel with assumption that random
  // algorithm, key, counter have been given.
  virtual void RoundOff(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
                        const Tensor& counter, Tensor* output) = 0;
};

}  // namespace internal
}  // namespace tensorflow

namespace Eigen {
namespace internal {

template <typename Scalar, typename IntResultType, typename Generator>
struct StochasticRoundToIntOp {
  static_assert(std::is_integral<IntResultType>::value,
                "Integer type expected");
  typedef tensorflow::random::UniformDistribution<Generator, Scalar>
      Distribution;
  const Scalar max =
      static_cast<Scalar>(std::numeric_limits<IntResultType>::max());
  const Scalar min =
      static_cast<Scalar>(std::numeric_limits<IntResultType>::min());

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC explicit StochasticRoundToIntOp(
      Generator* g)
      : gen(g) {}

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Scalar
  operator()(const Scalar& s) const {
    if (TF_PREDICT_FALSE(Eigen::numext::isnan(s))) {
      return Scalar{0};
    }
    if (s >= max) {
      return max;
    }
    if (s <= min) {
      return min;
    }
    // Already integer, doesn't need to be rounded.
    if (Eigen::numext::floor(s) == s) {
      return s;
    }
    // In order to match comparison-based algorithm on some hardware
    // implementations which rounds abs(operand) up when random <
    // abs(fractional), we deal with positive and negative operands differently.
    // TODO(b/232442915): Revisit RNG multi-threading issue when needed.
    Distribution dist;
    Scalar random = dist(gen)[0];
    if (s < 0) {
      return Eigen::numext::floor(s + random);
    } else {
      return Eigen::numext::floor(s + Scalar{1} - random);
    }
  }

  template <typename Packet>
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet packetOp(const Packet& p) const {
    constexpr size_t kPacketSize =
        Eigen::internal::unpacket_traits<Packet>::size;
    Scalar unpacked_random[kPacketSize];
    Distribution dist;
    auto const sample = dist(gen);
    for (int i = 0; i < kPacketSize; i += Distribution::kResultElementCount) {
      int granularity = std::min(Distribution::kResultElementCount,
                                 static_cast<int>(kPacketSize - i));
      std::copy(&sample[0], &sample[0] + granularity, &unpacked_random[i]);
    }
    Packet random = pload<Packet>(unpacked_random);
    Packet rounded =
        pselect(pcmp_eq(pfloor(p), p), p,
                pselect(pcmp_lt(p, pzero(p)), pfloor(padd(p, random)),
                        pfloor(padd(p, psub(pset1<Packet>(1), random)))));
    // Handles out of range inputs.
    Packet result =
        pselect(pcmp_le(pset1<Packet>(max), p), pset1<Packet>(max), rounded);
    result =
        pselect(pcmp_le(p, pset1<Packet>(min)), pset1<Packet>(min), result);
    // Handles NaN input.
    return pselect(pcmp_eq(p, p), result, pset1<Packet>(0));
  }
  Generator* gen;
};

template <typename Scalar, typename IntResultType, typename Generator>
struct functor_traits<
    StochasticRoundToIntOp<Scalar, IntResultType, Generator>> {
  enum {
    Cost = 3 * NumTraits<Scalar>::AddCost,
    PacketAccess =
        packet_traits<Scalar>::HasCmp && packet_traits<Scalar>::HasFloor,
  };
};

// TODO(b/232442915): Add support for rounding floats to lower precision floats.

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_STOCHASTIC_CAST_OP_H_
