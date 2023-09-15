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
#include "tensorflow/core/kernels/stochastic_cast_op.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/lib/random/philox_random.h"

namespace Eigen {
namespace internal {

// Comparison based stochastic rounding for cross verification.
template <typename Scalar, typename IntResultType, typename Generator>
struct StochasticCastVerifier {
  static_assert(std::is_integral<IntResultType>::value,
                "Integer type expected");
  const Scalar max =
      static_cast<Scalar>(std::numeric_limits<IntResultType>::max());
  const Scalar min =
      static_cast<Scalar>(std::numeric_limits<IntResultType>::min());
  typedef tensorflow::random::UniformDistribution<Generator, Scalar>
      Distribution;
  using T = typename Eigen::internal::make_integer<Scalar>::type;

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC explicit StochasticCastVerifier(
      Generator* g)
      : gen(g) {}

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Scalar
  operator()(const Scalar& s) const {
    if (Eigen::numext::isnan(s)) {
      return Scalar{0};
    }
    if (s >= max) {
      return max;
    }
    if (s <= min) {
      return min;
    }
    Scalar abs_s = Eigen::numext::abs(s);
    // Gets the integral piece of the floating point input.
    Scalar truncated = Eigen::numext::floor(abs_s);
    Scalar fractional = abs_s - truncated;
    if (fractional == Scalar{0}) {
      // No rounding necessary.
      return s < Scalar{0} ? -truncated : truncated;
    }
    Distribution dist;
    Scalar random = dist(gen)[0];
    // Rounds the integer output up if the fractional pieces is larger than
    // the input random number.
    if (random < fractional) {
      truncated++;
    }
    return s < Scalar{0} ? -truncated : truncated;
  }

  template <typename Packet>
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet packetOp(const Packet& p) const {
    Packet abs_p = pabs<Packet>(p);
    Packet truncated = pfloor(abs_p);
    Packet fractional = psub(abs_p, truncated);
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
    truncated = pselect(pcmp_lt(random, fractional),
                        padd(truncated, pset1<Packet>(1)), truncated);
    truncated = pselect(pcmp_lt(p, pzero(p)), pnegate(truncated), truncated);
    // Handles out of range inputs.
    Packet result =
        pselect(pcmp_le(pset1<Packet>(max), p), pset1<Packet>(max), truncated);
    result =
        pselect(pcmp_le(p, pset1<Packet>(min)), pset1<Packet>(min), result);
    // Handles NaN input.
    return pselect(pcmp_eq(p, p), result, pset1<Packet>(1));
  }
  Generator* gen;
};

template <typename Scalar, typename IntResultType, typename Generator>
struct functor_traits<
    StochasticCastVerifier<Scalar, IntResultType, Generator>> {
  enum {
    Cost = 3 * NumTraits<Scalar>::AddCost,
    PacketAccess =
        packet_traits<Scalar>::HasCmp && packet_traits<Scalar>::HasFloor,
  };
};

}  // namespace internal
}  // namespace Eigen

namespace tensorflow {

using Eigen::half;
using tensorflow::random::PhiloxRandom;

class StochasticCastOpToIntTest : public OpsTestBase {
 public:
  static const int kAlgorithm = 1;
  static const uint64_t kRngCounter = 30;
  static const uint64_t kRngKey = 20;

 protected:
  template <typename InType, typename OutType>
  void CastResultProbabilityTestHelper(DataType In_Type, DataType Out_Type) {
    // Choose a dim large enough without OOM issue.
    static const uint64_t kDim = 1 << 23;
    TF_ASSERT_OK(
        NodeDefBuilder("stochastic_cast_to_int_op", "StochasticCastToInt")
            .Input(FakeInput(In_Type))
            .Input(FakeInput(DT_UINT64))
            .Input(FakeInput(DT_UINT64))
            .Input(FakeInput(DT_INT32))
            .Attr("Tin", In_Type)
            .Attr("Tout", Out_Type)
            .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());

    const InType value = InType(0.625);
    AddInput<InType>(TensorShape({kDim, 1}), [&value](int i) { return value; });
    AddInput<uint64>(TensorShape({1}), [](int i) { return kRngKey; });
    AddInput<uint64>(TensorShape({1}), [](int i) { return kRngCounter; });
    AddInput<int>(TensorShape({}), [](int i) { return kAlgorithm; });

    TF_ASSERT_OK(RunOpKernel());

    OutType* result = GetOutput(0)->flat<OutType>().data();
    int floor_count = 0, ceil_count = 0;
    for (int i = 0; i < kDim; ++i) {
      if (*(result + i) == static_cast<OutType>(std::floor(value))) {
        floor_count++;
      } else if (*(result + i) == static_cast<OutType>(std::ceil(value))) {
        ceil_count++;
      }
    }
    float expected_probability_ratio = (Eigen::numext::ceil(value) - value) /
                                       (value - Eigen::numext::floor(value));
    float real_probability_ratio =
        static_cast<float>(floor_count) / static_cast<float>(ceil_count);
    double err_allowance = 0.03;
    EXPECT_TRUE(test::internal_test::IsClose(real_probability_ratio,
                                             expected_probability_ratio,
                                             err_allowance, err_allowance));
  }

  template <typename InType, typename OutType>
  void ExhasutiveTestHelper(DataType in_type, DataType out_type) {
    const uint64_t digits = Eigen::NumTraits<InType>::digits();
    const uint64_t dim = digits > 23 ? (1 << 23) : (1L << digits);
    const uint64_t total = uint64_t{1} << Eigen::NumTraits<InType>::digits();
    const uint64_t granularity = digits > 23 ? (total / dim) : 1;

    TF_ASSERT_OK(
        NodeDefBuilder("stochastic_cast_to_int_op", "StochasticCastToInt")
            .Input(FakeInput(in_type))
            .Input(FakeInput(DT_UINT64))
            .Input(FakeInput(DT_UINT64))
            .Input(FakeInput(DT_INT32))
            .Attr("Tin", in_type)
            .Attr("Tout", out_type)
            .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());

    TensorShape shape({static_cast<int64_t>(dim)});
    Tensor* input = AddInput(in_type, shape);
    AddInput<uint64>(TensorShape({1}), [](int i) { return kRngKey; });
    AddInput<uint64>(TensorShape({1}), [](int i) { return kRngCounter; });
    AddInput<int>(TensorShape({}), [](int i) { return kAlgorithm; });

    auto in = input->flat<InType>();
    using T = typename Eigen::internal::make_integer<InType>::type;
    for (int i = 0; i < dim; ++i) {
      in(i) = Eigen::numext::bit_cast<InType>(static_cast<T>(i * granularity));
    }

    Tensor expected(out_type, input->shape());
    PhiloxRandom gen(kRngCounter, kRngKey);
    expected.flat<OutType>() =
        input->flat<InType>()
            .unaryExpr(
                Eigen::internal::StochasticCastVerifier<InType, OutType,
                                                        PhiloxRandom>(&gen))
            .template cast<OutType>();

    TF_ASSERT_OK(RunOpKernel());

    tensorflow::test::ExpectEqual(expected, *GetOutput(0));
  }
};

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestHalfCastToInt32) {
  ExhasutiveTestHelper<half, int>(DT_HALF, DT_INT32);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestHalfCastToInt16) {
  ExhasutiveTestHelper<half, int16>(DT_HALF, DT_INT16);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestHalfCastToInt8) {
  ExhasutiveTestHelper<half, int8>(DT_HALF, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestBf16CastToInt32) {
  ExhasutiveTestHelper<bfloat16, int>(DT_BFLOAT16, DT_INT32);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestBf16CastToInt16) {
  ExhasutiveTestHelper<bfloat16, int16>(DT_BFLOAT16, DT_INT16);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestBf16CastToInt8) {
  ExhasutiveTestHelper<bfloat16, int8>(DT_BFLOAT16, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestFloatCastToInt32) {
  ExhasutiveTestHelper<float, int>(DT_FLOAT, DT_INT32);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestFloatCastToInt16) {
  ExhasutiveTestHelper<float, int16>(DT_FLOAT, DT_INT16);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestFloatCastToInt8) {
  ExhasutiveTestHelper<float, int8>(DT_FLOAT, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestDoubleCastToInt32) {
  ExhasutiveTestHelper<double, int>(DT_DOUBLE, DT_INT32);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestDoubleCastToInt16) {
  ExhasutiveTestHelper<double, int16>(DT_DOUBLE, DT_INT16);
}

TEST_F(StochasticCastOpToIntTest, ExhaustiveTestDoubleCastToInt8) {
  ExhasutiveTestHelper<double, int8>(DT_DOUBLE, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, CastProbabilityTestHalfToInt8) {
  CastResultProbabilityTestHelper<half, int8>(DT_HALF, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, CastProbabilityTestBf16ToInt8) {
  CastResultProbabilityTestHelper<bfloat16, int8>(DT_BFLOAT16, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, CastProbabilityTestFloatToInt8) {
  CastResultProbabilityTestHelper<float, int8>(DT_FLOAT, DT_INT8);
}

TEST_F(StochasticCastOpToIntTest, CastProbabilityTestDoubleToInt8) {
  CastResultProbabilityTestHelper<double, int8>(DT_DOUBLE, DT_INT8);
}

}  // namespace tensorflow
