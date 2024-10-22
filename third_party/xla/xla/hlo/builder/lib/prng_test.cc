/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/prng.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class PrngTest : public ClientLibraryTestBase {
 public:
  template <PrimitiveType value_type, PrimitiveType bit_type,
            typename ValueT = typename primitive_util::PrimitiveTypeToNative<
                value_type>::type,
            typename BitT =
                typename primitive_util::PrimitiveTypeToNative<bit_type>::type>
  void TestConvertRandomBitsToUniformFloatingPoint(uint32_t bits, float minval,
                                                   float maxval) {
    XlaBuilder builder("convert_random_bits_to_uniform_floating_point");
    XlaOp bits_op = ConstantR0<BitT>(&builder, static_cast<BitT>(bits));
    XlaOp minval_op = ConstantR0<ValueT>(&builder, static_cast<ValueT>(minval));
    XlaOp maxval_op = ConstantR0<ValueT>(&builder, static_cast<ValueT>(maxval));
    XlaOp seed = ConstantR0<uint64_t>(&builder, 42);
    XlaOp initial_state = Zero(&builder, PrimitiveType::U64);
    BitGeneratorTy bit_generator = [](XlaOp key, XlaOp state,
                                      const Shape& shape) {
      state = ConcatScalars(key.builder(), {key, state});
      XlaOp result =
          RngBitGenerator(RandomAlgorithm::RNG_DEFAULT, state, shape);
      return RngOutput{/*value=*/GetTupleElement(result, 1),
                       /*state=*/GetTupleElement(result, 0)};
    };
    // This controls the bit width in ConvertRandomBitsToUniformFloatingPoint.
    const Shape rng_shape = builder.GetShape(bits_op).value();
    EXPECT_EQ(rng_shape.element_type(), bit_type);

    // ConvertRandomBitsToUniformFloatingPoint is not declared in the header
    // file and tested through calling UniformFloatingPointDistribution.
    RngOutput rng_output = UniformFloatingPointDistribution(
        seed, initial_state, bit_generator, minval_op, maxval_op, rng_shape);
    if (rng_output.value.valid()) {
      XlaOp result = rng_output.value;
      EXPECT_EQ(builder.GetShape(result).value().element_type(), value_type);

      // Check that the result is in the range [minval, maxval)
      XlaOp result_ge_min = Ge(result, minval_op);
      XlaOp result_lt_max = Lt(result, maxval_op);
      And(result_ge_min, result_lt_max);
      // TODO: b/302787872 - rng-bit-generator unit testing is not supported.
      ComputeAndCompareR0<bool>(&builder, /*expected=*/true, {});
    } else {
      // Testing with invalid arguments should return InvalidArgument error.
      EXPECT_EQ(builder.first_error().code(),
                absl::StatusCode::kInvalidArgument);
    }
  }
};

XLA_TEST_F(PrngTest, RandomBitsToUniformFloatingPointInvalidArguments) {
  // Existing prng test targets do not test invalid arguments cases, where
  // the number of bits are smaller than the value type's mantissa bits.
  TestConvertRandomBitsToUniformFloatingPoint<PrimitiveType::F32,
                                              PrimitiveType::U16>(0x1234, 0.0f,
                                                                  1.0f);
  TestConvertRandomBitsToUniformFloatingPoint<PrimitiveType::F16,
                                              PrimitiveType::U8>(0x12, 0.0f,
                                                                 1.0f);
}

}  // namespace
}  // namespace xla
