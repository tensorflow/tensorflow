/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_MLIR_HLO_MHLO_TRANSFORMS_TRANSFORMATION_HELPERS_H_
#define XLA_MLIR_HLO_MHLO_TRANSFORMS_TRANSFORMATION_HELPERS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::mhlo {

// Returns the input value with a reduced precision as specified by the target
// exponent and mantissa bits. This function will preserve the input shape on
// the output - i.e. it works with both scalars and tensors.
//
// The templated bitcast type allows this function to work with different kinds
// of bitcats, e.g. `arith.bitcast` or `triton.bitcast`.
template <typename BitCastOp>
Value reducePrecision(Location loc, Value input, int destExponentBits,
                      int destMantissaBits, OpBuilder* builder) {
  using llvm::APInt;
  mlir::ImplicitLocOpBuilder b(loc, *builder);

  // Integer and float types for casting and constant generation.
  auto floatType = mlir::cast<FloatType>(getElementTypeOrSelf(input.getType()));
  int64_t nbits = floatType.getWidth();
  auto intScalarType = mlir::IntegerType::get(loc.getContext(), nbits);

  Type intType = intScalarType;
  std::optional<std::vector<int64_t>> shape;
  if (auto shapedType = dyn_cast<ShapedType>(input.getType())) {
    shape = shapedType.getShape().vec();
    intType = shapedType.clone(intScalarType);
  }

  Value xAsInt = b.create<BitCastOp>(intType, input);

  // SignificandWidth includes the implicit extra bit.
  auto srcMantissaBits = floatType.getFPMantissaWidth() - 1;
  int srcExponentBits = nbits - 1 - srcMantissaBits;

  // Clear the sign bit, it does not participate in rounding and we will restore
  // it later.
  APInt signBitMask(nbits, 1);
  signBitMask <<= nbits - 1;

  APInt expBitsMask(nbits, 1);
  expBitsMask = ((expBitsMask << srcExponentBits) - 1) << srcMantissaBits;

  auto createConstant = [&](const APInt& v) {
    return createScalarOrSplatConstant(b, loc, intType, v);
  };

  Value xAbsBits =
      b.create<arith::AndIOp>(xAsInt, createConstant(~signBitMask));
  Value xIsNan = b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, xAbsBits,
                                         createConstant(expBitsMask));

  if (destMantissaBits < static_cast<int>(srcMantissaBits)) {
    // Last remaining mantissa bit.
    APInt lastMantissaBitMask(nbits, 1);
    lastMantissaBitMask <<= srcMantissaBits - destMantissaBits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt baseRoundingBias = lastMantissaBitMask.lshr(1) - 1;

    Value mantissaDiff =
        createConstant(APInt(nbits, srcMantissaBits - destMantissaBits));

    Value highestMantissaMaskVal = createConstant(lastMantissaBitMask);
    Value baseRoundingBiasVal = createConstant(baseRoundingBias);
    Value xLastMantissaBit = b.create<arith::ShRUIOp>(
        b.create<arith::AndIOp>(xAsInt, highestMantissaMaskVal), mantissaDiff);
    Value xRoundingBias =
        b.create<arith::AddIOp>(xLastMantissaBit, baseRoundingBiasVal);

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    APInt truncationMask = ~(lastMantissaBitMask - 1);
    Value xRounded = b.create<arith::AddIOp>(xAsInt, xRoundingBias);
    xAsInt = b.create<arith::AndIOp>(xRounded, createConstant(truncationMask));
  }

  if (destExponentBits < srcExponentBits) {
    // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the most-
    // significant bit -- is equal to 1.0f for all exponent sizes.  Adding
    // 2^(n-1)-1 to this gives us the highest non-infinite exponent for a bit-
    // size of n, and subtracting 2^(n-1)-1 from this gives us the lowest'
    // exponent (corresponding to 0.0f).
    //
    // Thus, the f32 exponent corresponding to the highest non-infinite
    // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
    // exponent corresponding to the lowest exponent for a bit size of n is
    // (2^7-1) - 2^(n-1)-1.
    //
    // Note that we have already checked that exponents_bits >= 1.
    APInt exponentBias(nbits, 1);
    exponentBias = (exponentBias << (srcExponentBits - 1)) - 1;

    APInt reducedExponentBias(nbits, 1);
    reducedExponentBias = (reducedExponentBias << (destExponentBits - 1)) - 1;

    APInt reducedMaxExponent = exponentBias + reducedExponentBias;
    APInt reducedMinExponent = exponentBias - reducedExponentBias;

    // Do we overflow or underflow?
    Value xExponent =
        b.create<arith::AndIOp>(xAsInt, createConstant(expBitsMask));
    Value xOverflows = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::ugt, xExponent,
        createConstant(reducedMaxExponent << srcMantissaBits));
    Value xUnderflows = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::ule, xExponent,
        createConstant(reducedMinExponent << srcMantissaBits));

    // Compute appropriately-signed values of zero and infinity.
    Value xSignedZero =
        b.create<arith::AndIOp>(xAsInt, createConstant(signBitMask));
    Value xSignedInf =
        b.create<arith::OrIOp>(xSignedZero, createConstant(expBitsMask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    xAsInt = b.create<arith::SelectOp>(xOverflows, xSignedInf, xAsInt);
    xAsInt = b.create<arith::SelectOp>(xUnderflows, xSignedZero, xAsInt);
  }

  Value result = b.create<BitCastOp>(input.getType(), xAsInt);
  return b.create<arith::SelectOp>(xIsNan, input, result);
}
}  // namespace mlir::mhlo

#endif  // XLA_MLIR_HLO_MHLO_TRANSFORMS_TRANSFORMATION_HELPERS_H_
