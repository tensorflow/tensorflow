//===- FakeQuantSupport.h - Support utilities for FakeQuant ops -*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support utilities for interoperating with FakeQuant* based
// QAT (Quantized Aware Training) computations, as implemented by TFLite. Note
// that FakeQuant* operators mix multiple concerns specific to how TFLite
// originally implemented quantization. As such, utilities here enforce
// opinions taken by that codebase (vs providing any amount of genericity).
//
// Specifically, it combines the following concerns, each of which would be
// independent variables in a more generic setup:
//   - numBits and isSigned imply storage data type (uint8, int8, int16)
//   - numBits < 8 is promoted to uint8 or int8
//   - "narrow_range" narrows the lower bound of the storage type's range by
//     1
//   - the specified min/max values are "nudged" so that the result has a zero
//     that can be exactly expressed
//   - min=max=0 implies scale=0 and zero_point=0
//
// With the above assumptions applied, every conforming specified FakeQuant op
// can be represented by a UniformQuantizedType. This scheme is not expected to
// be generalized further in the future and should be considered to be a
// legacy set of rules.
//
// As canonically used in TensorFlow graphs, the presence of a FakeQuant node
// is a hint that the specific math represented here has been simulated at
// training time. As such, it is usually not advised to arbitrarily change
// quantization parameters derived from FakeQuant.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANTOPS_FAKEQUANTSUPPORT_H_
#define MLIR_DIALECT_QUANTOPS_FAKEQUANTSUPPORT_H_

#include "mlir/Dialect/QuantOps/QuantTypes.h"

namespace mlir {
namespace quant {

/// Converts per-layer FakeQuant attributes to the corresponding type.
/// In the event that the parameters cannot be converted, returns a nullptr
/// convertible Type and issues an appropriate error.
/// Note that there are multiple variants of a per-layer FakeQuant op, so
/// this function takes the attributes discretely vs taking a reference to the
/// originating op.
UniformQuantizedType fakeQuantAttrsToType(Location loc, unsigned numBits,
                                          double rmin, double rmax,
                                          bool narrowRange, Type expressedType,
                                          bool isSigned = false);

/// Converts per-channel FakeQuant attributes to the corresponding type.
/// In the event that the parameters cannot be converted, returns a nullptr
/// convertible Type and issues an appropriate error.
UniformQuantizedPerAxisType
fakeQuantAttrsToType(Location loc, unsigned numBits, int32_t quantizedDimension,
                     ArrayRef<double> rmins, ArrayRef<double> rmax,
                     bool narrowRange, Type expressedType,
                     bool isSigned = false);
} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANTOPS_FAKEQUANTSUPPORT_H_
