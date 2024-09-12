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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_FAKEQUANTSUPPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_FAKEQUANTSUPPORT_H_

#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quantfork {

/// Converts per-layer FakeQuant attributes to the corresponding type.
/// In the event that the parameters cannot be converted, returns a nullptr
/// convertible Type and issues an appropriate error.
/// Note that there are multiple variants of a per-layer FakeQuant op, so
/// this function takes the attributes discretely vs taking a reference to the
/// originating op.
quant::UniformQuantizedType fakeQuantAttrsToType(Location loc, unsigned numBits,
                                                 double rmin, double rmax,
                                                 bool narrowRange,
                                                 Type expressedType,
                                                 bool isSigned = false);

/// Converts per-channel FakeQuant attributes to the corresponding type.
/// In the event that the parameters cannot be converted, returns a nullptr
/// convertible Type and issues an appropriate error.
quant::UniformQuantizedPerAxisType fakeQuantAttrsToType(
    Location loc, unsigned numBits, int32_t quantizedDimension,
    ArrayRef<double> rmins, ArrayRef<double> rmax, bool narrowRange,
    Type expressedType, bool isSigned = false);
}  // namespace quantfork
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_FAKEQUANTSUPPORT_H_
