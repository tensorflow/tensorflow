/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_

#include <iosfwd>
#include <string>
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// High-level optimizer instruction opcodes -- these are linear-algebra level
// opcodes. They are a flattened form of the UnaryOp, BinaryOp, ... opcodes
// present in the XLA service protobuf.
//
// See the XLA documentation for the semantics of each opcode.
//
// Each entry has the format:
// (enum_name, opcode_name)
// or
// (enum_name, opcode_name, p1 | p2 | ...)
//
// with p1, p2, ... are members of HloOpcodeProperty. They are combined
// using bitwise-or.
//
// Note: Do not use ':' in opcode names. It is used as a special character
// in these places:
// - In extended opcode strings (HloInstruction::ExtendedOpcodeString()), to
//   separate the opcode from the fusion kind
// - In fully qualified names (HloInstruction::FullyQualifiedName()), to
//   separate the qualifiers (name of the computation and potentially the
//   fusion instruction) from the name
#define HLO_OPCODE_LIST(V)                                   \
  V(kAbs, "abs")                                             \
  V(kAdd, "add")                                             \
  V(kAtan2, "atan2")                                         \
  V(kBatchNormGrad, "batch-norm-grad")                       \
  V(kBatchNormInference, "batch-norm-inference")             \
  V(kBatchNormTraining, "batch-norm-training")               \
  V(kBitcast, "bitcast")                                     \
  V(kBroadcast, "broadcast")                                 \
  V(kCall, "call", kHloOpcodeIsVariadic)                     \
  V(kCeil, "ceil")                                           \
  V(kClamp, "clamp")                                         \
  V(kComplex, "complex")                                     \
  V(kConcatenate, "concatenate", kHloOpcodeIsVariadic)       \
  V(kConditional, "conditional")                             \
  V(kConstant, "constant")                                   \
  V(kConvert, "convert")                                     \
  V(kConvolution, "convolution")                             \
  V(kCopy, "copy")                                           \
  V(kCos, "cosine")                                          \
  V(kCrossReplicaSum, "cross-replica-sum")                   \
  V(kCustomCall, "custom-call")                              \
  V(kDivide, "divide")                                       \
  V(kDot, "dot")                                             \
  V(kDynamicSlice, "dynamic-slice")                          \
  V(kDynamicUpdateSlice, "dynamic-update-slice")             \
  V(kEq, "equal-to", kHloOpcodeIsComparison)                 \
  V(kExp, "exponential")                                     \
  V(kFloor, "floor")                                         \
  V(kFusion, "fusion", kHloOpcodeIsVariadic)                 \
  V(kGe, "greater-than-or-equal-to", kHloOpcodeIsComparison) \
  V(kGetTupleElement, "get-tuple-element")                   \
  V(kGt, "greater-than", kHloOpcodeIsComparison)             \
  V(kImag, "imag")                                           \
  V(kInfeed, "infeed")                                       \
  V(kIsFinite, "is-finite")                                  \
  V(kLe, "less-than-or-equal-to", kHloOpcodeIsComparison)    \
  V(kLog, "log")                                             \
  V(kAnd, "and")                                             \
  V(kNot, "not")                                             \
  V(kOr, "or")                                               \
  V(kLt, "less-than", kHloOpcodeIsComparison)                \
  V(kMap, "map", kHloOpcodeIsVariadic)                       \
  V(kMaximum, "maximum")                                     \
  V(kMinimum, "minimum")                                     \
  V(kMultiply, "multiply")                                   \
  V(kNe, "not-equal-to", kHloOpcodeIsComparison)             \
  V(kNegate, "negate")                                       \
  V(kOutfeed, "outfeed")                                     \
  V(kPad, "pad")                                             \
  V(kParameter, "parameter")                                 \
  V(kPower, "power")                                         \
  V(kReal, "real")                                           \
  V(kRecv, "recv")                                           \
  V(kRecvDone, "recv-done")                                  \
  V(kReduce, "reduce")                                       \
  V(kReducePrecision, "reduce-precision")                    \
  V(kReduceWindow, "reduce-window")                          \
  V(kRemainder, "remainder")                                 \
  V(kReshape, "reshape")                                     \
  V(kReverse, "reverse")                                     \
  V(kRng, "rng")                                             \
  V(kRoundNearestAfz, "round-nearest-afz")                   \
  V(kSelect, "select")                                       \
  V(kSelectAndScatter, "select-and-scatter")                 \
  V(kSend, "send")                                           \
  V(kSendDone, "send-done")                                  \
  V(kShiftLeft, "shift-left")                                \
  V(kShiftRightArithmetic, "shift-right-arithmetic")         \
  V(kShiftRightLogical, "shift-right-logical")               \
  V(kSign, "sign")                                           \
  V(kSin, "sine")                                            \
  V(kSlice, "slice")                                         \
  V(kSort, "sort")                                           \
  V(kSubtract, "subtract")                                   \
  V(kTanh, "tanh")                                           \
  V(kTrace, "trace")                                         \
  V(kTranspose, "transpose")                                 \
  V(kTuple, "tuple", kHloOpcodeIsVariadic)                   \
  V(kWhile, "while")

enum class HloOpcode {
#define DECLARE_ENUM(enum_name, opcode_name, ...) enum_name,
  HLO_OPCODE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
};

// List of properties associated with opcodes.
// Properties are defined as increasing powers of two, so that we can use
// bitwise-or to combine properties, and bitwise-and to test for them.
enum HloOpcodeProperty {
  kHloOpcodeIsComparison = 1 << 0,
  kHloOpcodeIsVariadic = 1 << 1,
};

// Returns a string representation of the opcode.
string HloOpcodeString(HloOpcode opcode);

// Returns a string representation of the opcode.
StatusOr<HloOpcode> StringToHloOpcode(const string& opcode_name);

inline std::ostream& operator<<(std::ostream& os, HloOpcode opcode) {
  return os << HloOpcodeString(opcode);
}

// Returns true iff the given opcode is a comparison operation.
bool HloOpcodeIsComparison(HloOpcode opcode);

// Returns true iff the given opcode has variadic operands.
bool HloOpcodeIsVariadic(HloOpcode opcode);

// Returns the number of HloOpcode values.
inline const uint32_t HloOpcodeCount() {
#define HLO_COUNT_ONE(...) +1
#define HLO_XLIST_LENGTH(list) list(HLO_COUNT_ONE)
  return HLO_XLIST_LENGTH(HLO_OPCODE_LIST);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_
