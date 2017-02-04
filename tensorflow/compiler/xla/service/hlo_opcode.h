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
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// High-level optimizer instruction opcodes -- these are linear-algebra level
// opcodes. They are a flattened form of the UnaryOp, BinaryOp, ... opcodes
// present in the XLA service protobuf.
//
// See the XLA documentation for the semantics of each opcode.
enum class HloOpcode {
  kAbs,
  kAdd,
  kBitcast,
  kBroadcast,
  kCall,
  kCeil,
  kClamp,
  kConcatenate,
  kConstant,
  kConvert,
  kConvolution,
  kCopy,
  kCrossReplicaSum,
  kCustomCall,
  kDivide,
  kDot,
  kDynamicSlice,
  kDynamicUpdateSlice,
  kEq,
  kExp,
  kFloor,
  kFusion,
  kGe,
  kGetTupleElement,
  kGt,
  kIndex,
  kInfeed,
  kLe,
  kLog,
  kLogicalAnd,
  kLogicalNot,
  kLogicalOr,
  kLt,
  kMap,
  kMaximum,
  kMinimum,
  kMultiply,
  kNe,
  kNegate,
  kOutfeed,
  kPad,
  kParameter,
  kPower,
  kRecv,
  kReduce,
  kReduceWindow,
  kRemainder,
  kReshape,
  kReverse,
  kRng,
  kSelect,
  kSelectAndScatter,
  kSend,
  kSign,
  kSlice,
  kSort,
  kSubtract,
  kTanh,
  kTrace,
  kTranspose,
  kTuple,
  kUpdate,
  kWhile,
};

// Returns a string representation of the opcode.
string HloOpcodeString(HloOpcode opcode);

inline std::ostream& operator<<(std::ostream& os, HloOpcode opcode) {
  return os << HloOpcodeString(opcode);
}

// Returns true iff the given opcode is a comparison operation.
bool HloOpcodeIsComparison(HloOpcode opcode);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_
