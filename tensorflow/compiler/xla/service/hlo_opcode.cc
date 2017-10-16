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

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

string HloOpcodeString(HloOpcode opcode) {
  // Note: Do not use ':' in opcode strings. It is used as a special character
  // in these places:
  // - In extended opcode strings (HloInstruction::ExtendedOpcodeString()), to
  //   separate the opcode from the fusion kind
  // - In fully qualified names (HloInstruction::FullyQualifiedName()), to
  //   separate the qualifiers (name of the computation and potentially the
  //   fusion instruction) from the name
  switch (opcode) {
    case HloOpcode::kAbs:
      return "abs";
    case HloOpcode::kAdd:
      return "add";
    case HloOpcode::kBatchNormTraining:
      return "batch-norm-training";
    case HloOpcode::kBatchNormInference:
      return "batch-norm-inference";
    case HloOpcode::kBatchNormGrad:
      return "batch-norm-grad";
    case HloOpcode::kBitcast:
      return "bitcast";
    case HloOpcode::kBroadcast:
      return "broadcast";
    case HloOpcode::kCall:
      return "call";
    case HloOpcode::kClamp:
      return "clamp";
    case HloOpcode::kConcatenate:
      return "concatenate";
    case HloOpcode::kConstant:
      return "constant";
    case HloOpcode::kConvert:
      return "convert";
    case HloOpcode::kConvolution:
      return "convolution";
    case HloOpcode::kCos:
      return "cosine";
    case HloOpcode::kCrossReplicaSum:
      return "cross-replica-sum";
    case HloOpcode::kCustomCall:
      return "custom-call";
    case HloOpcode::kCopy:
      return "copy";
    case HloOpcode::kDivide:
      return "divide";
    case HloOpcode::kDot:
      return "dot";
    case HloOpcode::kDynamicSlice:
      return "dynamic-slice";
    case HloOpcode::kDynamicUpdateSlice:
      return "dynamic-update-slice";
    case HloOpcode::kEq:
      return "equal-to";
    case HloOpcode::kExp:
      return "exponential";
    case HloOpcode::kFloor:
      return "floor";
    case HloOpcode::kCeil:
      return "ceil";
    case HloOpcode::kFusion:
      return "fusion";
    case HloOpcode::kGe:
      return "greater-than-or-equal-to";
    case HloOpcode::kGetTupleElement:
      return "get-tuple-element";
    case HloOpcode::kGt:
      return "greater-than";
    case HloOpcode::kIndex:
      return "index";
    case HloOpcode::kInfeed:
      return "infeed";
    case HloOpcode::kIsFinite:
      return "is-finite";
    case HloOpcode::kLe:
      return "less-than-or-equal-to";
    case HloOpcode::kLog:
      return "log";
    case HloOpcode::kAnd:
      return "and";
    case HloOpcode::kOr:
      return "or";
    case HloOpcode::kNot:
      return "not";
    case HloOpcode::kLt:
      return "less-than";
    case HloOpcode::kMap:
      return "map";
    case HloOpcode::kMaximum:
      return "maximum";
    case HloOpcode::kMinimum:
      return "minimum";
    case HloOpcode::kMultiply:
      return "multiply";
    case HloOpcode::kNe:
      return "not-equal-to";
    case HloOpcode::kNegate:
      return "negate";
    case HloOpcode::kOutfeed:
      return "outfeed";
    case HloOpcode::kPad:
      return "pad";
    case HloOpcode::kParameter:
      return "parameter";
    case HloOpcode::kPower:
      return "power";
    case HloOpcode::kRecv:
      return "recv";
    case HloOpcode::kReduce:
      return "reduce";
    case HloOpcode::kReducePrecision:
      return "reduce-precision";
    case HloOpcode::kReduceWindow:
      return "reduce-window";
    case HloOpcode::kRemainder:
      return "remainder";
    case HloOpcode::kReshape:
      return "reshape";
    case HloOpcode::kReverse:
      return "reverse";
    case HloOpcode::kRng:
      return "rng";
    case HloOpcode::kRoundNearestAfz:
      return "round-nearest-afz";
    case HloOpcode::kSelectAndScatter:
      return "select-and-scatter";
    case HloOpcode::kSelect:
      return "select";
    case HloOpcode::kSend:
      return "send";
    case HloOpcode::kShiftLeft:
      return "shift-left";
    case HloOpcode::kShiftRightArithmetic:
      return "shift-right-arithmetic";
    case HloOpcode::kShiftRightLogical:
      return "shift-right-logical";
    case HloOpcode::kSign:
      return "sign";
    case HloOpcode::kSin:
      return "sine";
    case HloOpcode::kSlice:
      return "slice";
    case HloOpcode::kSort:
      return "sort";
    case HloOpcode::kSubtract:
      return "subtract";
    case HloOpcode::kTanh:
      return "tanh";
    case HloOpcode::kTrace:
      return "trace";
    case HloOpcode::kTranspose:
      return "transpose";
    case HloOpcode::kTuple:
      return "tuple";
    case HloOpcode::kUpdate:
      return "update";
    case HloOpcode::kWhile:
      return "while";
  }
}

StatusOr<HloOpcode> StringToHloOpcode(const string& opcode_name) {
  static auto* opcode_map = new tensorflow::gtl::FlatMap<string, HloOpcode>(
      {{"abs", HloOpcode::kAbs},
       {"add", HloOpcode::kAdd},
       {"batch-norm-training", HloOpcode::kBatchNormTraining},
       {"batch-norm-inference", HloOpcode::kBatchNormInference},
       {"batch-norm-grad", HloOpcode::kBatchNormGrad},
       {"bitcast", HloOpcode::kBitcast},
       {"broadcast", HloOpcode::kBroadcast},
       {"call", HloOpcode::kCall},
       {"clamp", HloOpcode::kClamp},
       {"concatenate", HloOpcode::kConcatenate},
       {"constant", HloOpcode::kConstant},
       {"convert", HloOpcode::kConvert},
       {"convolution", HloOpcode::kConvolution},
       {"cosine", HloOpcode::kCos},
       {"cross-replica-sum", HloOpcode::kCrossReplicaSum},
       {"custom-call", HloOpcode::kCustomCall},
       {"copy", HloOpcode::kCopy},
       {"divide", HloOpcode::kDivide},
       {"dot", HloOpcode::kDot},
       {"dynamic-slice", HloOpcode::kDynamicSlice},
       {"dynamic-update-slice", HloOpcode::kDynamicUpdateSlice},
       {"equal-to", HloOpcode::kEq},
       {"exponential", HloOpcode::kExp},
       {"floor", HloOpcode::kFloor},
       {"ceil", HloOpcode::kCeil},
       {"fusion", HloOpcode::kFusion},
       {"greater-than-or-equal-to", HloOpcode::kGe},
       {"get-tuple-element", HloOpcode::kGetTupleElement},
       {"greater-than", HloOpcode::kGt},
       {"index", HloOpcode::kIndex},
       {"infeed", HloOpcode::kInfeed},
       {"is-finite", HloOpcode::kIsFinite},
       {"less-than-or-equal-to", HloOpcode::kLe},
       {"log", HloOpcode::kLog},
       {"and", HloOpcode::kAnd},
       {"or", HloOpcode::kOr},
       {"not", HloOpcode::kNot},
       {"less-than", HloOpcode::kLt},
       {"map", HloOpcode::kMap},
       {"maximum", HloOpcode::kMaximum},
       {"minimum", HloOpcode::kMinimum},
       {"multiply", HloOpcode::kMultiply},
       {"not-equal-to", HloOpcode::kNe},
       {"negate", HloOpcode::kNegate},
       {"outfeed", HloOpcode::kOutfeed},
       {"pad", HloOpcode::kPad},
       {"parameter", HloOpcode::kParameter},
       {"power", HloOpcode::kPower},
       {"recv", HloOpcode::kRecv},
       {"reduce", HloOpcode::kReduce},
       {"reduce-precision", HloOpcode::kReducePrecision},
       {"reduce-window", HloOpcode::kReduceWindow},
       {"remainder", HloOpcode::kRemainder},
       {"reshape", HloOpcode::kReshape},
       {"reverse", HloOpcode::kReverse},
       {"rng", HloOpcode::kRng},
       {"round-nearest-afz", HloOpcode::kRoundNearestAfz},
       {"select-and-scatter", HloOpcode::kSelectAndScatter},
       {"select", HloOpcode::kSelect},
       {"send", HloOpcode::kSend},
       {"shift-left", HloOpcode::kShiftLeft},
       {"shift-right-arithmetic", HloOpcode::kShiftRightArithmetic},
       {"shift-right-logical", HloOpcode::kShiftRightLogical},
       {"sign", HloOpcode::kSign},
       {"sine", HloOpcode::kSin},
       {"slice", HloOpcode::kSlice},
       {"sort", HloOpcode::kSort},
       {"subtract", HloOpcode::kSubtract},
       {"tanh", HloOpcode::kTanh},
       {"trace", HloOpcode::kTrace},
       {"transpose", HloOpcode::kTranspose},
       {"tuple", HloOpcode::kTuple},
       {"update", HloOpcode::kUpdate},
       {"while", HloOpcode::kWhile}});
  auto it = opcode_map->find(opcode_name);
  if (it == opcode_map->end()) {
    return InvalidArgument("Unknown opcode: %s", opcode_name.c_str());
  }
  return it->second;
}

bool HloOpcodeIsComparison(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kEq:
    case HloOpcode::kNe:
      return true;
    default:
      return false;
  }
}

bool HloOpcodeIsVariadic(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kCall:
    case HloOpcode::kConcatenate:
    case HloOpcode::kFusion:
    case HloOpcode::kMap:
    case HloOpcode::kTuple:
      return true;
    default:
      return false;
  }
}

}  // namespace xla
