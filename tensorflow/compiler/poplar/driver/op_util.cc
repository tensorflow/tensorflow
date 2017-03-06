#include <algorithm>

#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace poplarplugin {

#define POPLAR_OPCODE(O, N) case HloOpcode::O: return std::string(N)
#define UNUSED_OPCODE(O) case HloOpcode::O: break;

// NOTE 'Unused' Opcodes are implemented as a fixed function in the visitor
port::StatusOr<std::string>
LookupPoplarVertexName(HloOpcode opcode) {
  switch (opcode) {
    POPLAR_OPCODE(kAbs, "Abs");
    POPLAR_OPCODE(kAdd, "Add");
    UNUSED_OPCODE(kBitcast);
    UNUSED_OPCODE(kBroadcast);
    UNUSED_OPCODE(kCall);
    POPLAR_OPCODE(kCeil, "Ceil");
    POPLAR_OPCODE(kClamp, "");
    UNUSED_OPCODE(kConcatenate);
    UNUSED_OPCODE(kConstant);
    UNUSED_OPCODE(kConvert);
    UNUSED_OPCODE(kConvolution);
    UNUSED_OPCODE(kCopy);
    POPLAR_OPCODE(kCrossReplicaSum, "");
    UNUSED_OPCODE(kCustomCall);
    POPLAR_OPCODE(kDivide, "Div");
    POPLAR_OPCODE(kDot, "Dot");
    UNUSED_OPCODE(kDynamicSlice);
    UNUSED_OPCODE(kDynamicUpdateSlice);
    POPLAR_OPCODE(kEq, "EqualTo");
    POPLAR_OPCODE(kExp, "Exp");
    POPLAR_OPCODE(kFloor, "Floor");
    POPLAR_OPCODE(kFusion, "");
    POPLAR_OPCODE(kGe, "GreaterEqual");
    UNUSED_OPCODE(kGetTupleElement);
    POPLAR_OPCODE(kGt, "GreaterThan");
    POPLAR_OPCODE(kIndex, "");
    UNUSED_OPCODE(kInfeed);
    POPLAR_OPCODE(kIsFinite, "");
    POPLAR_OPCODE(kLe, "LessEqual");
    POPLAR_OPCODE(kLog, "Log");
    POPLAR_OPCODE(kLogicalAnd, "LogicalAnd");
    POPLAR_OPCODE(kLogicalNot, "LogicalNot");
    POPLAR_OPCODE(kLogicalOr, "LogicalOr");
    POPLAR_OPCODE(kLt, "LessThan");
    UNUSED_OPCODE(kMap);
    POPLAR_OPCODE(kMaximum, "Maximum");
    POPLAR_OPCODE(kMinimum, "Minimum");
    POPLAR_OPCODE(kMultiply, "Mul");
    POPLAR_OPCODE(kNe, "NotEqual");
    POPLAR_OPCODE(kNegate, "Neg");
    UNUSED_OPCODE(kOutfeed);
    UNUSED_OPCODE(kPad);
    UNUSED_OPCODE(kParameter);
    POPLAR_OPCODE(kPower, "Pow");
    UNUSED_OPCODE(kRecv);
    UNUSED_OPCODE(kReduce);
    UNUSED_OPCODE(kReduceWindow);
    POPLAR_OPCODE(kRemainder, "Remainder");
    UNUSED_OPCODE(kReshape);
    POPLAR_OPCODE(kReverse, "");
    POPLAR_OPCODE(kRng, "");
    POPLAR_OPCODE(kSelect, "Select");
    POPLAR_OPCODE(kSelectAndScatter, "");
    UNUSED_OPCODE(kSend);
    POPLAR_OPCODE(kSign, "Sign");
    UNUSED_OPCODE(kSlice);
    POPLAR_OPCODE(kSort, "");
    POPLAR_OPCODE(kSubtract, "Sub");
    POPLAR_OPCODE(kTanh, "Tanh");
    POPLAR_OPCODE(kTrace, "");
    UNUSED_OPCODE(kTranspose);
    UNUSED_OPCODE(kTuple);
    POPLAR_OPCODE(kUpdate, "");
    POPLAR_OPCODE(kWhile, "");
  }
  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

port::StatusOr<poplar::Tensor>
FindInstructionInput(const TensorMap& map,
                     const HloInstruction* inst,
                     uint64 input,
                     uint64 n) {
  auto it = map.find(std::make_pair(inst->operand(input)->name(),n));
  if (it == map.end()) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar] Couldn't find input ",
                                     input,
                                     " for ",
                                     inst->name()));
  }
  return it->second;
}

port::StatusOr<poplar::Tensor>
FindInstructionOutput(const TensorMap& map,
                      const HloInstruction* inst,
                      uint64 n) {
  auto it = map.find(std::make_pair(inst->name(),n));
  if (it == map.end()) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar] Couldn't find output for ",
                                     inst->name(),
                                     ":", n));
  }
  return it->second;
}

port::Status
AddOutputTensor(TensorMap& map,
                const HloInstruction* inst,
                uint64 n,
                const poplar::Tensor& tensor) {
  auto p = std::make_pair(inst->name(),n);
  auto it = map.find(p);
  if (it != map.end()) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar] Ouptut Tensor for ",
                                     inst->name(),
                                     " already exists"));
  }
  map[p] = tensor;
  return Status::OK();
}

}
}
