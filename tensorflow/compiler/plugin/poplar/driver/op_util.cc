#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace poplarplugin {

static int64
CountShapes(const Shape& shape) {
  int64 n = 0;
  if (ShapeUtil::IsTuple(shape)) {
    for (int64 i=0; i<ShapeUtil::TupleElementCount(shape); i++) {
      n += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return n;
  } else {
    return 1;
  }
};

std::vector<poplar::Tensor>
FindTupleInInstructionInput(const TensorMap& map,
                            const HloInstruction* inst,
                            int64 input,
                            int64 n) {
  const HloInstruction* operand = inst->operand(input);
  const Shape& shape = operand->shape();
  std::vector<poplar::Tensor> outputs = FindInstructionOutputs(map, operand);
  int64 start=0;
  for (int64 i=0; i<n; i++) {
    start += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
  }
  int64 end = start + CountShapes(ShapeUtil::GetTupleElementShape(shape, n));

  return std::vector<poplar::Tensor>(&outputs[start], &outputs[end]);
}

port::StatusOr<poplar::Tensor>
FindInstructionInput(const TensorMap& map,
                     const HloInstruction* inst,
                     int64 input) {
  const HloInstruction* operand = inst->operand(input);
  std::vector<poplar::Tensor> outputs = FindInstructionOutputs(map, operand);
  if (outputs.size() == 0) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar] Couldn't find input ",
                                     input,
                                     " for ",
                                     inst->name()));
  }
  return outputs[0];
}

std::vector<poplar::Tensor>
FindInstructionInputs(const TensorMap& map,
                      const HloInstruction* inst,
                      int64 input) {
  const HloInstruction* operand = inst->operand(input);
  std::vector<poplar::Tensor> outputs = FindInstructionOutputs(map, operand);
  return outputs;
}

std::vector<poplar::Tensor>
FindInstructionOutputs(const TensorMap& map,
                       const HloInstruction* inst) {
  auto lower = std::make_pair(inst->name(), 0);
  auto upper = std::make_pair(inst->name(), std::numeric_limits<int64>::max());
  std::vector<poplar::Tensor> outputs;
  for (auto it = map.lower_bound(lower); it != map.upper_bound(upper); it++) {
    outputs.push_back(it->second);
  }
  return outputs;
}

port::Status
AddOutputTensor(TensorMap& map,
                const HloInstruction* inst,
                int64 n,
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
