#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace poplarplugin {

ArgVector
FindTupleInInstructionInput(const TensorMap& map,
                            const HloInstruction* inst,
                            int64 input,
                            int64 n) {
  const HloInstruction* operand = inst->operand(input);
  const Shape& shape = operand->shape();
  OutVector outputs = FindInstructionOutputs(map, operand);
  int64 start=0;
  for (int64 i=0; i<n; i++) {
    start += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
  }
  int64 end = start + CountShapes(ShapeUtil::GetTupleElementShape(shape, n));

  return ArgVector(&outputs[start], &outputs[end]);
}

port::StatusOr<poplar::Tensor>
FindInstructionInput(const TensorMap& map,
                     const HloInstruction* inst,
                     int64 input) {
  const HloInstruction* operand = inst->operand(input);
  OutVector outputs = FindInstructionOutputs(map, operand);
  if (outputs.size() == 0) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar] Couldn't find input ",
                                     input,
                                     " for ",
                                     inst->name()));
  }
  return outputs[0];
}

ArgVector
FindInstructionInputs(const TensorMap& map,
                      const HloInstruction* inst,
                      int64 input) {
  const HloInstruction* operand = inst->operand(input);
  OutVector inputs = FindInstructionOutputs(map, operand);
  return inputs;
}

OutVector
FindInstructionOutputs(const TensorMap& map,
                       const HloInstruction* inst) {
  auto lower = std::make_pair(inst->name(), 0);
  auto upper = std::make_pair(inst->name(), std::numeric_limits<int64>::max());
  OutVector outputs;
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

template<typename TYPE>
static void SetVertexField(poplar::Graph& graph,
                           const poplar::FieldRef &field,
                           const Literal& literal) {
  const TYPE* value(static_cast<const TYPE*>(literal.untyped_data()));
  graph.setInitialValue<TYPE>(field, *value);
}

port::Status SetVertexField(poplar::Graph &graph,
                            const poplar::FieldRef &field,
                            const Literal &literal) {
  switch (literal.shape().element_type()) {
    case PRED:
      SetVertexField<bool>(graph, field, literal);
      break;
    case S32:
    case U32:
      SetVertexField<int>(graph, field, literal);
      break;
    case F16:
      SetVertexField<poplar::IeeeHalf>(graph, field, literal);
      break;
    case F32:
      SetVertexField<float>(graph, field, literal);
      break;
    default:
      return port::Status(port::error::FAILED_PRECONDITION,
                          port::StrCat("Unrecognised type in SetVertexField: ",
                                       literal.shape().element_type()));
  }
  return Status::OK();
}

}
}
