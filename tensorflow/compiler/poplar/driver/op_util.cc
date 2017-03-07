#include <algorithm>

#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace poplarplugin {

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
