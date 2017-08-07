#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace poplarplugin {

port::StatusOr<poplar::Tensor>
FindInstructionInput(const TensorMap& map,
                     const HloInstruction* inst,
                     int64 input,
                     int64 n) {
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
