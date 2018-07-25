#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

std::string GetDebugName(const HloInstruction* inst) {
  const std::string& tf_core_name = inst->metadata().op_name();
  return tf_core_name + "/" + inst->name();
}

std::pair<int64, int64> FindTupleInputIndices(const HloInstruction* tuple,
                                              int64 n) {
  int64 start = 0;
  for (int64 i = 0; i < n; i++) {
    start += CountShapes(tuple->operand(i)->shape());
  }
  int64 end = start + CountShapes(tuple->operand(n)->shape());
  return std::make_pair(start, end);
}

ArgVector FindTupleInInstructionInput(const TensorMap& map,
                                      const HloInstruction* inst, int64 input,
                                      int64 n) {
  const HloInstruction* operand = inst->operand(input);
  const Shape& shape = operand->shape();
  OutVector outputs = FindInstructionOutputs(map, operand);
  int64 start = 0;
  for (int64 i = 0; i < n; i++) {
    start += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
  }
  int64 end = start + CountShapes(ShapeUtil::GetTupleElementShape(shape, n));

  return ArgVector(&outputs[start], &outputs[end]);
}

StatusOr<poplar::Tensor> FindInstructionInput(const TensorMap& map,
                                              const HloInstruction* inst,
                                              int64 input) {
  const HloInstruction* operand = inst->operand(input);
  OutVector outputs = FindInstructionOutputs(map, operand);
  if (outputs.size() == 0) {
    return tensorflow::errors::Unknown(se::port::StrCat(
        "[Poplar] Couldn't find input ", input, " for ", inst->name()));
  }
  return outputs[0];
}

ArgVector FindInstructionInputs(const TensorMap& map,
                                const HloInstruction* inst, int64 input) {
  const HloInstruction* operand = inst->operand(input);
  OutVector inputs = FindInstructionOutputs(map, operand);
  return inputs;
}

OutVector FindInstructionOutputs(const TensorMap& map,
                                 const HloInstruction* inst) {
  auto lower = std::make_pair(inst->name(), 0);
  auto upper = std::make_pair(inst->name(), std::numeric_limits<int64>::max());
  OutVector outputs;
  for (auto it = map.lower_bound(lower); it != map.upper_bound(upper); it++) {
    outputs.push_back(it->second);
  }
  return outputs;
}

Status AddOutputTensor(TensorMap& map, const HloInstruction* inst, int64 n,
                       const poplar::Tensor& tensor) {
  auto p = std::make_pair(inst->name(), n);
  auto it = map.find(p);
  if (it != map.end()) {
    return tensorflow::errors::Unknown(se::port::StrCat(
        "[Poplar] Ouptut Tensor for ", GetDebugName(inst), " already exists"));
  }
  map[p] = tensor;
  return Status::OK();
}

template <typename TYPE>
static void SetVertexField(poplar::Graph& graph, const poplar::FieldRef& field,
                           const Literal& literal) {
  const TYPE* value(static_cast<const TYPE*>(literal.untyped_data()));
  graph.setInitialValue<TYPE>(field, *value);
}

static void SetFp16VertexField(poplar::Graph& graph,
                               const poplar::FieldRef& field,
                               const Literal& literal) {
  const uint16_t* value(static_cast<const uint16_t*>(literal.untyped_data()));
  graph.setInitialValueHalf(field, *value);
}

Status SetVertexField(poplar::Graph& graph, const poplar::FieldRef& field,
                      const Literal& literal) {
  switch (literal.shape().element_type()) {
    case PRED:
      SetVertexField<bool>(graph, field, literal);
      break;
    case S32:
    case U32:
      SetVertexField<int>(graph, field, literal);
      break;
    case F16:
      SetFp16VertexField(graph, field, literal);
      break;
    case F32:
      SetVertexField<float>(graph, field, literal);
      break;
    default:
      return xla::FailedPrecondition("Unrecognised type in SetVertexField: %d",
                                     literal.shape().element_type());
  }
  return Status::OK();
}

void PrintTensorMapping(const poplar::Graph& graph,
                        const TensorMap& tensor_map) {
  if (VLOG_IS_ON(2)) {
    std::stringstream ss;
    VLOG(2) << "[Poplar] Dumping tensor mapping";
    // Printed in JSON format where
    // {"mapping": [
    //  {
    //    "inst_name": "name",
    //    "output_index": output_index,
    //    "tiles_used": tiles_used,
    //    "total_memory_size": total_memory_size,
    //    "tiles": [ {"tile_id": tile_id, "num_intervals": num_intervals,
    //    "memory_size": memory_size}, ...]
    //  } ...
    //
    //  ]
    // }
    ss << " {\"mapping\": [";
    bool first_tensor = true;
    for (auto pair : tensor_map) {
      if (!first_tensor) ss << ",";
      const auto inst_name = pair.first.first;
      const auto output_index = pair.first.second;
      const auto tensor = pair.second;
      ss << " { "
         << "\"inst_name\": \"" << inst_name << "\", "
         << "\"output_index\": " << output_index << ", "
         << "\"tiles\": [";
      const auto mapping = graph.getTileMapping(tensor);
      bool first_tile = true;
      unsigned tilesUsed = 0;
      size_t totalMemory = 0;
      for (size_t tileIdx = 0; tileIdx < mapping.size(); tileIdx++) {
        const auto& tile = mapping[tileIdx];
        if (tile.size() != 0) {
          if (!first_tile) ss << ", ";
          tilesUsed++;
          size_t tileMemSize = 0;
          for (const auto& interval : tile) {
            tileMemSize += interval.size();
          }
          ss << "{\"tile_id\": " << tileIdx << ", "
             << "\"num_intervals\": " << tile.size() << ", "
             << "\"memory_size\": " << tileMemSize << "}";
          first_tile = false;
          totalMemory += tileMemSize;
        }
      }
      ss << "], "
         << "\"total_memory_size\": " << totalMemory << ", "
         << "\"tiles_used\": " << tilesUsed << " }";
      first_tensor = false;
    }
    ss << "]}";
    VLOG(2) << ss.str();
  }
}

}  // namespace poplarplugin
}  // namespace xla
