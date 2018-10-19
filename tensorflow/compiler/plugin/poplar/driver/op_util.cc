#include "include/json/json.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/stream_executor/lib/strcat.h"

#include <algorithm>
#include <limits>

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

StatusOr<poplar::Tensor> GetInplaceOutputTensor(poplar::Graph& graph,
                                                CompilerResources& res,
                                                poplar::program::Sequence& seq,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0));

  // We need to add a copy before an inplace op if:
  // 1. in0 is not ParallelWriteable
  // 2. inst has been removed from inplace ops by a different pass
  // 3. in0 is marked as inplace but its parent at operand 0 is not inplace but
  //    it's input and output tensor intersect

  bool requires_copy_inplace = !in0.isParallelWriteable();

  if (res.annotations.inplace_instructions.IsInPlace(inst)) {
    bool parent_not_inplace_same_tensor = false;
    const auto* parent = inst->operand(0);

    if (parent->operand_count() &&
        !res.annotations.inplace_instructions.IsInPlace(parent)) {
      poplar::Tensor parent_in0;
      TF_ASSIGN_OR_RETURN(parent_in0,
                          FindInstructionInput(tensor_map, parent, 0));
      OutVector parent_outs = FindInstructionOutputs(tensor_map, parent);
      CHECK_EQ(parent_outs.size(), 1);
      poplar::Tensor parent_out = parent_outs[0];
      parent_not_inplace_same_tensor |= parent_out.intersectsWith(parent_in0);
    }
    requires_copy_inplace |= parent_not_inplace_same_tensor;
  } else {
    requires_copy_inplace = true;
  }

  if (requires_copy_inplace) {
    VLOG(1) << "Adding a copy for inplace op " << inst->name();
    poplar::Tensor copy = graph.clone(in0, GetDebugName(inst) + ".clone");
    seq.add(poplar::program::Copy(in0, copy));
    in0 = copy;
  }
  return in0;
}

StatusOr<poplar::Tensor> AddOutputTensor(poplar::Graph& graph,
                                         CompilerResources& res,
                                         poplar::program::Sequence& seq,
                                         TensorMap& map,
                                         const HloInstruction* inst, int64 n,
                                         const poplar::Tensor& tensor) {
  poplar::Tensor out = tensor;
  if (inst->operand_count() &&
      !res.annotations.inplace_instructions.IsInPlace(inst)) {
    // If the output tensor for non inplace op intersects with the tensor for
    // inst->operand(0) and one of dependency successors of inst is an inplace
    // op with inst->operand(0) as the operand 0, then we need to clone this
    // output tensor so that the inplace op can still be performed
    poplar::Tensor in0;
    if (inst->opcode() == HloOpcode::kGetTupleElement) {
      in0 = FindTupleInInstructionInput(map, inst, 0, inst->tuple_index())[n];
    } else {
      TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(map, inst, 0));
    }

    if (in0.intersectsWith(tensor)) {
      bool clone_output = false;
      for (const auto* succ : inst->control_successors()) {
        if (res.annotations.inplace_instructions.IsInPlace(succ) &&
            succ->operand(0) == inst->operand(0)) {
          clone_output = true;
          break;
        }
      }
      if (clone_output) {
        VLOG(1) << "Adding a clone for output tensor of " << inst->name();
        out = graph.clone(tensor, GetDebugName(inst) + ".clone");
        seq.add(poplar::program::Copy(tensor, out));
      }
    }
  }

  auto p = std::make_pair(inst->name(), n);
  auto it = map.find(p);
  if (it != map.end()) {
    return tensorflow::errors::Unknown(se::port::StrCat(
        "[Poplar] Ouptut Tensor for ", GetDebugName(inst), " already exists"));
  }
  map[p] = out;
  return out;
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

std::string GetTensorMappingJson(const poplar::Graph& graph,
                                 const TensorMaps& tensor_maps) {
  Json::Value mappings;

  for (auto tm : tensor_maps) {
    mappings[tm.first] = Json::Value(Json::arrayValue);

    for (auto pair : tm.second) {
      const auto& pop_tensor = pair.second;

      Json::Value tensor;
      tensor["inst_name"] = Json::Value(pair.first.first);
      tensor["output_index"] = Json::Value::UInt64(pair.first.second);
      tensor["constant"] = Json::Value::UInt64(pop_tensor.containsConstant());
      tensor["tiles"] = Json::Value(Json::arrayValue);

      const auto& mapping = graph.getTileMapping(pop_tensor);
      unsigned tiles_used = 0;
      size_t total_elements = 0;

      for (size_t tile_idx = 0; tile_idx < mapping.size(); tile_idx++) {
        const auto& tile = mapping[tile_idx];
        if (tile.size() != 0) {
          tiles_used++;
          size_t tile_element_count = 0;
          for (const auto& interval : tile) {
            tile_element_count += interval.size();
          }

          Json::Value tile;
          tile["tile_id"] = Json::Value::UInt64(tile_idx);
          tile["num_intervals"] = Json::Value::UInt64(tile.size());
          tile["num_elements"] = Json::Value::UInt64(tile_element_count);
          tile["element_type"] =
              Json::Value(pop_tensor.elementType().toString());
          tensor["tiles"].append(tile);

          total_elements += tile_element_count;
        }
      }

      tensor["tiles_used"] = Json::Value::UInt64(tiles_used);
      tensor["total_elements"] = Json::Value::UInt64(total_elements);

      mappings[tm.first].append(tensor);
    }
  }

  Json::Value root;
  root["mappings"] = mappings;

  Json::StreamWriterBuilder json_builder;
  std::string json_msg = Json::writeString(json_builder, root);

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "[Poplar] Dumping tensor mapping";
    VLOG(2) << json_msg;
  }

  return json_msg;
}

Status PoplarExceptionToTensorflowStatus(const std::string& prefix,
                                         const std::logic_error& e) {
  /* NOTE: Reduce this list if/when Poplar errors are subclassed */
  try {
    std::rethrow_exception(std::current_exception());
  } catch (const poplar::file_load_error& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::missing_cycle_estimate& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::symbol_error& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::unknown_field& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::unknown_vertex_type& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::no_environment& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::parse_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::invalid_option& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::invalid_machine_model& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::stream_connection_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::graph_cycle_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::invalid_tile_mapping& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::type_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::no_size_specified& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::profiling_disabled& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::control_program_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::runtime_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::overflow_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::tensor_io_state_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_connection_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_object_load_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_object_creation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_program_compilation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poputil::poplib_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::link_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::stream_memory_allocation_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::graph_memory_allocation_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::tensor_creation_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::memory_elem_constraints_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::index_error& e) {
    return tensorflow::errors::OutOfRange(prefix, e.what());
  } catch (const poplar::poplar_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  }

  return tensorflow::errors::Unknown(prefix, e.what());
}

}  // namespace poplarplugin
}  // namespace xla
