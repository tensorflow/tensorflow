#include "include/json/json.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <algorithm>
#include <limits>

#include <poputil/TileMapping.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

poplar::Graph& GetGraph(CompilerResources& res, const HloInstruction* inst) {
  if (inst->has_sharding()) {
    const auto& sharding = inst->sharding();
    if (sharding.HasUniqueDevice()) {
      uint64 device_id = sharding.GetUniqueDevice();
      if (device_id < res.shard_graphs.size()) {
        return res.shard_graphs[device_id];
      }
    }
  }

  if (res.replicated_graph.has_value()) {
    return res.replicated_graph.value();
  }

  return res.main_graph;
}

uint64 GetShardingDeviceId(const HloInstruction* inst) {
  // This function works on the assumptions:
  // * that all the instructions either have sharding or none of them do (see
  //   ShardingPass).
  // * If an instruction has sharding, then that sharding contains a unique
  //   device.
  return inst->has_sharding() && inst->sharding().HasUniqueDevice()
             ? inst->sharding().GetUniqueDevice()
             : 0;
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
      SetVertexField<unsigned>(graph, field, literal);
      break;
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

Status PoplarExceptionToTensorflowStatus(const std::string& prefix,
                                         const std::exception& e) {
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
  } catch (const poputil::poplibs_error& e) {
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
