/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/kernels/tpu_functional_ops.h"

#include <memory>

#include "absl/strings/match.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

#define EIGEN_USE_THREADS

#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_fingerprint_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

constexpr char kTpuReplicateAttr[] = "_tpu_replicate";
constexpr int kLastDimOfTpuInputFastPath = 128;
constexpr int kOtherDimOfTpuInputFastPath = 8;

constexpr char kXLAShardingAttrName[] = "sharding";
constexpr char kXLAShardingAttrAltName[] = "_XlaSharding";

Status GenerateDeviceNaturalOrder(int x_num_cores, int y_num_cores,
                                  int z_num_cores, int num_cores_per_chip,
                                  std::vector<int>* natural_order) {
  for (int y = 0; y < y_num_cores; ++y) {
    for (int x = 0; x < x_num_cores; ++x) {
      for (int z = 0; z < z_num_cores; ++z) {
        for (int c = 0; c < num_cores_per_chip; ++c) {
          natural_order->push_back(x);
          natural_order->push_back(y);
          natural_order->push_back(z);
          natural_order->push_back(c);
        }
      }
    }
  }

  return OkStatus();
}

struct TPUVariableInfo {
  TPUVariableInfo(int device_ordinal_id, bool use_fast_mem)
      : device_ordinal(device_ordinal_id), fast_mem(use_fast_mem) {}
  // The TPU core which the variable will be placed on.
  int device_ordinal;
  // If true, try to place the variable on fast memory space if hardware
  // support.
  bool fast_mem;
};

// Check the descendants to parse the placement information for the input node.
// num_cores_per_replica descriables how many cores the single model uses.
Status ParseTPUVariableInfor(const Node* node, const int num_cores_per_replica,
                             TPUVariableInfo* var_info) {
  int core = 0;
  bool use_fast_mem = false;
  VLOG(3) << "Parse tpu variable information for " << node->name();
  for (const Edge* edge : node->out_edges()) {
    if (edge->IsControlEdge()) continue;
    Node* next = edge->dst();
    VLOG(3) << "Neighbor node " << next->name();
    // Looking through Enter/Switch/ReadVariableOp nodes.
    while (next->IsEnter() || next->IsSwitch() ||
           next->type_string() == "ReadVariableOp") {
      Node* new_node = nullptr;
      for (const Edge* e : next->out_edges()) {
        if (!e->IsControlEdge()) {
          new_node = e->dst();
          break;
        }
      }
      if (new_node == nullptr) break;
      next = new_node;
    }
    if (next != edge->dst()) {
      VLOG(3) << "Looked through Enter/Switch node " << next->DebugString();
    }
    TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                        ParseShardingFromDevice(*next, num_cores_per_replica,
                                                /*add_metadata=*/false));
    if (sharding.has_value() && sharding->tile_assignment_devices_size() > 0) {
      core = sharding->tile_assignment_devices(0);
      VLOG(3) << next->name() << " is placed on core " << core;
    }
    if (next->attrs().Find(TPU_FAST_MEM_ATTR) != nullptr) {
      use_fast_mem = true;
      VLOG(3) << next->name() << " has " << TPU_FAST_MEM_ATTR << " attribute";
    }
  }
  VLOG(1) << "Place " << node->name() << " to core: " << core
          << " fast_mem: " << use_fast_mem;
  var_info->device_ordinal = core;
  var_info->fast_mem = use_fast_mem;

  return OkStatus();
}

// Helper to instantiate function "func" in the library "lib".
Status Instantiate(FunctionLibraryRuntime* lib, const NameAttrList& func,
                   FunctionLibraryRuntime::Handle* handle) {
  return lib->Instantiate(func.name(), AttrSlice(&func.attr()), handle);
}

static constexpr const char* const kDeviceOrdinalAttr = "device_ordinal";

static constexpr const char* const kTPUExecuteOp = "TPUExecute";
static constexpr const char* const kInfeedEnqueueOp = "InfeedEnqueue";
static constexpr const char* const kInfeedEnqueueTupleOp = "InfeedEnqueueTuple";
static constexpr const char* const kOutfeedDequeueOp = "OutfeedDequeue";
static constexpr const char* const kOutfeedDequeueTupleOp =
    "OutfeedDequeueTuple";
static constexpr const char* const kOutfeedDequeueV2Op = "OutfeedDequeueV2";
static constexpr const char* const kOutfeedDequeueTupleV2Op =
    "OutfeedDequeueTupleV2";
static constexpr const char* const kVarHandleOp = "VarHandleOp";

static constexpr const char* const kTPUDeviceNamePrefix = "/device:TPU:";
static constexpr const int kTPUDefaultDeviceOrdinal = 0;

bool IsSupportedTPUOp(const string& op_name) {
  return op_name == kTPUExecuteOp || op_name == kInfeedEnqueueOp ||
         op_name == kInfeedEnqueueTupleOp || op_name == kOutfeedDequeueOp ||
         op_name == kOutfeedDequeueTupleOp || op_name == kOutfeedDequeueV2Op ||
         op_name == kOutfeedDequeueTupleV2Op;
}

// Sets the sharding attributes for an XlaSharding node.
void SetXlaShardingNodeAttr(Node* xla_sharding_node, int num_cores_per_replica,
                            int rank, int shard_dim) {
  auto sharding = absl::make_optional<xla::OpSharding>();
  sharding->set_type(xla::OpSharding::OTHER);

  std::vector<int64_t> dims(rank, 1LL);
  dims[shard_dim] = num_cores_per_replica;
  for (auto dim : dims) {
    sharding->add_tile_assignment_dimensions(dim);
  }

  // Sets up tile_assignment_devices.
  for (int d = 0; d < num_cores_per_replica; ++d) {
    sharding->add_tile_assignment_devices(d);
  }

  xla_sharding_node->ClearAttr(kXLAShardingAttrName);
  xla_sharding_node->ClearAttr(kXLAShardingAttrAltName);
  xla_sharding_node->AddAttr(kXLAShardingAttrName,
                             sharding->SerializeAsString());
  xla_sharding_node->AddAttr(kXLAShardingAttrAltName,
                             sharding->SerializeAsString());
}

// If 'device_name' is a TPU device, set its device_ordinal to 'device_ordinal'
// and set '*rewritten' to true. Otherwise, do nothing.
Status UpdateTPUDeviceOrdinal(int device_ordinal, string* device_name,
                              bool* rewritten) {
  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(*device_name, &device)) {
    return errors::InvalidArgument("Unable to parse device name ",
                                   *device_name);
  }
  if (device.type == DEVICE_TPU_NODE) {
    device.id = device_ordinal;
    *rewritten = true;
  }
  *device_name = DeviceNameUtils::ParsedNameToString(device);
  return OkStatus();
}

const Edge* FindHostToDeviceEdge(Node* arg_node) {
  const Edge* candidate_edge = nullptr;
  for (const Edge* edge : arg_node->out_edges())
    if (!edge->IsControlEdge()) {
      // Find CPU -> TPU input edge.
      const Edge* original_edge;
      while (edge->src()->attrs().Find(kTpuReplicateAttr) != nullptr ||
             edge->dst()->attrs().Find(kTpuReplicateAttr) == nullptr) {
        const Node* new_src = edge->dst();
        original_edge = edge;
        for (const Edge* new_edge : new_src->out_edges())
          if (!new_edge->IsControlEdge()) {
            original_edge = edge;
            edge = new_edge;
            break;
          }
        if (original_edge == edge) break;
      }
      // TPU input edge: src is on CPU and dest is on TPU.
      if (edge->src()->attrs().Find(kTpuReplicateAttr) != nullptr ||
          edge->dst()->attrs().Find(kTpuReplicateAttr) == nullptr)
        continue;
      // Won't work with GuaranteeConst.
      if (edge->src()->type_string() == "GuaranteeConst") break;
      candidate_edge = edge;
    }
  return candidate_edge;
}

Status CreateInputProxy(Graph* graph, const Edge* candidate_edge,
                        const Edge** tpu_input_edge) {
  std::vector<const Edge*> edges_to_replace;
  for (const Edge* input_edge : candidate_edge->src()->out_edges()) {
    if (!input_edge->IsControlEdge() &&
        input_edge->dst()->attrs().Find(kTpuReplicateAttr) != nullptr)
      edges_to_replace.push_back(input_edge);
  }
  // Build an Identity node as the proxy of the original edge source.
  Node* input_identity_node = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(candidate_edge->src()->name(), "/proxy"),
                  "Identity")
          .Input(candidate_edge->src())
          .Attr("T", candidate_edge->src()->output_type(0))
          .Attr(kTpuReplicateAttr,
                candidate_edge->dst()->attrs().Find(kTpuReplicateAttr)->s())
          .Finalize(graph, &input_identity_node));
  // Find the tpu input edge from original source to proxy identity.
  for (const Edge* input_edge : input_identity_node->in_edges())
    if (input_edge->src() == candidate_edge->src()) {
      *tpu_input_edge = input_edge;
      break;
    }
  // Replace original input edges with proxy's output.
  for (const Edge* input_edge : edges_to_replace) {
    graph->RemoveEdge(input_edge);
    graph->AddEdge(input_identity_node, 0, input_edge->dst(),
                   input_edge->dst_input());
  }
  return OkStatus();
}

Status GetClusterName(Graph* graph, string* cluster_name) {
  *cluster_name = "";
  for (const Node* node : graph->nodes()) {
    if (node->attrs().Find(kTpuReplicateAttr) == nullptr) continue;
    if (cluster_name->empty())
      *cluster_name = node->attrs().Find(kTpuReplicateAttr)->s();
    // When optimization is turned on, the graph should only have one TPU
    // cluster.
    if (*cluster_name != node->attrs().Find(kTpuReplicateAttr)->s())
      return errors::FailedPrecondition(
          "Only one cluster is allowed when optimization is turned on for "
          "TPUPartitionedCall. Found ",
          node->attrs().Find(kTpuReplicateAttr)->s(), " and ", *cluster_name);
  }
  return OkStatus();
}

// Removes nodes that has no effect that directly descends from _Arg node.
//
// This is currently used for removing TPUReplicatedInput and XlaSharding node
// are always descendants of _Arg node. During optimization, we try to insert
// nodes in between _Arg and _Arg's children, where some of the nodes inserted
// are TPU nodes. We will add the TPUReplicatedInput and XlaSharding op nodes
// back where necessary.
//
// Returns the number of nodes that were removed.
int64_t RemoveDescendantNodeOfArg(
    Graph* graph, const std::string& node_type_to_remove,
    const std::set<std::string>& must_be_child_of) {
  int64_t nodes_removed = 0;
  std::vector<std::pair<const Edge*, std::vector<const Edge*>>> edges_to_remove;

  for (Node* node : graph->nodes()) {
    if (node_type_to_remove != node->type_string()) continue;
    if (!must_be_child_of.empty()) {
      bool has_arg_parent = false;
      for (const Edge* edge : node->in_edges()) {
        if (must_be_child_of.count(edge->src()->type_string()) > 0) {
          has_arg_parent = true;
        }
      }
      if (!has_arg_parent) continue;
    }
    nodes_removed++;

    const Edge* input_edge = nullptr;
    std::vector<const Edge*> output_edges;
    for (const Edge* edge : node->in_edges())
      if (!edge->IsControlEdge()) {
        input_edge = edge;
        break;
      }
    for (const Edge* edge : node->out_edges())
      if (!edge->IsControlEdge()) {
        output_edges.push_back(edge);
      }
    if (input_edge != nullptr && !output_edges.empty())
      edges_to_remove.push_back(std::make_pair(input_edge, output_edges));
  }
  for (const auto& it : edges_to_remove) {
    for (const Edge* output_edge : it.second) {
      graph->RemoveEdge(output_edge);
      graph->AddEdge(it.first->src(), it.first->src_output(),
                     output_edge->dst(), output_edge->dst_input());
    }
    graph->RemoveNode(it.first->dst());
  }
  return nodes_removed;
}

uint64 GetInputHash(OpKernelContext* ctx) {
  uint64 input_hash = 0;  // initialization for determinism.
  // Use the number of elements to compute hash.
  // TODO(chiachenc): use fhe full shape to compute the hash.
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    VLOG(4) << "InputHash, combine input " << i
            << ", NumElements: " << ctx->input(i).NumElements();
    input_hash = Hash64Combine(input_hash, ctx->input(i).NumElements());
  }
  return input_hash;
}

string HashShapeAndType(const string prefix, const std::vector<int>& input_dims,
                        const DataType& dtype, const bool input_shape_opt) {
  string hash = strings::StrCat(prefix, dtype, "_dims");
  // We will concat at the last dimension.
  for (int d = 0; d < input_dims.size() - 1; ++d) {
    strings::StrAppend(&hash, "_", input_dims.at(d));
  }

  if (input_shape_opt) {
    if (input_dims.back() % kLastDimOfTpuInputFastPath == 0) {
      strings::StrAppend(&hash, "_last_", kLastDimOfTpuInputFastPath, "n");
    } else {
      strings::StrAppend(&hash, "_last_other");
    }
  }
  return hash;
}

// Get the information for input and output tensors (shapes, dtypes, etc).
Status GetInputOutputInfo(
    Graph* graph, GraphShapeInfo& tpu_inferred_info,
    std::map<int, InferredShape>& arg_shapes, EdgeShapes& tpu_input_shapes,
    absl::flat_hash_map<const Edge*, DataType>& tpu_input_dtypes,
    OpKernelContext* ctx) {
  // Search for the device-to-host or tpu-to-cpu edges.
  for (Node* node : graph->op_nodes()) {
    if (!node->IsArg()) continue;
    const DataType dtype = node->attrs().Find("T")->type();
    const int arg_index = node->attrs().Find("index")->i();
    if (dtype != DT_INT32 && dtype != DT_BFLOAT16 && dtype != DT_FLOAT &&
        dtype != DT_BOOL && dtype != DT_QINT8 && dtype != DT_QUINT8)
      continue;
    VLOG(3) << "Argnode: " << node->DebugString();
    const Tensor& tensor = ctx->input(arg_index);

    // Search for the cross-device edge from arg node.
    const Edge* candidate_edge = FindHostToDeviceEdge(node);
    if (candidate_edge == nullptr) continue;

    // Make proxy and get the sole tpu_input_edge for transfer the input tensor
    // corresponding to the current _Arg node.
    const Edge* tpu_input_edge = nullptr;
    TF_RETURN_IF_ERROR(
        CreateInputProxy(graph, candidate_edge, &tpu_input_edge));
    if (tpu_input_edge == nullptr)
      return errors::NotFound("Couldn't find TPU input edge for", node->name());

    // Optimize edge: original source to proxy identity.
    VLOG(3) << "Input: " << tpu_input_edge->src()->name();
    std::vector<int>& input_shapes = tpu_input_shapes[tpu_input_edge];
    input_shapes.clear();
    for (int d = 0; d < tensor.dims(); ++d) {
      input_shapes.push_back(tensor.dim_size(d));
      VLOG(3) << "Input Tensor: Dim[" << d << "] = " << tensor.dim_size(d);
    }
    tpu_input_dtypes[tpu_input_edge] = tensor.dtype();

    // Collect shapes for non-resource-variable args.
    PartialTensorShape partial_tensor_shape;
    auto partial_shape = PartialTensorShape::MakePartialShape(
        input_shapes.data(), input_shapes.size(), &partial_tensor_shape);
    InferredShape inferred_shape = {partial_tensor_shape};
    arg_shapes[arg_index] = inferred_shape;
  }
  return OkStatus();
}

// Converts a integer vector that represents the shapes to a Tensorshape.
Status ConvertEdgeShapesToTensorShapes(
    const std::map<std::string, std::vector<int>>& named_input_shapes,
    std::vector<TensorShape>* shapes) {
  shapes->resize(named_input_shapes.size());
  int32_t i = 0;
  // keys in tpu_input_shapes may be stale.
  for (const auto& iter : named_input_shapes) {
    VLOG(2) << iter.first << ", rank: " << iter.second.size();
    const int64_t rank = iter.second.size();
    std::vector<int64_t> dims(rank);
    for (int64_t d = 0; d < rank; ++d) {
      VLOG(2) << " dim[" << d << "]: " << iter.second.at(d);
      dims[d] = iter.second.at(d);
    }
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(dims, &(*shapes)[i]));
    i++;
  }
  return OkStatus();
}

// Get the TF fingerprint with the information from the TPUCompileOp or
// _TPUCompileMlirOp.
Status MaybeRegisterFingerprint(
    Graph* graph,
    const std::map<std::string, std::vector<int>>& named_input_shapes,
    uint64 input_hash) {
  // Find the compiler metadata.
  tpu::TPUCompileMetadataProto metadata_proto;
  std::map<std::string, std::vector<int>> inputs_to_keep;
  int num_dynamic_shapes = -1;
  tensorflow::uint64 fingerprint = 0;

  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == "TPUCompile" ||
        node->type_string() == "_TPUCompileMlir") {
      num_dynamic_shapes = node->attrs().Find("NumDynamicShapes")->i();
      if (num_dynamic_shapes <= 0) {
        break;
      }
      int visited = 0;
      // TPUCompileOp/_TPUCompileMlirOp take Shape nodes as inputs.
      // The number of Shape nodes matches the number of dynamic shaped inputs.
      // The Shape nodes come from the input nodes:
      //   [TPU Input] --> [Input Shape] --> [TPUCompileOp]
      for (auto in_node : node->in_nodes()) {
        if (in_node->type_string() != "Shape") {
          continue;
        }
        for (auto input_node : in_node->in_nodes()) {
          auto iter = named_input_shapes.find(input_node->name());
          if (iter != named_input_shapes.end()) {
            inputs_to_keep[iter->first] = iter->second;
          }
        }
        visited++;
        if (visited == num_dynamic_shapes) {
          break;
        }
      }
      std::string metadata = node->attrs().Find("metadata")->s();
      metadata_proto.ParseFromString(metadata);

      if (node->type_string() == "_TPUCompileMlir") {
        std::string mlir_module = node->attrs().Find("mlir_module")->s();
        fingerprint = tensorflow::Fingerprint64(mlir_module);
      } else {
        fingerprint = metadata_proto.function_library_fingerprint();
      }

      break;
    }
  }
  VLOG(2) << "inputs_to_keep size: " << inputs_to_keep.size();
  if (inputs_to_keep.size() != num_dynamic_shapes) {
    VLOG(2) << "Cannot match all inputs shapes. Skip fingerprint registration.";
    return OkStatus();
  }

  std::vector<TensorShape> input_shapes;
  TF_RETURN_IF_ERROR(
      ConvertEdgeShapesToTensorShapes(inputs_to_keep, &input_shapes));

  std::vector<TensorShape> arg_shapes;
  auto status =
      tpu::ComputeArgumentShapes(metadata_proto, input_shapes, &arg_shapes);
  if (!status.ok()) {
    VLOG(2) << status.error_message();
    return OkStatus();
  }
  uint64 tf_fingerprint =
      tpu::CreateFingerprintWithNameAndShapes(fingerprint, arg_shapes);
  VLOG(2) << "fingerprint: " << fingerprint;
  VLOG(2) << "TF fingerprint: " << tf_fingerprint;

  ResourceMgr* rm = GetTPUConfigResourceMgr();
  tpu::TpuFingerprintLookup* fingerprint_lookup;
  TF_RETURN_IF_ERROR(rm->Lookup<tpu::TpuFingerprintLookup>(
      rm->default_container(), tpu::kFingerprintLookupResourceName,
      &fingerprint_lookup));
  fingerprint_lookup->RegisterKeyAndIntermediatePair(input_hash,
                                                     tf_fingerprint);
  return OkStatus();
}

bool FindTpuReplicatedInputAndXlaSharding(
    const Graph* graph, XlaShardingInfoMap& xla_sharding_ops,
    TpuReplicatedInputInfoMap& tpu_replicated_input_ops) {
  bool xla_spmd_input_sharded = false;
  // Detect whether there are XLA Sharding on the inputs, if there are, then
  // we cannot remove the replicated inputs or the xla sharding ops.
  for (Node* xla_sharding_node : graph->nodes()) {
    if (xla_sharding_node->type_string() == "XlaSharding") {
      for (const Edge* edge : xla_sharding_node->in_edges()) {
        if (edge->src()->type_string() == "TPUReplicatedInput") {
          Node* tpu_replicated_input_node = edge->src();
          Node* tpu_replicated_metadata_node = nullptr;
          for (const Edge* input_edge : tpu_replicated_input_node->in_edges()) {
            if (input_edge->IsControlEdge()) {
              tpu_replicated_metadata_node = input_edge->src();
            }
          }

          for (const Edge* input_edge : tpu_replicated_input_node->in_edges()) {
            if (input_edge->src()->type_string() == "_Arg") {
              Node* arg_node = input_edge->src();

              xla_sharding_ops[arg_node->name()] = std::make_tuple(
                  xla_sharding_node->attrs().Find("T")->type(),
                  xla_sharding_node->attrs().Find("sharding")->s(),
                  xla_sharding_node->attrs().Find("_tpu_replicate")->s());

              tpu_replicated_input_ops[arg_node->name()] = std::make_tuple(
                  tpu_replicated_input_node->attrs().Find("T")->type(),
                  tpu_replicated_metadata_node);

              VLOG(2) << "Detected input is sharded. XlaSharding node: "
                      << xla_sharding_node->DebugString()
                      << ", TPUReplicatedInput node: "
                      << edge->src()->DebugString()
                      << ", _Arg node: " << arg_node->DebugString();
              xla_spmd_input_sharded = true;
              break;
            }
          }
        }
      }
    }
  }
  return xla_spmd_input_sharded;
}

// Returns the name of the framework that rewrote the graph to support
// inference on TPUs. This name is accessed later during metric collection.
string GetProducerName(const string& function_name) {
  if (absl::StrContains(function_name, "tpu_func_0") ||
      absl::StrContains(function_name, "_with_batch") ||
      absl::StrContains(function_name, "_optim"))
    return "TPU_INFERENCE_CONVERTER";
  return "UNKNOWN";
}

// Gets the proper tensor dimension from XLA OpSharding.
// "replicate_on_last_tile_dim" and "last_tile_dims" should be deducted from the
// real Tensor dimensions when tiled.
// For example:
// f32[8,512](sharding={devices=[1,1,2]0,1 last_tile_dims={REPLICATED})
// also means a replicated tensor over all devices.
//
// See xla_data.proto for detailed explanations on the fields.
int GetDimsFromXLAShardingTiled(const xla::OpSharding xla_sharding) {
  return xla_sharding.tile_assignment_dimensions_size() -
         (xla_sharding.replicate_on_last_tile_dim() ? 1 : 0) -
         xla_sharding.last_tile_dims_size();
}

}  // end namespace

namespace tpu_functional_internal {

// An optimization pass that separates tensors to leverage the fast path in
// TPU input preparation. The algorithm is as follows:
// (1) Group all tensors that have same dimensions except the last dimension. A
// group of tensors will be concatenated by the last dimension in a later pass.
// (2) Check all groups of tensors and find groups whose dimensions after concat
// cannot leverage the fast path.
// (3) For groups of tensors that don't leverage the fast path, split tensors
// into two sub-groups such that one sub-group of tensors can leverage the fast
// path.
// Exception in (2) is that concated tensors are small, which means separating
// tensors would introduce overheads of data transfer to device.
// This optimization takes effect when both --input_shape_opt and
// --group_tensors_for_packing are true.
GroupedEdges GroupTensorsForInputPacking(
    const EdgeShapes& tpu_input_shapes,
    const absl::flat_hash_map<const Edge*, DataType>& tpu_input_dtypes,
    bool input_shape_opt, bool group_tensors_for_packing) {
  GroupedEdges grouped_input_edges;
  for (const auto& iter : tpu_input_shapes) {
    if (iter.second.empty()) continue;
    DataType dtype = tpu_input_dtypes.find(iter.first)->second;
    string hash_key = HashShapeAndType("input_tensors_dtype_", iter.second,
                                       dtype, /*input_shape_opt*/ false);
    grouped_input_edges[hash_key].push_back(iter.first);
  }
  // Apply grouping when both are true.
  if (!input_shape_opt || !group_tensors_for_packing)
    return grouped_input_edges;

  GroupedEdges grouped_input_edges_opt;
  for (const auto& iter : grouped_input_edges) {
    int sum_last_dim = 0;
    int product_other_dims = 0;
    VLOG(3) << "group name: " << iter.first;
    for (const auto& edge : iter.second) {
      const std::vector<int>& input_shapes =
          tpu_input_shapes.find(edge)->second;
      sum_last_dim += input_shapes.back();
      if (product_other_dims == 0) {
        product_other_dims = 1;
        for (int d = 0; d < input_shapes.size() - 1; ++d)
          product_other_dims *= input_shapes.at(d);
      }
    }
    VLOG(3) << "sum_last_dim: " << sum_last_dim;
    VLOG(3) << "product_other_dims: " << product_other_dims;
    // Already uses fast path, skip further grouping.
    if ((sum_last_dim % kLastDimOfTpuInputFastPath) == 0 &&
        (product_other_dims % kOtherDimOfTpuInputFastPath) == 0) {
      grouped_input_edges_opt[iter.first] = iter.second;
      continue;
    }
    // Tensors are small, skip further grouping.
    if ((sum_last_dim * product_other_dims) <
        (kLastDimOfTpuInputFastPath * kOtherDimOfTpuInputFastPath)) {
      grouped_input_edges_opt[iter.first] = iter.second;
      continue;
    }
    VLOG(3) << "Splitting tensors.";
    for (const auto& edge : iter.second) {
      auto tpu_input_shape = tpu_input_shapes.find(edge)->second;
      string hash_key =
          HashShapeAndType("input_tensors_dtype_", tpu_input_shape,
                           tpu_input_dtypes.find(edge)->second,
                           /*input_shape_opt*/ true);
      grouped_input_edges_opt[hash_key].push_back(edge);
    }
  }
  return grouped_input_edges_opt;
}

GroupedEdges GroupTensorsForOutputPacking(Graph* graph,
                                          EdgeShapes& tpu_output_shapes,
                                          GraphShapeInfo* shape_info) {
  GroupedEdges shape_to_output;
  for (const Edge* edge : graph->edges()) {
    if (edge->IsControlEdge()) continue;

    // TPU input edge: src is on TPU and dest is on CPU.
    if (edge->dst()->type_string() != "TPUReplicatedOutput") continue;
    if (!shape_info->count(edge->src()->name())) continue;

    // output shapes for hashing
    std::vector<int>& output_shapes = tpu_output_shapes[edge];
    output_shapes.clear();

    int output_id = edge->src_output();
    auto inferred_shape_vec = shape_info->at(edge->src()->name());

    for (int d : inferred_shape_vec.at(output_id).shape.dim_sizes()) {
      output_shapes.push_back(d);
    }

    // Hash Shape and Types.
    DataType dtype = edge->src()->input_type(output_id);
    string hash_key =
        HashShapeAndType("output_tensors_dtype_", output_shapes, dtype, false);

    shape_to_output[hash_key].push_back(edge);
  }
  return shape_to_output;
}

// Concatenates input tensors on CPU along the last dimension if all other
// dimensions are the same, and split them on TPU to reduce input overhead.
// `tpu_input_shapes` maps an edge to the shape of its output tensor.
// `grouped_input_edges` maps tensor name to all edges output from this tensor.
Status CreateConcatAndSplitNodesForInputTensor(
    Graph* graph, const string& cluster_name, EdgeShapes* tpu_input_shapes,
    const absl::flat_hash_map<std::string, std::vector<const Edge*>>&
        grouped_input_edges,
    int32_t minimum_input_tensors_packing, bool xla_spmd_input_sharded,
    const XlaShardingInfoMap& xla_sharding_info,
    const TpuReplicatedInputInfoMap& tpu_replicated_input_info) {
  for (const auto& iter : grouped_input_edges) {
    std::vector<int> last_dim_vec;
    std::vector<NodeBuilder::NodeOut> concat_nodeouts;
    absl::flat_hash_map<std::string, int> tensor_to_split_output;
    int rank;
    DataType dtype = DT_INVALID;
    std::string src_name;
    for (const Edge* edge : iter.second) {
      src_name = edge->src()->name();
      string tensor_name =
          absl::StrCat(edge->src()->name(), ":", edge->src_output());
      // Create Concat / Split pair for a tensor if not exist yet.
      if (tensor_to_split_output.contains(tensor_name)) continue;
      tensor_to_split_output[tensor_name] = concat_nodeouts.size();
      concat_nodeouts.push_back(
          NodeBuilder::NodeOut(edge->src(), edge->src_output()));
      dtype = edge->src()->output_type(edge->src_output());
      rank = tpu_input_shapes->at(edge).size();
      last_dim_vec.push_back(tpu_input_shapes->at(edge).back());
    }

    const int num_tensors = tensor_to_split_output.size();
    VLOG(3) << iter.first << " num_tensors: " << num_tensors;
    if (num_tensors < minimum_input_tensors_packing) {
      VLOG(3) << "skip concat/split " << iter.first;
      continue;
    }

    Node* concat_axis_node = nullptr;
    TensorShape t_shape;
    Tensor dim_tensor(DT_INT32, t_shape);
    // Concat and Split at the last dim.
    dim_tensor.flat<int>()(0) = rank - 1;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/concat/axis"), "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", dim_tensor)
            .Finalize(graph, &concat_axis_node));

    Node* concat_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/concat"), "ConcatV2")
            .Input(concat_nodeouts)
            .Input(concat_axis_node)
            .Attr("T", dtype)
            .Attr("Tidx", DT_INT32)
            .Attr("N", num_tensors)
            .Finalize(graph, &concat_node));

    Node* split_dim_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/split/split_dim"), "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", dim_tensor)
            .Attr(kTpuReplicateAttr, cluster_name)
            .Finalize(graph, &split_dim_node));

    Node* split_vec_node = nullptr;
    TensorShape split_vec_shape;
    split_vec_shape.AddDim(1);
    split_vec_shape.set_dim(0, last_dim_vec.size());

    Tensor split_vec_tensor(DT_INT32, split_vec_shape);
    for (int i = 0; i < last_dim_vec.size(); ++i) {
      split_vec_tensor.flat<int>()(i) = last_dim_vec[i];
    }
    VLOG(3) << "split_vec_tensor: " << split_vec_tensor.DebugString();

    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/split/vec"), "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", split_vec_tensor)
            .Attr(kTpuReplicateAttr, cluster_name)
            .Finalize(graph, &split_vec_node));

    Node* split_node = nullptr;
    Node* input_to_split_node = concat_node;
    Node* output_from_concat_node = nullptr;
    if (xla_spmd_input_sharded &&
        tpu_replicated_input_info.count(src_name) > 0 &&
        xla_sharding_info.count(src_name) > 0) {
      // Create new TPUReplicatedInput and XLAShardingOp nodes
      //
      // Rewrite the graph from:
      //   Concat -> Split
      // to
      //   Concat -> TPUReplicatedInput -> XlaSharding -> Split
      Node* tpu_replicated_input = nullptr;
      Node* xla_sharding_op = nullptr;

      std::vector<NodeBuilder::NodeOut> replicated_input;
      replicated_input.push_back(NodeBuilder::NodeOut(concat_node));

      // TODO(b/183060455): Add TPUReplicatedInput to all graphs.
      TF_RETURN_IF_ERROR(
          NodeBuilder(strings::StrCat(iter.first, "/TPUReplicatedInput"),
                      "TPUReplicatedInput")
              .Input(replicated_input)
              .ControlInput(std::get<1>(tpu_replicated_input_info.at(src_name)))
              .Attr("N", 1)
              .Attr("T", std::get<0>(tpu_replicated_input_info.at(src_name)))
              .Attr("index", -1)
              .Attr("is_mirrored_variable", false)
              .Attr("is_packed", false)
              .Finalize(graph, &tpu_replicated_input));
      VLOG(2) << "Created new TPUReplicatedInput node "
              << tpu_replicated_input->DebugString();

      TF_RETURN_IF_ERROR(
          NodeBuilder(strings::StrCat(iter.first, "/XlaSharding"),
                      "XlaSharding")
              .Input(tpu_replicated_input)
              .Attr("T", std::get<0>(xla_sharding_info.at(src_name)))
              .Attr("sharding", std::get<1>(xla_sharding_info.at(src_name)))
              .Attr("_XlaSharding", std::get<1>(xla_sharding_info.at(src_name)))
              .Attr("_tpu_replicate",
                    std::get<2>(xla_sharding_info.at(src_name)))
              .Finalize(graph, &xla_sharding_op));
      VLOG(2) << "Created new XLA sharding node "
              << xla_sharding_op->DebugString();

      input_to_split_node = xla_sharding_op;
      output_from_concat_node = tpu_replicated_input;
    }
    // Update the `tpu_input_shapes` mapping: Add the new edge
    // from concat to split.
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/split"), "SplitV")
            .Input(input_to_split_node)
            .Input(split_vec_node)
            .Input(split_dim_node)
            .Attr("T", dtype)
            .Attr("num_split", num_tensors)
            .Attr(kTpuReplicateAttr, cluster_name)
            .Finalize(graph, &split_node));

    if (output_from_concat_node == nullptr)
      output_from_concat_node = split_node;

    const Edge* concat_to_split;
    for (const Edge* edge : concat_node->out_edges())
      if (edge->dst() == output_from_concat_node) {
        concat_to_split = edge;
        break;
      }
    if (rank > 1) {
      for (int d = 0; d < rank - 1; ++d)
        (*tpu_input_shapes)[concat_to_split].push_back(
            tpu_input_shapes->at(iter.second.back()).at(d));
    }
    (*tpu_input_shapes)[concat_to_split].push_back(
        std::accumulate(last_dim_vec.begin(), last_dim_vec.end(), 0));

    // Connect split node to original tensor output.
    for (const Edge* edge : iter.second) {
      string tensor_name =
          absl::StrCat(edge->src()->name(), ":", edge->src_output());
      int output_index = tensor_to_split_output.at(tensor_name);
      graph->RemoveEdge(edge);
      graph->AddEdge(split_node, output_index, edge->dst(), edge->dst_input());
      // Update the `tpu_input_shapes` mapping: Remove old edges.
      tpu_input_shapes->erase(edge);
    }
    VLOG(3) << "Concat node: " << concat_node->DebugString();
  }
  return OkStatus();
}

// Concatenates input tensors on TPU along the last dimension if all other
// dimensions are the same, and split them on CPU to reduce outfeed overhead.
// `tpu_inferred_info` maps an edge to the inferred shape of its output tensor.
// `shape_to_output` maps tensor name to all edges output from this tensor.
Status CreateConcatAndSplitNodesForOutputTensor(
    Graph* graph, const string& cluster_name, EdgeShapes* tpu_output_shapes,
    GraphShapeInfo* tpu_inferred_info, GroupedEdges shape_to_output,
    int32_t minimum_output_tensors_packing) {
  for (const auto& iter : shape_to_output) {
    std::vector<int> last_dim_vec;
    std::vector<NodeBuilder::NodeOut> concat_nodeouts;
    absl::flat_hash_map<std::string, int> tensor_to_split_output;
    int rank;
    DataType dtype = DT_INVALID;
    for (const Edge* edge : iter.second) {
      string tensor_name =
          absl::StrCat(edge->src()->name(), ":", edge->src_output());

      // Create Concat / Split pair for a tensor if not exist yet.
      if (tensor_to_split_output.contains(tensor_name)) continue;
      tensor_to_split_output[tensor_name] = concat_nodeouts.size();

      concat_nodeouts.push_back(
          NodeBuilder::NodeOut(edge->src(), edge->src_output()));
      dtype = edge->src()->output_type(edge->src_output());
      rank = tpu_output_shapes->at(edge).size();
      last_dim_vec.push_back(tpu_output_shapes->at(edge).back());
    }

    const int num_tensors = tensor_to_split_output.size();
    if (num_tensors < minimum_output_tensors_packing) {
      VLOG(3) << "skip concat/split " << iter.first;
      continue;
    }

    Node* concat_axis_node = nullptr;
    TensorShape t_shape;
    Tensor dim_tensor(DT_INT32, t_shape);
    // Concat and Split at the last dim.
    dim_tensor.flat<int>()(0) = rank - 1;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/concat/axis"), "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", dim_tensor)
            .Attr(kTpuReplicateAttr, cluster_name)
            .Finalize(graph, &concat_axis_node));

    Node* concat_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/concat"), "ConcatV2")
            .Input(concat_nodeouts)
            .Input(concat_axis_node)
            .Attr("T", dtype)
            .Attr("Tidx", DT_INT32)
            .Attr("N", num_tensors)
            .Attr(kTpuReplicateAttr, cluster_name)
            .Finalize(graph, &concat_node));

    Node* tpu_replicated_output_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/tpu_replicated_output"),
                    "TPUReplicatedOutput")
            .Input(concat_node)
            .Attr("T", dtype)
            .Attr("num_replicas", 1)
            .Finalize(graph, &tpu_replicated_output_node));

    Node* split_dim_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/split/split_dim"), "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", dim_tensor)
            .Finalize(graph, &split_dim_node));

    Node* split_vec_node = nullptr;
    TensorShape split_vec_shape;
    split_vec_shape.AddDim(1);
    split_vec_shape.set_dim(0, last_dim_vec.size());

    Tensor split_vec_tensor(DT_INT32, split_vec_shape);
    for (int i = 0; i < last_dim_vec.size(); ++i) {
      split_vec_tensor.flat<int>()(i) = last_dim_vec[i];
    }
    VLOG(3) << "split_vec_tensor: " << split_vec_tensor.DebugString();

    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/split/vec"), "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", split_vec_tensor)
            .Finalize(graph, &split_vec_node));

    Node* split_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(iter.first, "/split"), "SplitV")
            .Input(tpu_replicated_output_node)
            .Input(split_vec_node)
            .Input(split_dim_node)
            .Attr("T", dtype)
            .Attr("num_split", num_tensors)
            .Finalize(graph, &split_node));

    // Update the `tpu_out_shapes` mapping: Add the new edge
    // from concat to split.
    const Edge* concat_to_split;
    for (const Edge* edge : concat_node->out_edges())
      if (edge->dst() == split_node) {
        concat_to_split = edge;
        break;
      }

    if (rank > 1) (*tpu_output_shapes)[concat_to_split].push_back(-1);
    for (int d = 1; d < rank - 1; ++d)
      (*tpu_output_shapes)[concat_to_split].push_back(
          tpu_output_shapes->at(iter.second.back()).at(d));
    (*tpu_output_shapes)[concat_to_split].push_back(
        std::accumulate(last_dim_vec.begin(), last_dim_vec.end(), 0));

    for (const Edge* edge : iter.second) {
      // 1. Find old TPURelicatedOutput output edges
      std::vector<const Edge*> output_edge_vec;
      for (const Edge* output_edge : edge->dst()->out_edges())
        output_edge_vec.push_back(output_edge);

      string tensor_name =
          absl::StrCat(edge->src()->name(), ":", edge->src_output());
      int output_index = tensor_to_split_output.at(tensor_name);
      VLOG(3) << "output_index: " << output_index;

      // Connect split node to original tensor output.
      for (const Edge* output_edge : output_edge_vec) {
        VLOG(3) << "output_edge" << output_edge->DebugString();
        graph->RemoveEdge(output_edge);
        graph->AddEdge(split_node, output_index, output_edge->dst(),
                       output_edge->dst_input());
        // Update the `tpu_output_shapes` mapping: Remove old edges.
        tpu_output_shapes->erase(output_edge);
      }
      graph->RemoveNode(edge->dst());
    }
    VLOG(3) << "Concat node: " << concat_node->DebugString();
  }
  return OkStatus();
}

Status InsertReshapeNodePairs(Graph* graph, const string& cluster_name,
                              EdgeShapes* tpu_input_shapes,
                              int num_cores_per_replica) {
  std::vector<const Edge*> tpu_input_edges_original;
  for (const auto& it : *tpu_input_shapes)
    if (!it.second.empty()) tpu_input_edges_original.push_back(it.first);
  for (const Edge* edge : tpu_input_edges_original) {
    VLOG(3) << "Reshape input: " << edge->DebugString();

    // Check if there is a TPUReplicatedInput and XlaSharding in the middle
    bool xla_sharded_input = false;
    Node* xla_sharding_node = nullptr;
    if (edge->dst()->type_string() == "TPUReplicatedInput" &&
        edge->dst()->out_nodes().begin()->type_string() == "XlaSharding") {
      VLOG(3) << "Detected TPUReplicatedInput " << edge->dst()->DebugString()
              << " and XlaSharding "
              << edge->dst()->out_nodes().begin()->DebugString()
              << ", setting xla_sharded_input = true";
      xla_sharded_input = true;
      xla_sharding_node = *(edge->dst()->out_nodes().begin());
    }

    // 1. Build Reshape node for flatten.

    // 1.1 Build Const node for shape
    Node* flatten_reshape_shape_node = nullptr;
    Tensor flattened_input_shape_tensor;
    flattened_input_shape_tensor =
        Tensor(DT_INT32, TensorShape({static_cast<int64_t>(1)}));
    flattened_input_shape_tensor.flat<int>()(0) = -1;
    TF_RETURN_IF_ERROR(
        NodeBuilder(absl::StrCat(edge->src()->name(), "/flatten/Reshape/shape"),
                    "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", flattened_input_shape_tensor)
            .Finalize(graph, &flatten_reshape_shape_node));

    // 1.2 Build Reshape node for flatten.
    Node* flatten_reshape_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(absl::StrCat(edge->src()->name(), "/flatten/Reshape"),
                    "Reshape")
            .Input(edge->src(), edge->src_output())
            .Input(flatten_reshape_shape_node)
            .Attr("T", edge->src()->output_type(edge->src_output()))
            .Attr("Tshape", DT_INT32)
            .Finalize(graph, &flatten_reshape_node));

    // 2. Build Reshape node for recover.

    // 2.1 Build Const node for shape.
    Node* recover_reshape_shape_node = nullptr;
    Tensor original_input_shape_tensor(
        DT_INT32,
        TensorShape({static_cast<int64_t>(tpu_input_shapes->at(edge).size())}));
    original_input_shape_tensor.flat<int>()(0) = -1;
    for (int d = 1; d < tpu_input_shapes->at(edge).size(); ++d)
      original_input_shape_tensor.flat<int>()(d) =
          tpu_input_shapes->at(edge).at(d);
    TF_RETURN_IF_ERROR(
        NodeBuilder(absl::StrCat(edge->src()->name(), "/recover/Reshape/shape"),
                    "Const")
            .Attr("dtype", DT_INT32)
            .Attr("value", original_input_shape_tensor)
            .Attr(kTpuReplicateAttr, cluster_name)  // This node is on TPU.
            .Finalize(graph, &recover_reshape_shape_node));

    // 2.2 Build Reshape node for recover.
    Node* recover_reshape_input_node = flatten_reshape_node;
    const Edge* original_recover_reshape_input_edge = nullptr;
    if (xla_sharded_input) {
      // We want to find the node after the XlaSharding node
      original_recover_reshape_input_edge =
          *(edge->dst()->out_nodes().begin()->out_edges().begin());
      recover_reshape_input_node = *(edge->dst()->out_nodes().begin());
      VLOG(3) << "Recover reshape input node: "
              << recover_reshape_input_node->DebugString()
              << ", recover reshape input edge: "
              << original_recover_reshape_input_edge->DebugString();
    }

    Node* recover_reshape_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(absl::StrCat(edge->src()->name(), "/recover/Reshape"),
                    "Reshape")
            .Input(recover_reshape_input_node)
            .Input(recover_reshape_shape_node)
            .Attr("T", edge->src()->output_type(edge->src_output()))
            .Attr("Tshape", DT_INT32)
            .Attr(kTpuReplicateAttr, cluster_name)  // This node is on TPU.
            .Finalize(graph, &recover_reshape_node));

    // 3. Rewrite XlaSharding attribute if necessary
    if (xla_sharding_node != nullptr) {
      // The flattened tensor always has rank = 1 and we want to shard the only
      // dimension (0).
      SetXlaShardingNodeAttr(xla_sharding_node, num_cores_per_replica, 1, 0);
    }

    // 4. Connect / disconnect nodes.
    if (xla_sharded_input) {
      graph->AddEdge(flatten_reshape_node, 0, edge->dst(), edge->dst_input());
    }

    if (original_recover_reshape_input_edge != nullptr) {
      graph->AddEdge(recover_reshape_node, 0,
                     original_recover_reshape_input_edge->dst(),
                     original_recover_reshape_input_edge->dst_input());
    } else {
      graph->AddEdge(recover_reshape_node, 0, edge->dst(), edge->dst_input());
    }

    graph->RemoveEdge(edge);
    if (original_recover_reshape_input_edge != nullptr) {
      graph->RemoveEdge(original_recover_reshape_input_edge);
    }

    // 4. Update EdgeShapes.
    int dimension = 1;
    for (auto& it : (*tpu_input_shapes)[edge]) {
      dimension *= it;
    }
    VLOG(3) << "Dimension after reshape: " << dimension;
    for (const Edge* out_edge : flatten_reshape_node->out_edges()) {
      if (out_edge->dst() == recover_reshape_node) {
        (*tpu_input_shapes)[out_edge].push_back(dimension);
        tpu_input_shapes->erase(edge);
        break;
      }
    }
    VLOG(3) << "Reshape optimization done for " << edge->src()->name();
  }
  return OkStatus();
}
}  // namespace tpu_functional_internal

void TPUPartitionedCallOp::ComputeAsync(OpKernelContext* ctx,
                                        DoneCallback done) {
  Status init_status;
  absl::call_once(once_, [&]() {
    library_runtime_ = ctx->function_library();
    if (library_runtime_ == nullptr) {
      init_status = errors::Internal("No function library is provided.");
      return;
    }
    flib_def_ = std::make_unique<FunctionLibraryDefinition>(
        *library_runtime_->GetFunctionLibraryDefinition());
    device_mgr_ = library_runtime_->device_mgr();
    for (auto d : device_mgr_->ListDevices()) {
      device_set_.AddDevice(d);
    }

    DeviceNameUtils::ParsedName tpu_device_name;
    tpu_device_name.has_type = true;
    tpu_device_name.type = "TPU";
    std::vector<Device*> tpu_devices;
    device_set_.FindMatchingDevices(tpu_device_name, &tpu_devices_);
  });
  OP_REQUIRES_OK_ASYNC(ctx, init_status, done);

  // Initialize the ordinal selector with information from the graph if it is
  // the first time we are running this op.
  absl::call_once(ordinal_selector_once_, [&]() {
    std::unique_ptr<Graph> graph(new Graph(flib_def_.get()));
    int num_cores_per_replica = 1;
    bool enable_spmd_xla_partitioning = false;
    {
      absl::MutexLock l(&mu_);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          GetGraphFromFunction(graph.get(), /*device_ordinal=*/0,
                               &num_cores_per_replica,
                               &enable_spmd_xla_partitioning),
          done);
    }
    if (enable_spmd_xla_partitioning) {
      ordinal_selector_ =
          std::make_shared<tpu::TPUOrdinalSelector>(num_cores_per_replica);
    } else {
      ordinal_selector_ = std::make_shared<tpu::TPUOrdinalSelector>();
    }

    metrics::RecordTPUXlaSpmdCoresPerReplica(num_cores_per_replica);
  });
  OP_REQUIRES_ASYNC(
      ctx, ordinal_selector_ != nullptr,
      errors::Internal("The TPUOrdinalSelector is not initialized."), done);

  uint64 input_hash = GetInputHash(ctx);
  int64_t ordinal_selector_req_id = -1;
  // Select a TPU core.
  int32_t device_ordinal = 0;
  OP_REQUIRES_OK_ASYNC(
      ctx,
      GetTpuCoreOrdinal(ctx, input_hash, &ordinal_selector_req_id,
                        &device_ordinal),
      done);
  uint64 cache_hash = Hash64Combine(input_hash, device_ordinal);
  absl::ReleasableMutexLock lock(&mu_);

  const std::vector<DeviceAndFHandle>* functions;

  bool cache_miss = !partition_cache_.count(cache_hash);
  if (cache_miss) {
    VLOG(3) << "Cache Miss: partitioning function " << func_.name()
            << " cache_hash: " << cache_hash
            << " device_ordinal: " << device_ordinal;

    profiler::TraceMe trace_me(
        "TPUPartitionedCallOp-RewriteAndInstantiateFunctions");
    std::unique_ptr<Graph> graph(new Graph(flib_def_.get()));
    int num_cores_per_replica = 1;
    bool enable_spmd_xla_partitioning = false;
    OP_REQUIRES_OK_ASYNC(ctx,
                         GetGraphFromFunction(graph.get(), device_ordinal,
                                              &num_cores_per_replica,
                                              &enable_spmd_xla_partitioning),
                         done);

    VLOG(1) << DumpGraphToFile("before_input_output_optimizations", *graph,
                               flib_def_.get());

    std::map<std::string, std::vector<int>> named_input_shapes;
    OP_REQUIRES_OK_ASYNC(ctx,
                         OptimizeTpuInputOutputTensors(
                             graph.get(), enable_spmd_xla_partitioning,
                             num_cores_per_replica, named_input_shapes, ctx),
                         done);

    VLOG(1) << DumpGraphToFile(
        "before_replace_resource_args_with_var_handle_ops", *graph,
        flib_def_.get());
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ReplaceResourceArgsWithVarHandleOps(graph.get(), ctx, device_ordinal,
                                            num_cores_per_replica,
                                            enable_spmd_xla_partitioning),
        done);

    VLOG(1) << DumpGraphToFile(
        "after_replace_resource_args_with_var_handle_ops", *graph,
        flib_def_.get());

    // Graph rewrite passes.
    GraphOptimizationPassOptions optimization_options;
    // TODO(akshayka): Thread the SessionOptions into this kernel, or make
    // it possible to specify the relevant options via attributes.
    SessionOptions session_options;
    session_options.config.mutable_experimental()
        ->set_xla_fusion_autotuner_thresh(autotuner_thresh_);

    session_options.env = ctx->env();
    optimization_options.session_handle = ctx->session_handle();
    optimization_options.session_options = &session_options;
    optimization_options.graph = &graph;
    optimization_options.flib_def = flib_def_.get();
    optimization_options.device_set = &device_set_;
    OP_REQUIRES_OK_ASYNC(
        ctx, PlacementHelper(device_set_, optimization_options, func_.name()),
        done);

    if (!enable_spmd_xla_partitioning || num_cores_per_replica == 1) {
      OP_REQUIRES_OK_ASYNC(
          ctx,
          MaybeRegisterFingerprint(graph.get(), named_input_shapes, input_hash),
          done);
    }
    // `subgraphs` maps from device names to functions.
    std::unordered_map<std::string, std::unique_ptr<Graph>> subgraphs;
    optimization_options.graph = nullptr;
    optimization_options.device_set = nullptr;
    optimization_options.partition_graphs = &subgraphs;
    VLOG(1) << DumpGraphToFile("before_partition_helper.pbtxt", *graph,
                               flib_def_.get());
    OP_REQUIRES_OK_ASYNC(ctx,
                         PartitionHelper(device_set_, optimization_options,
                                         graph.get(), &subgraphs),
                         done);
    OP_REQUIRES_OK_ASYNC(ctx,
                         InstantiateFunctionsFromSubgraphs(
                             device_set_, device_ordinal, cache_hash,
                             num_cores_per_replica, std::move(subgraphs)),
                         done);
  }
  functions = &partition_cache_[cache_hash];
  lock.Release();

  ExecuteFunctions(*functions, ctx, device_ordinal, ordinal_selector_req_id,
                   std::move(done));
}

Status TPUPartitionedCallOp::GetTpuCoreOrdinal(OpKernelContext* ctx,
                                               uint64 input_hash,
                                               int64_t* ordinal_selector_req_id,
                                               int32_t* core_ordinal) {
  profiler::TraceMe trace_me("TPUPartitionedCallOp-GetTpuCoreOrdinal");
  const Tensor* device_ordinal_t;
  TF_RETURN_IF_ERROR(ctx->input(kDeviceOrdinalAttr, &device_ordinal_t));
  int device_ordinal = device_ordinal_t->scalar<int>()();
  if (device_ordinal == tpu::kDeferredCoreSelectionReserved) {
    device_ordinal =
        ordinal_selector_->GetOrdinal(input_hash, ordinal_selector_req_id);
  }
  *core_ordinal = device_ordinal;
  return OkStatus();
}

Status TPUPartitionedCallOp::InitializeVarOnTPU(
    OpKernelContext* ctx, const core::RefCountPtr<Var>& var, NodeDef* ndef,
    int device_ordinal, bool fast_mem) {
  const string device = strings::StrCat(kTPUDeviceNamePrefix, device_ordinal);
  Status status;
  std::unique_ptr<Graph> init_graph(new Graph(OpRegistry::Global()));
  TF_ASSIGN_OR_RETURN(Node * init_handle, init_graph->AddNode(*ndef));
  init_handle->set_assigned_device_name(device);

  NodeDef init_const_ndef;
  init_const_ndef.set_name("initial_value");
#if defined(LIBTPU_ON_GCE)  // TODO(b/217559071) - Remove once _TPUConst is OSS
  init_const_ndef.set_op("Const");
#else
  init_const_ndef.set_op("_TPUConst");
  AddNodeAttr("memory_space", "HBM", &init_const_ndef);
#endif
  init_const_ndef.set_device(device);
  AddNodeAttr("dtype", var->tensor()->dtype(), &init_const_ndef);
  AddNodeAttr("value", *var->tensor(), &init_const_ndef);

  TF_ASSIGN_OR_RETURN(Node * init_const, init_graph->AddNode(init_const_ndef));

  NodeDef assign_node_def;
  assign_node_def.set_name("Assign");
  assign_node_def.set_op("AssignVariableOp");
  assign_node_def.set_device(device);
  AddNodeAttr("dtype", var->tensor()->dtype(), &assign_node_def);
  TF_ASSIGN_OR_RETURN(Node * init_assign, init_graph->AddNode(assign_node_def));

  init_graph->AddEdge(init_handle, 0, init_assign, 0);
  init_graph->AddEdge(init_const, 0, init_assign, 1);
  FHandle fhandle;
  const string fname =
      strings::StrCat(ndef->name(), "_init_ord_", device_ordinal);

  TF_RETURN_IF_ERROR(
      InstantiatePartition(*init_graph, fname, device, &fhandle, nullptr));

  FunctionLibraryRuntime::Options opts;
  opts.step_container = ctx->step_container();
  opts.cancellation_manager = ctx->cancellation_manager();
  opts.stats_collector = ctx->stats_collector();

  // Blocking on threads in the same thread pool is disallowed because
  // concurrent warm-up requests can exhaust the default thread pool.
  // Create a new thread pool to initialize variables on TPU.
  std::function<void(std::function<void()>)> runner =
      [this](std::function<void()> fn) { pool_.Schedule(fn); };
  opts.runner = &runner;

  opts.source_device = local_device_name_;
  PrivateIntraProcessRendezvous rendez(device_mgr_);
  opts.rendezvous = &rendez;
  opts.remote_execution = true;

  std::vector<Tensor> dummy_args;
  std::vector<Tensor>* dummy_rets = new std::vector<Tensor>;
  Notification done;
  profiler::TraceMe trace_me("TPUPartitionedCallOp-InitializeVarOnTPU");
  library_runtime_->Run(opts, fhandle, dummy_args, dummy_rets,
                        [dummy_rets, &done, ctx](const Status& status) {
                          if (!status.ok()) {
                            ctx->SetStatus(status);
                          }
                          delete dummy_rets;
                          done.Notify();
                        });
  done.WaitForNotification();
  // We don't actually want the variable initialization functions
  // in the function library definition and the function library
  // runtime, because flib_def_ is used for the graph rewrite passes.
  // The TPU distributed rewrite pass computes a fingerprint for
  // flib_def_, which will throw an length error if there are
  // many variables whose initialization functions are added
  // to the library definition.
  TF_RETURN_IF_ERROR(flib_def_->RemoveFunction(fname));
  TF_RETURN_IF_ERROR(library_runtime_->ReleaseHandle(fhandle));
  return OkStatus();
}

Status TPUPartitionedCallOp::InitializeShardedVarOnTPU(
    OpKernelContext* ctx, const core::RefCountPtr<Var>& var,
    std::vector<NodeDef>& ndefs, int split_dim, int device_ordinal) {
  std::unique_ptr<Graph> init_graph(new Graph(OpRegistry::Global()));
  int num_cores = ndefs.size();
  string cpu_device = "/device:CPU:0";

  Status status;
  std::vector<std::string> devices;
  std::vector<Node*> init_handles;
  for (int i = 0; i < num_cores; i++) {
    TF_ASSIGN_OR_RETURN(Node * init_handle, init_graph->AddNode(ndefs[i]));
    string device = strings::StrCat(kTPUDeviceNamePrefix, device_ordinal + i);
    init_handle->set_assigned_device_name(device);
    devices.push_back(device);
    init_handles.push_back(init_handle);
  }

  NodeDef init_const_ndef;
  init_const_ndef.set_name("initial_value");
  init_const_ndef.set_op("Const");
  init_const_ndef.set_device(cpu_device);
  AddNodeAttr("dtype", var->tensor()->dtype(), &init_const_ndef);
  AddNodeAttr("value", *var->tensor(), &init_const_ndef);
  TF_ASSIGN_OR_RETURN(Node * init_const, init_graph->AddNode(init_const_ndef));
  init_const->set_assigned_device_name(cpu_device);

  Node* assign_value_node = init_const;
  // If the variable is sharded, we will insert "Split" node between the initial
  // value and AssignVariableOp, so the variables on each TPU device get
  // assigned to the splitted value.
  //
  // initial_value--Split--AssignVariableOp ("/device:TPU:0")
  //                  |
  //            AssignVariableOp ("/device:TPU:1")
  if (split_dim >= 0) {
    // Add a split dimension node.
    NodeDef split_dim_def;
    split_dim_def.set_name("initial_value_split_dim");
    split_dim_def.set_op("Const");
    split_dim_def.set_device(cpu_device);
    AddNodeAttr("dtype", DT_INT32, &split_dim_def);
    TensorProto tensor_proto;
    tensor_proto.set_dtype(DT_INT32);
    tensor_proto.add_int_val(split_dim);
    TensorShape shape({});
    shape.AsProto(tensor_proto.mutable_tensor_shape());
    AddNodeAttr("value", tensor_proto, &split_dim_def);
    TF_ASSIGN_OR_RETURN(Node * split_dim_node,
                        init_graph->AddNode(split_dim_def));
    split_dim_node->set_assigned_device_name(cpu_device);

    // Add a split node.
    NodeDef split_def;
    int split_num = ndefs.size();
    split_def.set_name("initial_value_split");
    split_def.set_op("Split");
    split_def.set_device(cpu_device);
    AddNodeAttr("num_split", split_num, &split_def);
    AddNodeAttr("T", var->tensor()->dtype(), &split_def);
    split_def.add_input(absl::StrCat(split_dim_node->name(), ":0"));
    split_def.add_input(absl::StrCat(init_const->name(), ":0"));
    TF_ASSIGN_OR_RETURN(Node * split_node, init_graph->AddNode(split_def));
    split_node->set_assigned_device_name(cpu_device);

    init_graph->AddEdge(split_dim_node, 0, split_node, 0);
    init_graph->AddEdge(init_const, 0, split_node, 1);

    assign_value_node = split_node;
  }

  for (int i = 0; i < num_cores; i++) {
    NodeDef assign_node_def;
    assign_node_def.set_name(absl::StrCat("Assign_", i));
    assign_node_def.set_op("AssignVariableOp");
    assign_node_def.set_device(devices[i]);
    AddNodeAttr("dtype", var->tensor()->dtype(), &assign_node_def);
    TF_ASSIGN_OR_RETURN(Node * init_assign,
                        init_graph->AddNode(assign_node_def));
    init_assign->set_assigned_device_name(devices[i]);

    init_graph->AddEdge(init_handles[i], 0, init_assign, 0);
    if (split_dim >= 0) {
      init_graph->AddEdge(assign_value_node, i, init_assign, 1);
    } else {
      init_graph->AddEdge(assign_value_node, 0, init_assign, 1);
    }
  }

  GraphOptimizationPassOptions optimization_options;
  SessionOptions session_options;
  session_options.env = ctx->env();
  optimization_options.session_handle = ctx->session_handle();
  optimization_options.session_options = &session_options;
  optimization_options.flib_def = flib_def_.get();
  optimization_options.graph = nullptr;
  optimization_options.device_set = nullptr;
  std::unordered_map<std::string, std::unique_ptr<Graph>> subgraphs;
  optimization_options.partition_graphs = &subgraphs;
  TF_RETURN_IF_ERROR(PartitionHelper(device_set_, optimization_options,
                                     init_graph.get(), &subgraphs));

  std::vector<DeviceAndFHandle> functions;
  std::vector<std::string> function_names;
  for (auto& pair : subgraphs) {
    string target = pair.first;
    Device* device;
    TF_RETURN_IF_ERROR(
        library_runtime_->device_mgr()->LookupDevice(target, &device));
    Graph* subgraph = pair.second.get();
    string function_name = flib_def_->UniqueFunctionName(
        strings::StrCat(func_.name(), "_hash_", pair.first));
    function_names.push_back(function_name);
    FHandle handle;
    TF_RETURN_IF_ERROR(InstantiatePartition(*subgraph, function_name, target,
                                            &handle, nullptr));
    functions.push_back(DeviceAndFHandle{.device = target, .handle = handle});
  }

  FunctionLibraryRuntime::Options opts;

  // Blocking on threads in the same thread pool is disallowed because
  // concurrent warm-up requests can exhaust the default thread pool.
  // Create a new thread pool to initialize variables on TPU.
  std::function<void(std::function<void()>)> runner =
      [this](std::function<void()> fn) { pool_.Schedule(fn); };
  opts.runner = &runner;

  opts.step_container = ctx->step_container();
  opts.cancellation_manager = ctx->cancellation_manager();
  opts.stats_collector = ctx->stats_collector();
  opts.source_device = local_device_name_;
  opts.run_all_kernels_inline = ctx->run_all_kernels_inline();

  OpInputList arguments;
  TF_RETURN_IF_ERROR(ctx->input_list("args", &arguments));

  PrivateIntraProcessRendezvous rendez(device_mgr_);
  opts.rendezvous = &rendez;

  BlockingCounter bcount(functions.size());
  for (const DeviceAndFHandle& entry : functions) {
    const string& target_device = entry.device;
    FHandle handle = entry.handle;

    TF_RETURN_IF_ERROR(
        ShouldUseRemoteExecutionForFn(target_device, &(opts.remote_execution)));
    std::vector<Tensor> dummy_args;
    std::vector<Tensor>* dummy_rets = new std::vector<Tensor>;

    profiler::TraceMe trace_me(
        "TPUPartitionedCallOp-InitializeShardedVarOnTPU");
    library_runtime_->Run(opts, handle, dummy_args, dummy_rets,
                          [dummy_rets, &bcount, ctx](const Status& status) {
                            if (!status.ok()) {
                              ctx->SetStatus(status);
                            }
                            delete dummy_rets;
                            bcount.DecrementCount();
                          });
  }
  bcount.Wait();

  for (int i = 0; i < functions.size(); i++) {
    TF_RETURN_IF_ERROR(flib_def_->RemoveFunction(function_names[i]));
    TF_RETURN_IF_ERROR(library_runtime_->ReleaseHandle(functions[i].handle));
  }
  return OkStatus();
}

bool TPUPartitionedCallOp::IsInputToTPUReplicate(Node* node) {
  for (Node* successor : node->out_nodes()) {
    if (successor->attrs().Find(kTpuReplicateAttr) != nullptr) {
      return true;
    }
  }
  return false;
}

Status TPUPartitionedCallOp::ReplaceResourceArgsWithVarHandleOps(
    Graph* graph, OpKernelContext* ctx, int device_ordinal,
    int num_cores_per_replica, bool enable_spmd_xla_partitioning) {
  // Currently variable deduplication is not supported for XLA SPMD
  // partitioning. It is possible that it could be supported in the future.
  bool enable_variable_deduplication =
      runtime_params_.enable_variable_deduplication;
  if (enable_spmd_xla_partitioning && num_cores_per_replica > 1) {
    // If enable_spmd_xla_partitioning is true, the user set the
    // enable_auto_xla_input_sharding flag. Warn them that only one of the flags
    // can be set safely when num_cores_per_replica > 1. If
    // num_cores_per_replica==1, enable_spmd_xla_partitioning is effectively a
    // no-op so we can skip this check.
    LOG(WARNING) << "Disabling variable deduplication because it is not "
                    "compatible with enable_auto_xla_input_sharding.";
    enable_variable_deduplication = false;
  }
  std::vector<Node*> tpu_resource_args;
  std::vector<int> arg_indices;
  absl::flat_hash_map<const Node*, xla::OpSharding> variable_to_xla_sharding;
  for (Node* node : graph->op_nodes()) {
    if (node->IsArg()) {
      const AttrValue* attr_value;
      TF_RETURN_IF_ERROR(node->attrs().Find("T", &attr_value));
      DataType dtype = attr_value->type();
      if (dtype == DT_RESOURCE && IsInputToTPUReplicate(node)) {
        // If this VarHandleOp is used by a TPU computation,
        // we need to create a TPU version of the variable,
        TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
        int index = attr_value->i();
        tpu_resource_args.push_back(node);
        arg_indices.push_back(index);
        replaced_input_indices_[index] = true;
      }
    }
  }

  VLOG(3) << "tpu_resource_args.size(): " << tpu_resource_args.size();
  // Create a mapping from ResourceHandle to variable node. When a
  // ResourceHandle backs several variable nodes, the variable nodes refer to
  // the same underlying resource. In that case, only one variable node needs
  // to be mirrored to the TPU for that resource.
  absl::flat_hash_map<uint64, Node*> tpu_variables;
  for (int i = 0; i < tpu_resource_args.size(); i++) {
    Node* node = tpu_resource_args[i];
    ResourceHandle handle = HandleFromInput(ctx, arg_indices[i]);

    if (num_cores_per_replica > 1 && enable_spmd_xla_partitioning) {
      TF_RETURN_IF_ERROR(ReplaceAndPartitionXLAShardingVariable(
          graph, ctx, device_ordinal, handle, node, num_cores_per_replica));
      continue;
    }
    TPUVariableInfo var_info(/*device_ordinal_id=*/0, /*use_fast_mem=*/false);
    TF_RETURN_IF_ERROR(
        ParseTPUVariableInfor(node, num_cores_per_replica, &var_info));
    // Only respect graph's placement when model parallelism enabled.
    if (num_cores_per_replica > 1) device_ordinal = var_info.device_ordinal;

    const uint64 handle_fp =
        Fingerprint64(strings::StrCat(handle.container(), handle.name()));
    if (enable_variable_deduplication && tpu_variables.contains(handle_fp) &&
        num_cores_per_replica == 1) {
      Node* tpu_variable = tpu_variables.at(handle_fp);
      std::vector<Node*> dst_nodes;
      std::vector<int> src_indices;
      std::vector<int> dst_indices;
      for (const Edge* edge : node->out_edges()) {
        dst_nodes.push_back(edge->dst());
        src_indices.push_back(edge->src_output());
        dst_indices.push_back(edge->dst_input());
      }
      graph->RemoveNode(node);
      for (int i = 0; i < dst_nodes.size(); i++) {
        graph->AddEdge(tpu_variable, src_indices[i], dst_nodes[i],
                       dst_indices[i]);
      }
    } else {
      uint64 fp =
          Fingerprint64(strings::StrCat(handle.container(), handle.name(), i));
      NodeDef ndef;
      ndef.set_name(strings::StrCat(handle.name(), fp));
      ndef.set_op(kVarHandleOp);
      if (num_cores_per_replica > 1) {
        ndef.set_device(strings::StrCat(kTPUDeviceNamePrefix, device_ordinal));
      } else {
        // Assign this new VarHandleOp to TPU:0 so the partitioner only
        // partiitons the graph into two subgraphs, one on CPU and one on TPU.
        // The actual device ordinal on which this VarHandleOp runs is assigned
        // after partitioning (in SetDeviceOrdinal).
        ndef.set_device(
            strings::StrCat(kTPUDeviceNamePrefix, kTPUDefaultDeviceOrdinal));
      }

      // Replace each _Arg node of type DT_RESOURCE that goes into a TPU node
      // by a VarHandleOp on TPU with shared_name "v_tpu_x" where "v" is the
      // shared_name of the variable on CPU and "x" is the rewritten device
      // ordinal.
      const string sname =
          strings::StrCat(handle.name(), "_tpu_", device_ordinal);
      AddNodeAttr("shared_name", sname, &ndef);
      const string cname = ctx->resource_manager()->default_container();
      AddNodeAttr("container", cname, &ndef);
      core::RefCountPtr<Var> var;
      TF_RETURN_IF_ERROR(LookupResource(ctx, handle, &var));
      AddNodeAttr("dtype", var->tensor()->dtype(), &ndef);
      TensorShapeProto proto;
      var->tensor()->shape().AsProto(&proto);
      AddNodeAttr("shape", proto, &ndef);
      TF_ASSIGN_OR_RETURN(Node * new_node, graph->AddNode(ndef));
      std::vector<const Edge*> in_edges(node->in_edges().begin(),
                                        node->in_edges().end());
      for (const Edge* edge : in_edges) {
        graph->AddEdge(edge->src(), edge->src_output(), new_node,
                       edge->dst_input());
      }
      std::vector<Node*> dst_nodes;
      std::vector<int> src_indices;
      std::vector<int> dst_indices;
      for (const Edge* edge : node->out_edges()) {
        dst_nodes.push_back(edge->dst());
        src_indices.push_back(edge->src_output());
        dst_indices.push_back(edge->dst_input());
      }
      graph->RemoveNode(node);
      for (int i = 0; i < dst_nodes.size(); i++) {
        graph->AddEdge(new_node, src_indices[i], dst_nodes[i], dst_indices[i]);
      }
      // Don't initialize variables on TPU if it is done for the ordinal
      // already.
      if (seen_ordinals_.contains(device_ordinal)) continue;

      Device* d;
      TF_RETURN_IF_ERROR(library_runtime_->device_mgr()->LookupDevice(
          strings::StrCat(kTPUDeviceNamePrefix, device_ordinal), &d));
      Var* tpu_var;
      Status status = d->resource_manager()->Lookup(cname, sname, &tpu_var);
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(InitializeVarOnTPU(ctx, var, &ndef, device_ordinal,
                                              var_info.fast_mem));
        VLOG(3) << "Initialized variable on TPU: " << sname
                << " device_ordinal: " << device_ordinal;
      }
      tpu_variables[handle_fp] = new_node;
    }
  }

  // adjust the index attr of other non-resource arg nodes
  int new_index = 0;
  for (Node* node : graph->op_nodes()) {
    if (node->IsArg()) {
      node->ClearAttr("index");
      node->AddAttr("index", new_index);
      new_index++;
    }
  }

  seen_ordinals_.insert(device_ordinal);

  return OkStatus();
}

Status TPUPartitionedCallOp::ReplaceAndPartitionXLAShardingVariable(
    Graph* graph, OpKernelContext* ctx, int device_ordinal,
    ResourceHandle& handle, Node* variable, int num_cores_per_replica) {
  TF_ASSIGN_OR_RETURN(
      auto sharding,
      GetShardingFromNodeDef(variable->def(), /*add_metadata=*/false));
  xla::OpSharding xla_sharding;
  bool is_var_sharded = false;
  if (sharding.has_value() &&
      sharding.value().type() == xla::OpSharding::OTHER) {
    xla_sharding = sharding.value();
    for (int dim = 0; dim < GetDimsFromXLAShardingTiled(xla_sharding); dim++) {
      is_var_sharded |= xla_sharding.tile_assignment_dimensions(dim) > 1;
    }
  } else {
    xla_sharding.set_type(xla::OpSharding::REPLICATED);
    is_var_sharded = false;
  }
  VLOG(3) << "Replace and partition variable " << variable->name()
          << " with xla_sharding: " << xla_sharding.DebugString();

  core::RefCountPtr<Var> var;
  TF_RETURN_IF_ERROR(LookupResource(ctx, handle, &var));

  int split_dim = -1;
  int split_size = 0;

  if (is_var_sharded) {
    for (int dim = 0; dim < GetDimsFromXLAShardingTiled(xla_sharding); dim++) {
      if (xla_sharding.tile_assignment_dimensions(dim) > 1) {
        if (split_dim != -1) {
          return errors::InvalidArgument(
              "Currently we only support inference with one split dimension, "
              "however got sharding: ",
              xla_sharding.DebugString());
        }
        split_dim = dim;
        split_size = xla_sharding.tile_assignment_dimensions(dim);
      }
    }
    if (split_dim == -1 || split_dim >= var->tensor()->dims()) {
      return errors::InvalidArgument(
          "sharding split_dim ", split_dim, " for variable: ", variable->name(),
          " is -1 or large than the number of dimensions ",
          var->tensor()->dims());
    }
  }

  const string cname = ctx->resource_manager()->default_container();
  std::vector<Node*> per_core_vars;
  for (int core_index = device_ordinal;
       core_index < (device_ordinal + num_cores_per_replica); core_index++) {
    NodeDef ndef;
    uint64 fp = Fingerprint64(
        strings::StrCat(handle.container(), handle.name(), "_", core_index));
    ndef.set_name(strings::StrCat(handle.name(), fp));
    ndef.set_op(kVarHandleOp);
    ndef.set_device(strings::StrCat(kTPUDeviceNamePrefix, core_index));

    // Replace each _Arg node of type DT_RESOURCE that goes into a TPU node
    // by a VarHandleOp on TPU with shared_name "v_tpu_x" where "v" is the
    // shared_name of the variable on CPU and "x" is the rewritten device
    // ordinal.
    const string sname = strings::StrCat(handle.name(), "_tpu_", core_index);
    AddNodeAttr("shared_name", sname, &ndef);
    AddNodeAttr("container", cname, &ndef);
    AddNodeAttr("dtype", var->tensor()->dtype(), &ndef);

    TensorShapeProto proto;
    var->tensor()->shape().AsProto(&proto);

    if (is_var_sharded) {
      int dim_size = proto.dim(split_dim).size();
      if (dim_size % split_size != 0) {
        return errors::InvalidArgument("dimension size ", dim_size,
                                       " cannot be divisible by split size ",
                                       split_size);
      }
      proto.mutable_dim(split_dim)->set_size(dim_size / split_size);
    }
    AddNodeAttr("shape", proto, &ndef);

    TF_ASSIGN_OR_RETURN(Node * new_node, graph->AddNode(ndef));
    per_core_vars.push_back(new_node);
  }

  // Insert TPUPartitionedInput op.
  NodeDefBuilder builder(absl::StrCat(handle.name(), "/tpu_partitioned_input"),
                         "TPUPartitionedInput");
  builder.Attr("N", num_cores_per_replica);
  builder.Attr("T", DT_RESOURCE);
  builder.Attr("partition_dim", split_dim);
  builder.Attr("_XlaSharding", xla_sharding.SerializeAsString());
  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.reserve(num_cores_per_replica);
  for (int core_index = 0; core_index < num_cores_per_replica; core_index++) {
    inputs.push_back({per_core_vars[core_index]->name(), 0, DT_RESOURCE});
  }
  builder.Input(inputs);
  NodeDef node_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&node_def));
  TF_ASSIGN_OR_RETURN(Node * tpu_partitioned_input_node,
                      graph->AddNode(node_def));

  for (int core_index = 0; core_index < num_cores_per_replica; core_index++) {
    graph->AddEdge(per_core_vars[core_index], 0, tpu_partitioned_input_node,
                   core_index);
  }

  // Insert TPUReplicatedInput op.
  NodeDefBuilder replicated_builder(
      absl::StrCat(handle.name(), "/tpu_replicated_input"),
      "TPUReplicatedInput");
  replicated_builder.Attr("N", 1);
  replicated_builder.Attr("T", DT_RESOURCE);
  replicated_builder.Attr("is_mirrored_variable", true);
  std::vector<NodeDefBuilder::NodeOut> replicated_inputs;
  replicated_inputs.push_back(
      {tpu_partitioned_input_node->name(), 0, DT_RESOURCE});
  replicated_builder.Input(replicated_inputs);
  NodeDef replicated_node_def;
  TF_RETURN_IF_ERROR(replicated_builder.Finalize(&replicated_node_def));
  Status replicated_s;
  Node* tpu_replicated_input_node =
      graph->AddNode(replicated_node_def, &replicated_s);
  if (!replicated_s.ok()) {
    return replicated_s;
  }
  graph->AddEdge(tpu_partitioned_input_node, 0, tpu_replicated_input_node, 0);

  // Connect the TPUReplicatedInput node to the previous output nodes of the
  // variable, and remove the variable node.
  std::vector<Node*> dst_nodes;
  std::vector<int> src_indices;
  std::vector<int> dst_indices;
  for (const Edge* edge : variable->out_edges()) {
    dst_nodes.push_back(edge->dst());
    src_indices.push_back(edge->src_output());
    dst_indices.push_back(edge->dst_input());
  }
  for (int i = 0; i < dst_nodes.size(); i++) {
    graph->AddEdge(tpu_replicated_input_node, src_indices[i], dst_nodes[i],
                   dst_indices[i]);
  }

  graph->RemoveNode(variable);

  std::vector<NodeDef> ndefs;
  Status status;
  for (int core_index = 0; core_index < num_cores_per_replica; core_index++) {
    Device* d;
    TF_RETURN_IF_ERROR(library_runtime_->device_mgr()->LookupDevice(
        strings::StrCat(kTPUDeviceNamePrefix, device_ordinal + core_index),
        &d));
    string sname;
    const NodeDef& ndef = per_core_vars[core_index]->def();
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "shared_name", &sname));
    ndefs.push_back(ndef);
    Var* tpu_var;
    status = d->resource_manager()->Lookup(cname, sname, &tpu_var);
  }

  if (!status.ok()) {
    TF_RETURN_IF_ERROR(
        InitializeShardedVarOnTPU(ctx, var, ndefs, split_dim, device_ordinal));
  }

  return OkStatus();
}

Status TPUPartitionedCallOp::InferShapesWithResourceVar(
    Graph* graph, OpKernelContext* ctx,
    std::map<int, InferredShape>& arg_shapes,
    GraphShapeInfo* tpu_inferred_info) {
  auto shape_inference_graph_interim =
      absl::make_unique<Graph>(graph->flib_def());
  CopyGraph(*graph, shape_inference_graph_interim.get());

  for (Node* node : shape_inference_graph_interim->nodes()) {
    if (node->type_string() != "_Arg" ||
        node->attrs().Find("T")->type() != DT_RESOURCE)
      continue;

    std::vector<std::function<void()>> to_remove;

    for (const Edge* out_edge : node->out_edges()) {
      Node* read_node = out_edge->dst();
      if (read_node->type_string() != "ReadVariableOp") continue;

      for (const Edge* variable_edge : read_node->out_edges()) {
        // We are delaying these modifications as we cannot do in-place
        // modification of EdgeSets.
        to_remove.push_back(
            [variable_edge, graph = shape_inference_graph_interim.get(), node] {
              Node* dst = variable_edge->dst();
              graph->RemoveEdge(variable_edge);
              graph->AddEdge(node, variable_edge->src_output(), dst,
                             variable_edge->dst_input());
            });
      }
      to_remove.push_back(
          [graph = shape_inference_graph_interim.get(), out_edge, read_node] {
            graph->RemoveEdge(out_edge);
            graph->RemoveNode(read_node);
          });
    }

    for (auto& func : to_remove) {
      func();
    }

    int resource_arg_index = node->attrs().Find("index")->i();

    // Get resource variable tensor
    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, resource_arg_index);
    TF_RETURN_IF_ERROR(LookupResource(ctx, handle, &variable));

    const Tensor* variable_tensor = variable->tensor();
    std::vector<int> variable_tensor_vec;

    variable_tensor_vec.reserve(variable_tensor->dims());
    for (int d = 0; d < variable_tensor->dims(); ++d) {
      variable_tensor_vec.push_back(variable_tensor->dim_size(d));
    }

    PartialTensorShape partial_tensor_shape;
    auto partial_shape = PartialTensorShape::MakePartialShape(
        variable_tensor_vec.data(), variable_tensor_vec.size(),
        &partial_tensor_shape);
    InferredShape inferred_shape = {partial_tensor_shape};
    arg_shapes.emplace(resource_arg_index, inferred_shape);
  }

  TF_RETURN_IF_ERROR(tensorflow::InferShapes(
      shape_inference_graph_interim.get(), arg_shapes,
      &shape_inference_graph_interim->flib_def(), tpu_inferred_info));
  return OkStatus();
}

Status TPUPartitionedCallOp::ShardInputsWithXlaSharding(
    Graph* graph, int num_cores_per_replica, OpKernelContext* ctx) {
  for (Node* replicated_input_node : graph->nodes()) {
    if (replicated_input_node->type_string() != "TPUReplicatedInput") continue;

    Node* arg_node;
    auto input_node_status = replicated_input_node->input_node(0, &arg_node);
    if (!input_node_status.ok()) {
      VLOG(2) << "Skip because cannot retrieve input node 0 of "
              << replicated_input_node->name() << " because "
              << input_node_status.ToString();
      continue;
    }

    // Check if this TPUReplicatedInput can qualify because it has _Arg
    // as input and doesn't have XlaSharding already as an output, then
    // try to shard inputs automatically.
    //
    // In short, we want to see the following graph:
    //    _Arg -> TPUReplicatedInput -> (not XlaSharding op)
    // and transform it to:
    //    _Arg -> TPUReplicatedInput -> XlaSharding -> (not XlaSharding op)
    if (arg_node->IsArg() &&
        replicated_input_node->out_nodes().begin()->type_string() !=
            "XlaSharding") {
      int arg_id;
      if (!absl::SimpleAtoi(absl::StripPrefix(arg_node->name(), "arg_"),
                            &arg_id)) {
        VLOG(3) << "Skip auto-sharding because we are unable to extract "
                   "argument number from "
                << arg_node->name();
        continue;
      }

      auto shape = ctx->input(arg_id).shape();

      VLOG(3) << "Identified arg node " << arg_node->DebugString()
              << " for TPUReplicatedInput "
              << replicated_input_node->DebugString();
      VLOG(3) << "Shape within TPUReplicatedInput is: " << shape.DebugString();

      int rank = shape.dims();
      int shard_dim =
          (runtime_params_.auto_xla_input_sharding_dim + rank) % rank;

      if (shape.dim_size(shard_dim) % num_cores_per_replica != 0) {
        VLOG(3) << "Skip auto-sharding " << replicated_input_node->name()
                << " because the specified sharding dimension " << shard_dim
                << " cannot be evenly split by " << num_cores_per_replica;
        continue;
      }

      auto sharding = absl::make_optional<xla::OpSharding>();
      sharding->set_type(xla::OpSharding::OTHER);

      // Sets up tile_assignment_dimensions.
      std::vector<int64_t> dims(rank, 1LL);
      dims[shard_dim] = num_cores_per_replica;
      for (auto dim : dims) {
        sharding->add_tile_assignment_dimensions(dim);
      }

      // Sets up tile_assignment_devices.
      for (int d = 0; d < num_cores_per_replica; ++d) {
        sharding->add_tile_assignment_devices(d);
      }

      std::vector<const Edge*> edges_to_remove;
      for (const Edge* edge : replicated_input_node->out_edges()) {
        if (edge->IsControlEdge()) continue;
        edges_to_remove.push_back(edge);
      }

      // Create XlaSharding Op.
      Node* sharding_op = nullptr;
      TF_RETURN_IF_ERROR(
          NodeBuilder(absl::StrCat(replicated_input_node->name(), "/sharding"),
                      "XlaSharding")
              .Input(replicated_input_node, 0)
              .Attr("T", replicated_input_node->output_type(0))
              .Attr(kXLAShardingAttrName, sharding->SerializeAsString())
              .Attr(kXLAShardingAttrAltName, sharding->SerializeAsString())
              .Attr("_tpu_replicate", "cluster")
              .Finalize(graph, &sharding_op));
      for (const Edge* edge : edges_to_remove) {
        VLOG(3) << "XlaSharding op creation output edge "
                << edge->DebugString();
        graph->RemoveEdge(edge);
        graph->AddEdge(sharding_op, 0, edge->dst(), edge->dst_input());
      }

      VLOG(3) << "Auto shard " << replicated_input_node->name() << " by dim "
              << shard_dim << " into " << num_cores_per_replica << " slices";

      VLOG(3) << "Created XlaSharding Op " << sharding_op->DebugString();
    }
  }

  return OkStatus();
}

// OptimizeTpuInputOutputTensors does the following things;
//  (1) Detect input arguments, and add XlaSharding op to the arguments if the
//  enable_auto_xla_input_sharding is turned on
//  (2) Pack multiple input tensors into one tensor by a concat to avoid PCIe
//  transfer overheads for small tensors.
//  (3) Reshape input tensors to R1 to leverage the fast path in TPU input
//  preparation done by runtime.
//  (4) Pack multiple output tensors into one tensor by a concat.
//
// (1) is controlled by --enable_auto_xla_input_sharding and
// --auto_xla_input_sharding_dim
// (2) and (3) are controlled by flags --minimum_input_tensors_packing
// and --input_shape_opt, respectively, while (4) is controlled by
// --minimum_output_tensors_packing.
Status TPUPartitionedCallOp::OptimizeTpuInputOutputTensors(
    Graph* graph, bool enable_spmd_xla_partitioning, int num_cores_per_replica,
    std::map<std::string, std::vector<int>>& named_input_shapes,
    OpKernelContext* ctx) {
  if (runtime_params_.enable_auto_xla_input_sharding) {
    VLOG(2) << DumpGraphToFile("before_enable_auto_xla_input_sharding", *graph,
                               flib_def_.get());

    TF_RETURN_IF_ERROR(
        ShardInputsWithXlaSharding(graph, num_cores_per_replica, ctx));
  }

  GraphShapeInfo tpu_inferred_info;
  std::map<int, InferredShape> arg_shapes;
  EdgeShapes tpu_input_shapes;
  absl::flat_hash_map<const Edge*, DataType> tpu_input_dtypes;

  // Contains attrs "T", "sharding", "_tpu_replicate" for each XlaSharding op.
  XlaShardingInfoMap xla_sharding_ops;

  // Contains attrs "T", and a pointer to tpu_replicated_metadata for ctrl dep
  TpuReplicatedInputInfoMap tpu_replicated_input_ops;

  bool xla_spmd_input_sharded = false;

  if (enable_spmd_xla_partitioning) {
    xla_spmd_input_sharded = FindTpuReplicatedInputAndXlaSharding(
        graph, xla_sharding_ops, tpu_replicated_input_ops);
  }

  VLOG(1) << "xla_spmd_input_sharded: " << xla_spmd_input_sharded;
  VLOG(2) << DumpGraphToFile("before_remove_descendant_nodes", *graph,
                             flib_def_.get());

  if (!xla_spmd_input_sharded ||
      runtime_params_.minimum_input_tensors_packing > 1 ||
      runtime_params_.enable_auto_xla_input_sharding) {
    // Currently we remove `TPUReplicatedInput` nodes when the input tensors are
    // not sharded, input tensors packing optimization is enabled or when
    // auto xla input sharding is there.
    //
    // In all thse cases, we want to remove both the TPUReplicatedInput and
    // XlaSharding ops or else downstream rewrites will be confused.
    RemoveDescendantNodeOfArg(graph, "TPUReplicatedInput",
                              /*must_be_child_of=*/{});
  }

  if (xla_spmd_input_sharded) {
    // We are setting must_be_child_of to {"Arg"} because we do not want
    // to remove other XlaSharding ops that might be in the graph. We only
    // want the XlaSharding ops that are directly attached to the input
    // arguments to be removed.
    RemoveDescendantNodeOfArg(graph, "XlaSharding",
                              /*must_be_child_of=*/{"_Arg"});
  }

  VLOG(2) << DumpGraphToFile("before_get_input_output_info", *graph,
                             flib_def_.get());

  TF_RETURN_IF_ERROR(GetInputOutputInfo(graph, tpu_inferred_info, arg_shapes,
                                        tpu_input_shapes, tpu_input_dtypes,
                                        ctx));

  VLOG(2) << DumpGraphToFile("before_optimize_tpu_input_output_tensors", *graph,
                             flib_def_.get());

  string cluster_name;
  TF_RETURN_IF_ERROR(GetClusterName(graph, &cluster_name));

  if (runtime_params_.minimum_output_tensors_packing > 1) {
    // Copy graph to shape_inference_graph
    EdgeShapes tpu_output_shapes;
    TF_RETURN_IF_ERROR(
        InferShapesWithResourceVar(graph, ctx, arg_shapes, &tpu_inferred_info));

    // Find TPU -> CPU output edges.
    GroupedEdges shape_to_output =
        tpu_functional_internal::GroupTensorsForOutputPacking(
            graph, tpu_output_shapes, &tpu_inferred_info);

    TF_RETURN_IF_ERROR(
        tpu_functional_internal::CreateConcatAndSplitNodesForOutputTensor(
            graph, cluster_name, &tpu_output_shapes, &tpu_inferred_info,
            shape_to_output, runtime_params_.minimum_output_tensors_packing));
  }

  if (runtime_params_.minimum_input_tensors_packing > 1) {
    GroupedEdges grouped_input_edges =
        tpu_functional_internal::GroupTensorsForInputPacking(
            tpu_input_shapes, tpu_input_dtypes, runtime_params_.input_shape_opt,
            runtime_params_.group_tensors_for_packing);
    TF_RETURN_IF_ERROR(
        tpu_functional_internal::CreateConcatAndSplitNodesForInputTensor(
            graph, cluster_name, &tpu_input_shapes, grouped_input_edges,
            runtime_params_.minimum_input_tensors_packing,
            xla_spmd_input_sharded, xla_sharding_ops,
            tpu_replicated_input_ops));
  }
  if (runtime_params_.input_shape_opt) {
    TF_RETURN_IF_ERROR(tpu_functional_internal::InsertReshapeNodePairs(
        graph, cluster_name, &tpu_input_shapes, num_cores_per_replica));
  }
  VLOG(1) << DumpGraphToFile("optim_result", *graph);

  // With or without optimizations, collect the input names and shapes.
  for (const auto& iter : tpu_input_shapes) {
    std::string name = iter.first->src()->name();
    named_input_shapes[name] = iter.second;
  }
  return OkStatus();
}

Status TPUPartitionedCallOp::GetGraphFromFunction(
    Graph* graph, int device_ordinal, int* num_core_per_replica,
    bool* use_spmd_for_xla_partitioning) {
  FunctionLibraryRuntime::InstantiateOptions opts;
  FHandle handle;
  TF_RETURN_IF_ERROR(library_runtime_->Instantiate(
      func_.name(), AttrSlice(&func_.attr()), opts, &handle));
  const FunctionBody* fbody = library_runtime_->GetFunctionBody(handle);
  if (fbody == nullptr) {
    return errors::Internal("Could not find handle ", handle);
  }
  CopyGraph(*fbody->graph, graph);

  // Pin the inputs and outputs to the local device to simplify the
  // function-dispatching logic.
  local_device_name_ = library_runtime_->device()->name();
  replaced_input_indices_.resize(fbody->arg_nodes.size(), false);
  for (Node* node : graph->op_nodes()) {
    if (node->IsArg() || node->IsRetval()) {
      node->set_assigned_device_name(local_device_name_);
    } else if (node->type_string() == "TPUReplicateMetadata") {
      // Record the producer name so it can be accessed later during metric
      // collection.
      string producer_name = GetProducerName(func_.name());
      node->AddAttr("_producer_name", producer_name);

      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "num_cores_per_replica",
                                     num_core_per_replica));
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(),
                                     "use_spmd_for_xla_partitioning",
                                     use_spmd_for_xla_partitioning));
      VLOG(1) << "num_core_per_replica = " << *num_core_per_replica
              << ", use_spmd_for_xla_partitioning = "
              << *use_spmd_for_xla_partitioning;

      if (*num_core_per_replica > 1) {
        std::string topology_str;
        std::vector<int> device_assignment;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(node->attrs(), "topology", &topology_str));
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "device_assignment",
                                       &device_assignment));

        tpu::TopologyProto topology;
        topology.ParseFromString(topology_str);
        int num_cores = topology.device_coordinates_size() / 4;

        if (device_assignment.empty()) {
          VLOG(1) << "Auto assigning device assignment";
          // Number of devices match the cores per replica, so we can just use
          // the device assignment from the existing topology instead of
          // generating our own.
          //
          // TODO(b/179292031): Add support for non-natural orders for pods.

          // check that the device coordinates for a donut is always in
          // natural order.
          std::vector<int> natural_order;
          // Be smart about mesh choice considering TPU platform, given V4
          // has potentially different mesh shapes.
          tpu::TpuPlatformInterface* tpu_platform =
              tpu::TpuPlatformInterface::GetRegisteredPlatform();
          tpu::TpuTopologyExternal tpu_topology = tpu_platform->topology();
          bool single_logic_device_per_chip =
              tpu_topology.LogicalDevicesPerChip(
                  TpuCoreTypeEnum::kTensorCore) == 1;
          switch (num_cores) {
            case 2:
              if (single_logic_device_per_chip) {
                TF_RETURN_IF_ERROR(GenerateDeviceNaturalOrder(
                    /*x_num_cores=*/1, /*y_num_cores=*/2, /*z_num_cores=*/1,
                    /*num_cores_per_chip=*/1, &natural_order));
              } else {
                TF_RETURN_IF_ERROR(GenerateDeviceNaturalOrder(
                    /*x_num_cores=*/1, /*y_num_cores=*/1, /*z_num_cores=*/1,
                    /*num_cores_per_chip=*/2, &natural_order));
              }
              break;
            case 4:
              if (single_logic_device_per_chip) {
                TF_RETURN_IF_ERROR(GenerateDeviceNaturalOrder(
                    /*x_num_cores=*/2, /*y_num_cores=*/2, /*z_num_cores=*/1,
                    /*num_cores_per_chip=*/1, &natural_order));
              } else {
                TF_RETURN_IF_ERROR(GenerateDeviceNaturalOrder(
                    /*x_num_cores=*/1, /*y_num_cores=*/2, /*z_num_cores=*/1,
                    /*num_cores_per_chip=*/2, &natural_order));
              }
              break;
            case 8:
              if (!single_logic_device_per_chip) {
                TF_RETURN_IF_ERROR(GenerateDeviceNaturalOrder(
                    /*x_num_cores=*/2, /*y_num_cores=*/2, /*z_num_cores=*/1,
                    /*num_cores_per_chip=*/2, &natural_order));
                break;
              }
              // Intentionally fall through since with v4 shape and 8 cores per
              // replica, we're crossing host bounds -- so we ask for a explicit
              // device assignment.
              ABSL_FALLTHROUGH_INTENDED;
            default:
              return errors::Unimplemented(
                  "Unable to auto assign device assignment. For topology cross "
                  "host bounds, you must explicit specify an assignment.");
          }
          if (*num_core_per_replica != num_cores &&
              !std::equal(natural_order.begin(), natural_order.end(),
                          topology.device_coordinates().begin())) {
            return errors::InvalidArgument(
                "Topology device coordinates for XLA SPMD on donuts must be in "
                "natural order.");
          }

          auto coordinates_start =
              topology.device_coordinates().begin() + device_ordinal * 4;
          auto coordinates_end = topology.device_coordinates().begin() +
                                 (device_ordinal + *num_core_per_replica) * 4;

          node->ClearAttr("device_assignment");
          device_assignment.insert(device_assignment.begin(), coordinates_start,
                                   coordinates_end);
          node->AddAttr("device_assignment", device_assignment);
        }
      }
    }
  }
  return OkStatus();
}

Status TPUPartitionedCallOp::PlacementHelper(
    const DeviceSet& device_set,
    const GraphOptimizationPassOptions& optimization_options,
    const string& function_name) {
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));
  Placer placer(optimization_options.graph->get(), function_name,
                optimization_options.flib_def, &device_set);
  TF_RETURN_IF_ERROR(placer.Run());
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PLACEMENT, optimization_options));
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));
  return OkStatus();
}

Status TPUPartitionedCallOp::PartitionHelper(
    const DeviceSet& device_set,
    const GraphOptimizationPassOptions& optimization_options, Graph* graph,
    std::unordered_map<std::string, std::unique_ptr<Graph>>* subgraphs) {
  PartitionOptions partition_options;
  partition_options.node_to_loc = [](const Node* node) {
    // TODO(akshayka): To better support the distributed case, first split
    // the graph by worker (e.g,. using the master session's
    // `SplitByWorker` policy), and then recursively partition the
    // per-worker shards at the remote worker(s).
    return node->assigned_device_name();
  };
  int64_t edge_name_counter = 0;
  partition_options.new_name = [&edge_name_counter](const string& prefix) {
    return strings::StrCat(prefix, "/_", ++edge_name_counter);
  };
  partition_options.get_incarnation = [&device_set](const string& name) {
    const Device* d = device_set.FindDeviceByName(name);
    if (d == nullptr) {
      return PartitionOptions::kIllegalIncarnation;
    } else {
      return d->attributes().incarnation();
    }
  };
  partition_options.control_flow_added = false;
  std::unordered_map<std::string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(partition_options, graph, &partitions));

  VLOG(3) << "Partitioned function '" << func_.name() << "', yielding "
          << partitions.size() << " shards.";

  const FunctionLibraryDefinition* flib_def = &graph->flib_def();
  for (auto& partition : partitions) {
    std::unique_ptr<Graph> subgraph(new Graph(flib_def));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    const string& device = partition.first;
    GraphDef& graph_def = partition.second;
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(opts, std::move(graph_def), subgraph.get()));
    subgraphs->emplace(device, std::move(subgraph));
  }

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  return OkStatus();
}

Status TPUPartitionedCallOp::InstantiatePartition(
    const Graph& graph, const string& function_name,
    const string& target_device, FHandle* handle,
    std::unique_ptr<FunctionLibraryDefinition>* out_flib_def) {
  FunctionDef shard;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(graph, function_name, &shard));
  TF_RETURN_IF_ERROR(flib_def_->AddFunctionDef(shard));
  FunctionLibraryRuntime::InstantiateOptions opts;
  opts.target = target_device;
  if (out_flib_def) {
    *out_flib_def = std::make_unique<FunctionLibraryDefinition>(*flib_def_);
    opts.lib_def = out_flib_def->get();
  } else {
    opts.lib_def = flib_def_.get();
  }
  return library_runtime_->Instantiate(function_name, AttrSlice(&shard.attr()),
                                       opts, handle);
}

Status TPUPartitionedCallOp::SetDeviceOrdinal(const DeviceSet& device_set,
                                              int device_ordinal, Graph* graph,
                                              bool* modified) {
  int ordinal = -1;
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == kVarHandleOp) {
      if (IsInputToTPUReplicate(node)) {
        // If this VarHandleOp is going to a TPU computation,
        // it refers to the TPU variable that we created when replacing the
        // resource arguments with VarHandleOps.
        node->set_assigned_device_name(
            strings::StrCat(kTPUDeviceNamePrefix, device_ordinal));
      }
      continue;
    }
    if (HasNodeAttr(node->def(), kXlaHasHostTransferAttrName)) {
      // Outside compilation related node.
      TF_RETURN_IF_ERROR(
          SetDeviceOrdinalAttributeForNode(node, device_ordinal));
      *modified = true;
      continue;
    }
    const AttrValue* attr = node->attrs().Find(kDeviceOrdinalAttr);
    if (attr != nullptr) {
      if (!IsSupportedTPUOp(node->type_string())) {
        return errors::InvalidArgument("Node ", node->type_string(),
                                       " is not yet supported.");
      }
      if (ordinal == -1) {
        ordinal = attr->i();
      } else {
        if (ordinal != attr->i()) {
          return errors::InvalidArgument(
              "Can only partition graphs that use a single device ordinal.");
        }
      }
      node->ClearAttr(kDeviceOrdinalAttr);
      node->AddAttr(kDeviceOrdinalAttr, device_ordinal);
      VLOG(3) << "Set device ordinal of " << node->type_string() << " to "
              << device_ordinal;
      *modified = true;
    }
    if (node->IsSend() || node->IsRecv()) {
      static const char* kSendDevice = "send_device";
      static const char* kSendDeviceIncarnation = "send_device_incarnation";
      static const char* kRecvDevice = "recv_device";
      const AttrValue* attr = node->attrs().Find(kSendDevice);
      if (attr != nullptr) {
        string device = attr->s();
        TF_RETURN_IF_ERROR(
            UpdateTPUDeviceOrdinal(device_ordinal, &device, modified));
        node->ClearAttr(kSendDevice);
        node->AddAttr(kSendDevice, device);
        node->ClearAttr(kSendDeviceIncarnation);
        const Device* d = device_set.FindDeviceByName(device);
        int64_t send_incarnation = (d == nullptr)
                                       ? PartitionOptions::kIllegalIncarnation
                                       : d->attributes().incarnation();
        node->AddAttr(kSendDeviceIncarnation, send_incarnation);
      }
      attr = node->attrs().Find(kRecvDevice);
      if (attr != nullptr) {
        string device = attr->s();
        TF_RETURN_IF_ERROR(
            UpdateTPUDeviceOrdinal(device_ordinal, &device, modified));
        node->ClearAttr(kRecvDevice);
        node->AddAttr(kRecvDevice, device);
      }
    }
  }
  return OkStatus();
}

Status TPUPartitionedCallOp::InstantiateFunctionsFromSubgraphs(
    const DeviceSet& device_set, int replica_id, uint64 cache_hash,
    int num_cores_per_replica,
    std::unordered_map<std::string, std::unique_ptr<Graph>> subgraphs) {
  const Device* reference_device = nullptr;
  auto entry =
      partition_cache_.emplace(cache_hash, std::vector<DeviceAndFHandle>());

  bool rewritten = false;
  for (auto& pair : subgraphs) {
    string target = pair.first;
    int device_ordinal = replica_id;
    if (num_cores_per_replica > 1) {
      DeviceNameUtils::ParsedName parsed_device;
      if (!DeviceNameUtils::ParseFullName(target, &parsed_device)) {
        return errors::InvalidArgument("Malformed assigned device '", target,
                                       "'");
      }
      device_ordinal = parsed_device.id;
    }
    Device* device;
    TF_RETURN_IF_ERROR(
        library_runtime_->device_mgr()->LookupDevice(target, &device));
    if (reference_device == nullptr) {
      reference_device = device;
    } else {
      if (!DeviceNameUtils::IsSameAddressSpace(
              device->parsed_name(), reference_device->parsed_name())) {
        return errors::InvalidArgument(
            "TPUPartitionedCallOp does not yet support inter-process"
            "execution.");
      }
    }
    TF_RETURN_IF_ERROR(device->MaybeRewriteGraph(&pair.second));
    Graph* subgraph = pair.second.get();
    // For model paralleism inference, we only support num_replica == 1, thus
    // there is no need to update the device_ordinal anymore.
    if (num_cores_per_replica == 1) {
      TF_RETURN_IF_ERROR(
          SetDeviceOrdinal(device_set, device_ordinal, subgraph, &rewritten));
    } else {
      VLOG(1) << "Skip SetDeviceOrdinal()";
    }
    string function_name = flib_def_->UniqueFunctionName(
        strings::StrCat(func_.name(), "_hash_", cache_hash));
    TF_RETURN_IF_ERROR(
        UpdateTPUDeviceOrdinal(device_ordinal, &target, &rewritten));
    FHandle handle;
    // Use a copy of the current `flib_def_` to instantiate the function to
    // avoid races.
    std::unique_ptr<FunctionLibraryDefinition> sub_flib_def;
    TF_RETURN_IF_ERROR(InstantiatePartition(*subgraph, function_name, target,
                                            &handle, &sub_flib_def));
    // Add handle to the cache entry.
    entry.first->second.push_back(
        DeviceAndFHandle{.device = target,
                         .handle = handle,
                         .flib_def = std::move(sub_flib_def)});
  }

  if (!rewritten) {
    // For regular use cases, TPUPartitionedCallOp only works when the
    // function being called in rewritten for TPU. If we don't see any signs
    // of this rewriting, warn the user about it.
    // We don't raise an error because we want to support the use case of
    // running tpu.initialize_system eagerly. In this case, we can't use
    // tpu.rewrite because it will add compilation ops that require TPU
    // to be initialized, i.e. there is a chicken and egg problem.
    // We run tpu.initialize_system through TPUPartitionedCallOp because it
    // invokes graph rewrite passes that are necessary for initialization to
    // work.
    LOG(INFO) << "Function body was not rewritten for TPU. "
              << "This is probably a bug unless you are initializing "
              << "TPUs eagerly.";
  }
  return OkStatus();
}

void TPUPartitionedCallOp::ExecuteRemoteFunction(
    const FunctionLibraryRuntime::Options& opts, FHandle handle,
    OpKernelContext* ctx, ReffedStatusCallback* done) {
  std::vector<Tensor> dummy_args;
  std::vector<Tensor>* dummy_rets = new std::vector<Tensor>;

  profiler::TraceMe trace_me("TPUPartitionedCallOp-ExecuteRemote");
  library_runtime_->Run(opts, handle, dummy_args, dummy_rets,
                        [dummy_rets, done, ctx](const Status& status) {
                          if (!status.ok()) {
                            done->UpdateStatus(status);
                          }
                          delete dummy_rets;
                          done->Unref();
                        });
}

void TPUPartitionedCallOp::ExecuteLocalFunction(
    const FunctionLibraryRuntime::Options& opts, const OpInputList& arguments,
    FHandle handle, OpKernelContext* ctx, ReffedStatusCallback* done) {
  std::vector<Tensor> args;

  for (int i = 0; i < arguments.size(); ++i) {
    if (!replaced_input_indices_[i]) {
      // _Arg nodes of type DT_RESOURCE that go into a TPU node have been
      // replaced by TPU VarHandleOp nodes. No longer need to pass them as
      // inputs.
      args.push_back(arguments[i]);
    }
  }
  auto* rets = new std::vector<Tensor>;

  profiler::TraceMe trace_me("TPUPartitionedCallOp-ExecuteLocal");
  library_runtime_->Run(opts, handle, args, rets,
                        [rets, done, ctx](const Status& status) {
                          if (!status.ok()) {
                            done->UpdateStatus(status);
                          } else {
                            for (int i = 0; i < rets->size(); ++i) {
                              ctx->set_output(i, (*rets)[i]);
                            }
                          }
                          delete rets;
                          done->Unref();
                        });
}

void TPUPartitionedCallOp::ExecuteFunctions(
    const std::vector<DeviceAndFHandle>& functions, OpKernelContext* ctx,
    int device_ordinal, int64_t ordinal_selector_req_id, DoneCallback done) {
  profiler::TraceMe trace_me("TPUPartitionedCallOp-ExecuteFunctions");
  FunctionLibraryRuntime::Options opts;
  opts.step_container = ctx->step_container();
  opts.stats_collector = ctx->stats_collector();
  // TODO(akshayka): Consider selecting a runner on a per-device basis,
  // i.e., using device-specific threadpools when available.
  opts.runner = ctx->runner();
  opts.source_device = local_device_name_;
  opts.run_all_kernels_inline = ctx->run_all_kernels_inline();

  OpInputList arguments;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &arguments), done);

  auto* local_cm = new CancellationManager(ctx->cancellation_manager());
  auto* rendez = new RefCountedIntraProcessRendezvous(device_mgr_);
  opts.cancellation_manager = local_cm;
  opts.rendezvous = rendez;

  StatusCallback callback(
      [rendez = rendez, local_cm, done = std::move(done),
       device_ordinal = device_ordinal, req_id = ordinal_selector_req_id, ctx,
       ordinal_selector = ordinal_selector_](const Status& status) {
        delete local_cm;
        rendez->Unref();
        if (!status.ok()) {
          ctx->SetStatus(status);
        }
        done();
        if (req_id >= 0) {
          ordinal_selector->DequeueFromCoreSelector(device_ordinal, req_id);
        }
      });

  auto* refcounted_done = new ReffedStatusCallback(std::move(callback));
  for (int i = 1; i < functions.size(); ++i) {
    refcounted_done->Ref();
  }
  for (const DeviceAndFHandle& entry : functions) {
    const string& target_device = entry.device;
    FHandle handle = entry.handle;
    VLOG(3) << "Running function shard on device " << target_device
            << " with local device name " << local_device_name_;
    if (target_device == local_device_name_) {
      opts.remote_execution = false;
      ExecuteLocalFunction(opts, arguments, handle, ctx, refcounted_done);
    } else {
      opts.remote_execution = true;
      ExecuteRemoteFunction(opts, handle, ctx, refcounted_done);
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("TPUPartitionedCall").Device(DEVICE_CPU),
                        TPUPartitionedCallOp);

}  // end namespace tensorflow
