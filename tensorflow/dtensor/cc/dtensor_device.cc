/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_topology.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/dtensor_device_util.h"
#include "tensorflow/dtensor/cc/dtensor_graph_to_mlir_pass.h"
#include "tensorflow/dtensor/cc/small_constant_optimization.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/cc/tpu_system_interface.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

// TODO(b/189332820): Replace this with a Partitioner stub swapped in by the
// Copybara workflow.
StatusOr<ExecutionFunctions> ABSL_ATTRIBUTE_WEAK PipeliningPartitionerRun(
    const absl::flat_hash_map<std::string, const MeshWithParallelDevice*>*
        device_name_to_mesh_device,
    FunctionLibraryDefinition* flib_def, DTensorMlirPassRunner* pass_runner,
    const FunctionDef& fdef, const NameAttrList& eager_attributes,
    const std::vector<TensorWithLayout*>& inputs, const DeviceSet& device_set,
    int num_outputs) {
  // The actual definition is in the pipelining package.
  return errors::Unimplemented("DTensor pipelining is unavailable.");
}

class DTensorDevice {
 public:
  explicit DTensorDevice(absl::string_view name)
      : name_(name),
        same_shape_policy_enabled_(false),
        cancellation_manager_(std::make_unique<CancellationManager>()) {}

  void AddMesh(std::unique_ptr<MeshWithParallelDevice> mesh,
               bool is_host_mesh) {
    if (is_host_mesh) {
      std::string& tpu_host_mesh = Mesh::tpu_host_mesh();
      const std::string new_tpu_host_mesh = mesh->mesh_config().ToString();
      if (!tpu_host_mesh.empty()) {
        // TODO(b/180046115): Add per-TPU-mesh host mesh bookkeeping.
        LOG(WARNING)
            << "A new TPU host mesh is overwriting the old TPU host mesh. The "
               "old TPU mesh cannot be used in sea of donuts mode anymore.";
      }
      tpu_host_mesh.assign(new_tpu_host_mesh);
    }
    // For idempotency, don't register the same mesh twice.
    if (!mesh_to_device_map_.insert({mesh->mesh_config(), std::move(mesh)})
             .second)
      return;
    if (!default_mesh_) {
      global_default_mesh_ = mesh_to_device_map_.begin()->second.get();
      default_mesh_ = global_default_mesh_;
    }
  }

  // Returns sub meshes of pipelining.
  // Key is the name of a composite device.
  StatusOr<absl::flat_hash_map<std::string, const MeshWithParallelDevice*>>
  PipelineSubMeshes(TFE_Context* context) {
    absl::flat_hash_map<std::string, const MeshWithParallelDevice*>
        device_to_mesh;
    for (const auto& pair : mesh_to_device_map_) {
      TF_ASSIGN_OR_RETURN(CompositeDevice * device,
                          pair.second->FindOrCreateCompositeDevice(context));
      if (device != nullptr) {
        device_to_mesh[pair.second->composite_device()->name()] =
            pair.second.get();
      }
    }
    return device_to_mesh;
  }

  // Runs an operation on the DTensorDevice,
  //
  // Ignoring the placement of the original op (TFE_OpGetDevice(original_op)).
  // This indicates whether the user explicitly placed the op on the DTensor
  // device (vs. having it placed on the DTensor device because an input was
  // placed there), but DTensor is doing type-based dispatch and so handles
  // these cases identically at the moment.
  void Execute(const TFE_Op* original_op, int* num_outputs,
               TFE_TensorHandle** outputs, TF_Status* status);

  void SetDefaultLayout(Layout layout) { default_layout_.emplace(layout); }
  void ClearDefaultLayout() { default_layout_.reset(); }
  void SetDefaultMesh(Mesh mesh) {
    default_mesh_ = mesh_to_device_map_.at(mesh).get();
  }
  void ClearDefaultMesh() { default_mesh_ = global_default_mesh_; }
  void SetSameShapePolicy(bool enabled) {
    same_shape_policy_enabled_ = enabled;
  }

  Status SetTPUCoreIDs(const std::string& mesh_name,
                       const std::vector<int>& tpu_core_ids) {
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "Setting TPU core IDs for "
                << (mesh_name.empty() ? "default mesh" : mesh_name) << ": ";
      for (auto i : tpu_core_ids) {
        LOG(INFO) << i;
      }
    }
    // Setting the default mesh under an empty name repeatedly is fine, which
    // happens when dtensor_initialize_tpu_system is called multiple times
    // especially in tests. All the set mappings should be the same anyway.
    if (!mesh_name.empty() && Mesh::tpu_core_ids().count(mesh_name) > 0) {
      return errors::AlreadyExists("Mesh name already in use: ", mesh_name);
    }
    Mesh::tpu_core_ids()[mesh_name].assign(tpu_core_ids.begin(),
                                           tpu_core_ids.end());
    return OkStatus();
  }

  void ClearTPUCoreIDs() { Mesh::tpu_core_ids().clear(); }

  std::vector<std::vector<int>> TPUCoreIDsToLocations(
      TFE_Context* context, const std::vector<int>& tpu_core_ids) {
    TpuSystemInterface* tpu_system = GetPreferredTpuSystem();
    if (tpu_system == nullptr) {
      VLOG(1) << "Calling TPUCoreIDsToLocations on the default TPU system.";
      std::vector<std::vector<int>> tpu_core_locations;
      tpu_core_locations.reserve(tpu_core_ids.size());
      tpu::TpuPlatformInterface* tpu_platform =
          tpu::TpuPlatformInterface::GetRegisteredPlatform();
      if (tpu_platform == nullptr) {
        LOG(WARNING) << "No TPU platform is found.";
        return {{}};
      }
      if (!tpu_platform->Initialized()) {
        LOG(WARNING) << "TPU platform is not initialized.";
        return {{}};
      }
      tpu::TpuTopologyExternal tpu_topology = tpu_platform->topology();

      for (const int& tpu_core_id : tpu_core_ids) {
        tpu::TpuCoreLocationExternal core =
            tpu_topology.CoreForId(TpuCoreTypeEnum::kTensorCore, tpu_core_id);
        tpu::TpuDimensionsExternal tpu_chip_location = core.chip_coordinates();
        tpu_core_locations.push_back({tpu_chip_location.x, tpu_chip_location.y,
                                      tpu_chip_location.z, core.index()});
      }
      return tpu_core_locations;
    } else {
      VLOG(1) << "Calling TPUCoreIDsToLocations on a preferred TPU system.";
      return tpu_system->TPUCoreIDsToLocations(context, tpu_core_ids);
    }
  }

  std::vector<int> TPUCoreLocationsToIDs(
      TFE_Context* context,
      const std::vector<std::vector<int>>& tpu_core_locations) {
    TpuSystemInterface* tpu_system = GetPreferredTpuSystem();
    if (tpu_system == nullptr) {
      VLOG(1) << "Calling TPUCoreLocationsToIDs on the default TPU system.";
      std::vector<int> tpu_core_ids;
      tpu_core_ids.reserve(tpu_core_locations.size());
      tpu::TpuPlatformInterface* tpu_platform =
          tpu::TpuPlatformInterface::GetRegisteredPlatform();
      if (tpu_platform == nullptr) {
        LOG(WARNING) << "No TPU platform is found.";
        return {};
      }
      if (!tpu_platform->Initialized()) {
        LOG(WARNING) << "TPU platform is not initialized.";
        return {};
      }
      tpu::TpuTopologyExternal tpu_topology = tpu_platform->topology();

      for (const std::vector<int>& tpu_core_location : tpu_core_locations) {
        tpu::TpuCoreLocationExternal core = tpu_topology.Core(
            TpuCoreTypeEnum::kTensorCore, tpu_core_location[0],
            tpu_core_location[1], tpu_core_location[2], tpu_core_location[3]);
        tpu_core_ids.push_back(core.Id());
      }
      return tpu_core_ids;
    } else {
      VLOG(1) << "Calling TPUCoreLocationsToIDs on a preferred TPU system.";
      return tpu_system->TPUCoreLocationsToIDs(context, tpu_core_locations);
    }
  }

  // Waits for ops to finish in ALL meshes as we share the cancellation manager.
  void AsyncWait(TFE_Context* context, TF_Status* status) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> first_bad_status(
        nullptr, TF_DeleteStatus);

    for (const auto& pair : mesh_to_device_map_) {
      std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> async_wait_status(
          TF_NewStatus(), TF_DeleteStatus);

      pair.second->parallel_device().AsyncWait(context,
                                               async_wait_status.get());

      TF_Code error_code = TF_GetCode(async_wait_status.get());
      if (error_code != TF_OK &&
          (first_bad_status == nullptr ||
           TF_GetCode(first_bad_status.get()) == TF_CANCELLED)) {
        first_bad_status.reset(TF_NewStatus());
        TF_SetStatus(first_bad_status.get(), error_code,
                     TF_Message(async_wait_status.get()));
      }
    }

    if (first_bad_status != nullptr) {
      TF_SetStatus(status, TF_GetCode(first_bad_status.get()),
                   TF_Message(first_bad_status.get()));
    }

    // Reset the global function rendezvous, which otherwise stores a failure
    // state.
    tensorflow::unwrap(context)->ResetGlobalRendezvousForFunction();

    // Reset the cancellation manager on (potential) failure so we don't cancel
    // future ops. This is only safe because we have just cleared pending async
    // nodes, which may have had a reference to he cancellation manager.
    cancellation_manager_ = std::make_unique<CancellationManager>();
  }

  TFE_TensorHandle* Pack(TFE_Context* context, int num_inputs,
                         TFE_TensorHandle** inputs,
                         const std::string& string_layout, TF_Status* status);

  std::vector<TFE_TensorHandle*> Unpack(TFE_Context* context,
                                        TFE_TensorHandle* input,
                                        TF_Status* status);

  // Return the layout for the input tensor.
  std::string FetchLayout(TFE_Context* context, TFE_TensorHandle* input,
                          TF_Status* status);

  TFE_TensorHandle* SparsePack(TFE_Context* context, int num_inputs,
                               TFE_TensorHandle** indices,
                               TFE_TensorHandle** values,
                               TFE_TensorHandle** shapes,
                               const std::string& string_layout,
                               TF_Status* status);

  bool IsSparseDTensor(TFE_Context* context, TFE_TensorHandle* input,
                       TF_Status* status);

  std::unordered_map<std::string, int> GetFunctionCacheHitAndMissCount(
      TFE_Context* context, TF_Status* status) const;

 private:
  // If the `operation_name` of an op indicates a custom DTensor op (e.g.
  // CopyToMesh), then separately handle those custom ops instead of running
  // default DTensor graph compilation.
  void MaybeHandleDTensorCustomOps(
      const char* operation_name, const int num_inputs,
      const TFE_OpAttrs* attributes, TFE_Context* context,
      TFE_TensorHandle** inputs, int* num_outputs, TFE_TensorHandle** outputs,
      bool* is_custom_dtensor_op, TF_Status* status);

  // Copies non-dtensor eager tensor or DTensor to a mesh specified by
  // `attributes`.
  // Currently, only copy to replicated layout on target mesh is supported.
  void CopyToMesh(TFE_Context* context, int num_inputs,
                  TFE_TensorHandle** inputs, const TFE_OpAttrs* attributes,
                  TFE_TensorHandle** outputs, int* num_outputs,
                  TF_Status* status);

  // Update output layouts for eager ops based on same shape policy.
  void UpdateOutputLayoutsWithSameShapePolicy(
      const std::vector<PartialTensorShape>& global_output_shapes,
      const absl::flat_hash_set<Mesh>& input_meshes, absl::string_view op_name,
      tensorflow::Graph* graph, std::vector<const Layout*>* output_layouts,
      TF_Status* status);

  // Takes the description of an operation and makes a function out of it with
  // the same signature, running DTensor MLIR passes. Registers that function
  // with `context`. `translated_function_name` is set to the name of the
  // function.
  //
  // The resulting function expects a device ID as its first input.
  void LowerToSPMDFunction(TFE_Context* context,
                           const std::vector<TensorWithLayout*>& inputs,
                           const DTensorOperation& doperation,
                           const TFE_OpAttrs* attributes, const int num_outputs,
                           const ExecutionFunctions** execution_functions,
                           TF_Status* status);

  // Execute a given function.
  void ExecuteFunctionAndWait(
      TFE_Context* context, const TranslatedFunction* function_ptr,
      const MeshWithParallelDevice* parallel_device_mesh,
      const std::vector<parallel_device::ParallelTensor*>& parallel_inputs,
      const int64_t step_id, const TFE_OpAttrs* attributes, TF_Status* status);

  // Implements `Execute` for operations which aren't special-cased in
  void ExecuteRegularOperation(TFE_Context* context,
                               const std::vector<TensorWithLayout*>& inputs,
                               const DTensorOperation& doperation,
                               const TFE_OpAttrs* attributes, int* num_outputs,
                               TFE_TensorHandle** outputs, TF_Status* status);

  // Wraps a TensorWithLayout into a TFE_TensorHandle.
  TFE_TensorHandle* MakeLayoutTensorHandle(TFE_Context* context,
                                           std::unique_ptr<TensorWithLayout> t,
                                           TF_Status* status);

  void RecordInShapeLayoutCache(const TensorWithLayout& tensor);

  // Returns whether a given mesh is a remote mesh.
  bool is_remote_mesh(const Mesh& mesh) const;

  // The name of the device (the custom device)
  std::string name_;
  // Mesh configs with matching parallel devices.
  //
  // For now we just consider the first entry added to dtensor_device as the
  // default mesh. Before we reach an agreement on this, we'll leave it as is.
  absl::flat_hash_map<Mesh, std::unique_ptr<MeshWithParallelDevice>>
      mesh_to_device_map_;
  // TODO(hthu): Consider whether we want to preserve the default_mesh semantic.
  // Current default mesh consistent to default_layout_. If default_layout_ is
  // not set, it equals to global_default_mesh_.
  const MeshWithParallelDevice* default_mesh_ = nullptr;
  // The default mesh of a DTensorDevice, which cannot be modified once being
  // set.
  const MeshWithParallelDevice* global_default_mesh_ = nullptr;
  // If the user has specified a default output layout.
  absl::optional<Layout> default_layout_;

  // Determines whether tensors with a shape previously associated with only one
  // layout use that layout if nothing else can be inferred.
  bool same_shape_policy_enabled_;

  DTensorMlirPassRunner pass_runner_;

  struct CachedLayout {
    // The first layout seen with this shape
    Layout layout;
    // Whether the layout is unique for this shape
    bool is_unique;
  };
  absl::flat_hash_map<int64_t, CachedLayout> shape_layout_cache_;

  FunctionManager function_manager_;

  // Records the function compilation cache hits and misses.
  std::unordered_map<std::string, int> function_compilation_hits_and_misses_;

  // Coordinates cancelling ops across meshes on error. Must outlive any queued
  // async op launches, so we only reset it after seeing a failure status.
  std::unique_ptr<CancellationManager> cancellation_manager_;

  // Map each function_mesh_fingerprint (based on the set of the mesh involved)
  // to the number of times of the function execution. The
  // function_mesh_fingerprint and the counter together are used for generating
  // the step id, which is used for rendezvous creation.
  absl::flat_hash_map<uint64, uint64> func_mesh_fingerprint_to_step_counter_;
};

int64_t FingerprintShape(const absl::Span<const int64_t> shape) {
  int64_t fprint = 0;
  for (int64_t dim : shape) {
    fprint = FingerprintCat64(fprint, dim);
  }
  return fprint;
}

parallel_device::ParallelTensor* MeshWithParallelDevice::DeviceIDs(
    TFE_Context* context, TF_Status* status) const {
  if (device_ids_tensor_ == nullptr) {
    // Global device IDs sequentially increase.
    //
    // This is the assumption in the dtensor software stack. MLIR pass relies on
    // this assumption to generate mesh coordinates for each core efficiently.
    //
    // The rule to set local ids and the mapping from global ids to real
    // physical core index, e.g., TPU, is nontrivial unfortunately. It is
    // possible to set identical mapping but the collective operation
    // performance is terrible for most of cases.
    //
    // - For ICI-connected TPU slice, see go/dtensor-device-assignment-summary
    //   for guide how to create efficient core assignments toward peak
    //   performance.
    //
    //   The global id to core assignment mapping is bridged by
    //   `Mesh::tpu_core_ids()` and consumed by `UpdateTPUCompileMetadata`.
    //
    // - For DCN-connected topology, we need to map different sections of the
    //   global ids to its real physical cores separately according to the
    //   runtime requirements. For example, for a 4x32 mesh, in which the outer
    //   dimension is connected via DCN and inner dimension is connected by ICI,
    //   the device assignments for inner dimension should typically form its
    //   own ring order (not plain physical core index) in each sub-meshes and
    //   the outer dimension should be assigned according to the real physical
    //   ring of DNC hosts.
    //
    // Note: In order to change this assumption, MLIR pass needs adjustment. One
    // possible approach is to take a N-D mapping vector for N-D mesh and lookup
    // the coordinates in MLIR, by consulting tensor layout as well, rather than
    // calculation on-the-fly.

    // LINT.IfChange
    for (int64_t i = 0; i < mesh_config_.global_device_ids().size(); ++i) {
      if (mesh_config_.global_device_ids()[i] - i !=
          mesh_config_.global_device_ids()[0]) {
        TF_SetStatus(
            status, TF_INTERNAL,
            absl::StrCat("Global device IDs should be consecutive: ",
                         absl::StrJoin(mesh_config_.global_device_ids(), ", "))
                .c_str());
        return nullptr;
      }
    }
    // LINT.ThenChange(//tensorflow/dtensor/python/layout.py)

    // Local device IDs are a subset of global device IDs, arranged in device
    // ordinal order.
    std::vector<int32_t> ids;
    for (int64_t id : mesh_config_.local_device_ids()) {
      ids.push_back(id);
    }
    VLOG(1) << "Parallel device IDs: " << absl::StrJoin(ids, ", ");
    device_ids_tensor_ =
        parallel_device_->ScalarsFromSequence<int32_t>(ids, context, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  return device_ids_tensor_.get();
}

int TensorWithLayoutNumDims(void* data, TF_Status* status) {
  return reinterpret_cast<TensorWithLayout*>(data)->global_shape().size();
}

int64_t TensorWithLayoutDim(void* data, int dim_index, TF_Status* status) {
  return reinterpret_cast<TensorWithLayout*>(data)->global_shape()[dim_index];
}

void TensorWithLayoutDeallocator(void* data) {
  delete reinterpret_cast<TensorWithLayout*>(data);
}

TF_Buffer* TensorWithLayoutSummarize(void* data, TF_Status* status) {
  std::string summary =
      reinterpret_cast<TensorWithLayout*>(data)->SummarizeValue();
  return TF_NewBufferFromString(summary.data(), summary.size());
}

TFE_TensorHandle* DTensorDevice::MakeLayoutTensorHandle(
    TFE_Context* context, std::unique_ptr<TensorWithLayout> t,
    TF_Status* status) {
  TF_DataType dtype = t->dtype();
  TFE_CustomDeviceTensorHandleMethods handle_methods;
  handle_methods.num_dims = &TensorWithLayoutNumDims;
  handle_methods.dim = &TensorWithLayoutDim;
  handle_methods.deallocator = &TensorWithLayoutDeallocator;
  handle_methods.summarize = &TensorWithLayoutSummarize;
  return TFE_NewCustomDeviceTensorHandle(context, name_.c_str(), dtype,
                                         /*data=*/t.release(), handle_methods,
                                         status);
}

void DTensorDevice::RecordInShapeLayoutCache(const TensorWithLayout& tensor) {
  auto existing = shape_layout_cache_.insert(
      {FingerprintShape(tensor.global_shape()),
       CachedLayout{tensor.layout(), /*is_unique=*/true}});

  if (!existing.second) {
    // There is an entry already; if the layout doesn't match we should record
    // the fact that it's not unique.
    if (tensor.layout() != existing.first->second.layout) {
      existing.first->second.is_unique = false;
    }
  }
}

bool DTensorDevice::is_remote_mesh(const Mesh& mesh) const {
  // An empty mesh might be assigned to VarHandleOp during DTensor MLIR lowering
  // pass. Decide whether the empty mesh is remote based on the current default
  // mesh.
  return mesh.is_remote() ||
         (mesh.IsEmpty() && default_mesh_->mesh_config().is_remote());
}

StatusOr<NameAttrList> FetchAttributes(const TFE_OpAttrs* attributes) {
  // TODO(allenl): Should we just give up on the public C API to save on
  // serialization/deserialization? We need all of the attributes and to treat
  // them generically, which isn't going to be pleasant with typed attribute
  // methods.
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> serialized_attributes(
      TF_NewBuffer(), TF_DeleteBuffer);

  TF_Status* status = TF_NewStatus();
  TFE_OpAttrsSerialize(attributes, serialized_attributes.get(), status);
  if (TF_GetCode(status) == TF_OK) {
    TF_DeleteStatus(status);
  } else {
    Status failure_status = StatusFromTF_Status(status);
    TF_DeleteStatus(status);
    return failure_status;
  }

  NameAttrList name_and_attrs;
  if (!name_and_attrs.ParseFromArray(serialized_attributes->data,
                                     serialized_attributes->length)) {
    return tensorflow::errors::Unknown("Could not parse attributes");
  }
  return name_and_attrs;
}

StatusOr<Layout> FetchLayoutFromAttributes(const TFE_OpAttrs* attributes,
                                           absl::string_view attribute_name) {
  // Get attributes.
  TF_ASSIGN_OR_RETURN(NameAttrList name_and_attrs, FetchAttributes(attributes));

  // Get layout string from attributes.
  absl::string_view layout_str =
      name_and_attrs.attr().find(std::string(attribute_name))->second.s();

  // This would probably be slow at the moment without caching.
  // We should consider making this faster in the future.
  return Layout::FromString(string(layout_str));
}

std::string DTensorDevice::FetchLayout(TFE_Context* context,
                                       TFE_TensorHandle* input,
                                       TF_Status* status) {
  VLOG(1) << "Checking layout...";
  const char* input_device = TFE_TensorHandleDeviceName(input, status);
  if (input_device != name_) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "FetchLayout expects a tensor placed on the layout device.");
    return {};
  }
  TensorWithLayout* t = reinterpret_cast<TensorWithLayout*>(
      TFE_TensorHandleDevicePointer(input, status));
  if (TF_GetCode(status) != TF_OK) return {};
  return t->layout().ToString();
}

std::vector<TFE_TensorHandle*> DTensorDevice::Unpack(TFE_Context* context,
                                                     TFE_TensorHandle* input,
                                                     TF_Status* status) {
  std::vector<TFE_TensorHandle*> outputs;

  const char* input_device = TFE_TensorHandleDeviceName(input, status);
  if (TF_GetCode(status) != TF_OK) return outputs;
  if (input_device != name_) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        absl::StrCat(
            "DTensorUnpack expects a tensor placed on the DTensor device: ",
            name_, ", but input was placed on device: ", input_device)
            .c_str());
    return outputs;
  }
  TensorWithLayout* t = reinterpret_cast<TensorWithLayout*>(
      TFE_TensorHandleDevicePointer(input, status));
  if (TF_GetCode(status) != TF_OK) return outputs;

  if (is_remote_mesh(t->mesh().mesh_config())) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "DTensorUnpack is not supported on a remote mesh.");
    return outputs;
  }
  const int output_size = t->num_tensors();
  outputs.resize(output_size);

  for (int output_index = 0; output_index < output_size; ++output_index) {
    outputs[output_index] =
        TFE_TensorHandleCopySharingTensor(t->get_tensor(output_index), status);
    if (TF_GetCode(status) != TF_OK) {
      return outputs;
    }
  }
  return outputs;
}

void DTensorDevice::MaybeHandleDTensorCustomOps(
    const char* operation_name, const int num_inputs,
    const TFE_OpAttrs* attributes, TFE_Context* context,
    TFE_TensorHandle** inputs, int* num_outputs, TFE_TensorHandle** outputs,
    bool* is_custom_dtensor_op, TF_Status* status) {
  *is_custom_dtensor_op = true;
  if (operation_name == std::string("_EagerConst")) {
    // Op-by-op const has no obvious layout. DTensor skips an SPMD expansion and
    // instead relies on copy-on when the value is used later.
    std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
        TFE_NewOp(context, operation_name, status), TFE_DeleteOp);
    if (TF_GetCode(status) != TF_OK) return;
    for (int input_index = 0; input_index < num_inputs; ++input_index) {
      TFE_OpAddInput(op.get(), inputs[input_index], status);
      if (TF_GetCode(status) != TF_OK) return;
    }
    TFE_OpAddAttrs(op.get(), attributes);
    TFE_Execute(op.get(), outputs, num_outputs, status);
    return;
  }
  if (operation_name == std::string("CopyToMesh")) {
    CopyToMesh(context, num_inputs, inputs, attributes, outputs, num_outputs,
               status);
    return;
  }

  *is_custom_dtensor_op = false;
}

void DTensorDevice::CopyToMesh(TFE_Context* context, int num_inputs,
                               TFE_TensorHandle** inputs,
                               const TFE_OpAttrs* attributes,
                               TFE_TensorHandle** outputs, int* num_outputs,
                               TF_Status* status) {
  if (num_inputs != 1) {
    RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                  "DTensor CopyToMesh requires exactly 1 input.");
  }
  if (*num_outputs < 1) {
    RETURN_STATUS(status, TF_INTERNAL,
                  "DTensor CopyToMesh must have output buffer to allocate at "
                  "least 1 output.");
  }

  // Assign layout.
  StatusOr<Layout> target_layout_or =
      FetchLayoutFromAttributes(attributes, kQualifiedLayoutAttr);
  if (!target_layout_or.ok()) {
    RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                  "DTensor CopyToMesh requires valid layout attribute for "
                  "destination DTensor.");
  }

  const Layout target_layout = *target_layout_or;
  const Mesh& target_mesh = target_layout.mesh();

  // TODO(b/193443769): Support sharded layout for eager copy to mesh.
  if (!target_layout.IsFullyReplicated()) {
    RETURN_STATUS(status, TF_UNIMPLEMENTED,
                  "Target layout of DTensor CopyToMesh must be replicated. "
                  "Consider changing the target layout to replicated layout or "
                  "file a bug to the DTensor team (b/193443769).");
  }

  TFE_TensorHandle* input_tensor = inputs[0];

  // Check that if input tensor is DTensor, then input layout of the DTensor
  // must be replicated.
  const char* input_device = TFE_TensorHandleDeviceName(input_tensor, status);
  if (TF_GetCode(status) != TF_OK) return;

  if (name_ == input_device) {
    // Handle input which is on DTensor device already.
    TensorWithLayout* t = reinterpret_cast<TensorWithLayout*>(
        TFE_TensorHandleDevicePointer(input_tensor, status));
    if (TF_GetCode(status) != TF_OK) return;

    if (!t->layout().IsFullyReplicated())
      RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                    "Input tensor to CopyToMesh must be replicated DTensor or "
                    "normal eager Tensor.");

    // If input to CopyToMesh is a DTensor, we use the first local tensor as
    // input tensor handle to invoke copy.
    input_tensor = t->get_tensor(0);
  }

  auto it = mesh_to_device_map_.find(target_mesh);
  if (it == mesh_to_device_map_.end()) {
    RETURN_STATUS(
        status, TF_INTERNAL,
        "DTensor CopyToMesh target mesh is not registered. Meshes should be "
        "automatically registered. Please file a bug. (component id: 833864)");
  }

  const MeshWithParallelDevice* target_parallel_mesh = it->second.get();

  // Broadcast non-dtensor value to dtensor.
  std::unique_ptr<TensorWithLayout> wrapper = TensorWithLayout::Broadcast(
      context, input_tensor, *target_parallel_mesh, name_, status);
  if (TF_GetCode(status) != TF_OK) return;

  RecordInShapeLayoutCache(*wrapper);
  *num_outputs = 1;
  *outputs = MakeLayoutTensorHandle(context, std::move(wrapper), status);
}

namespace {

// Verifies that all components have the same dtype and shape.
// The component shape will be set upon success.
void VerifyPackTensorShapeAndDtype(
    std::vector<parallel_device::TensorHandlePtr>& components,
    std::vector<int64_t>* component_shape, TF_Status* status) {
  TF_DataType dtype = TFE_TensorHandleDataType(components[0].get());
  auto size = TFE_TensorHandleNumDims(components[0].get(), status);
  if (TF_GetCode(status) != TF_OK) return;
  component_shape->clear();
  component_shape->reserve(size);
  for (int i = 0; i < size; ++i) {
    component_shape->push_back(
        TFE_TensorHandleDim(components[0].get(), i, status));
    if (TF_GetCode(status) != TF_OK) return;
  }

  // Verify that the TensorHandle's shape and dtype match all of the component
  // shapes and dtypes.
  for (const auto& component : components) {
    for (int i = 0; i < component_shape->size(); ++i) {
      int64_t tensor_dim = TFE_TensorHandleDim(component.get(), i, status);
      if (TF_GetCode(status) != TF_OK) return;
      if (tensor_dim != (*component_shape)[i]) {
        TF_SetStatus(status, TF_UNIMPLEMENTED,
                     "Components of a PackedTensor must currently all have "
                     "the same shape");
        return;
      }
      if (TFE_TensorHandleDataType(component.get()) != dtype) {
        TF_SetStatus(status, TF_INTERNAL,
                     "Components of a PackedTensor must all have "
                     "the same dtype");
        return;
      }
    }
  }
}

// Verifies that all TensorHandles have rank `rank` of dtype `dtype`.
void VerifyTensorRankAndDType(TFE_TensorHandle** tensors, int num_input,
                              int expected_rank, TF_DataType* expected_dtype,
                              TF_Status* status) {
  for (int i = 0; i < num_input; ++i) {
    auto actual_rank = TFE_TensorHandleNumDims(tensors[i], status);
    if (TF_GetCode(status) != TF_OK)
      RETURN_STATUS(status, TF_INTERNAL, "Error getting rank of tensor.");
    if (actual_rank != expected_rank)
      RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                    "Rank of tensor did not match the expected rank.");
    if (expected_dtype != nullptr &&
        TFE_TensorHandleDataType(tensors[i]) != *expected_dtype)
      RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                    "Dtype of tensor did not match the expected dtype.");
  }
}

}  // namespace

TFE_TensorHandle* DTensorDevice::Pack(TFE_Context* context, int num_inputs,
                                      TFE_TensorHandle** inputs,
                                      const std::string& string_layout,
                                      TF_Status* status) {
  if (num_inputs < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "DTensorPack requires 1 or more inputs");
    return nullptr;
  }
  StatusOr<Layout> target_layout = Layout::FromString(string_layout);
  if (!target_layout.ok()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Failed to parse layout from string layout");
    return nullptr;
  }
  const Mesh& target_mesh = target_layout->mesh();
  const MeshWithParallelDevice* target_parallel_device =
      mesh_to_device_map_[target_mesh].get();
  if (target_parallel_device == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 absl::StrCat("Required mesh : ", target_mesh.ToString(),
                              "is not registered with DTensor ")
                     .c_str());
    return nullptr;
  }

  std::unique_ptr<TensorWithLayout> packed_tensor;
  if (is_remote_mesh(target_parallel_device->mesh_config())) {
    // Create a dummy output for DTensorPack if inputs are on a remote mesh.
    TF_DataType dtype = TFE_TensorHandleDataType(inputs[0]);
    auto size = TFE_TensorHandleNumDims(inputs[0], status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    std::vector<int64_t> component_shape;
    component_shape.reserve(size);
    for (int i = 0; i < size; ++i) {
      component_shape.push_back(TFE_TensorHandleDim(inputs[0], i, status));
      if (TF_GetCode(status) != TF_OK) return nullptr;
    }
    packed_tensor = TensorWithLayout::Dummy(
        component_shape, dtype, *target_parallel_device, *target_layout);

  } else {
    auto local_devices = target_parallel_device->mesh_config().local_devices();

    if (num_inputs !=
        target_parallel_device->parallel_device().num_underlying_devices()) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   absl::StrCat("The dtensor device ", name_, " expected ",
                                local_devices.size(),
                                " inputs to DTensorPack, but got ", num_inputs)
                       .c_str());
      return nullptr;
    }

    std::vector<parallel_device::TensorHandlePtr> components;
    components.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      TFE_TensorHandle* input = inputs[i];
      const char* input_device = TFE_TensorHandleDeviceName(input, status);
      if (TF_GetCode(status) != TF_OK) return nullptr;
      if (name_ == input_device) {
        TF_SetStatus(status, TF_INVALID_ARGUMENT,
                     "Does not support packing a Tensor that is already on "
                     "dtensor device");
        return nullptr;
      }
      // If `input` is on the target device, this creates a new handle sharing
      // the underlying data; otherwise, async copies are invoked.
      components.emplace_back(TFE_TensorHandleCopyToDevice(
          input, context, local_devices[i].c_str(), status));
      if (TF_GetCode(status) != TF_OK) return nullptr;
    }

    std::vector<int64_t> component_shape;
    VerifyPackTensorShapeAndDtype(components, &component_shape, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;

    std::unique_ptr<parallel_device::ParallelTensor> parallel_tensor =
        parallel_device::ParallelTensor::FromTensorHandles(
            target_parallel_device->parallel_device(), std::move(components),
            status);
    if (TF_GetCode(status) != TF_OK) return nullptr;

    if (target_layout->rank() != component_shape.size()) {
      TF_SetStatus(
          status, TF_INVALID_ARGUMENT,
          absl::StrCat(
              "Packed layout should have the same rank as the rank for each "
              "component. The rank of each component is: ",
              component_shape.size(),
              ", while layout has rank: ", target_layout->rank(),
              "\nLayout: ", target_layout->ToString(), "\n")
              .c_str());
      return nullptr;
    }

    packed_tensor =
        TensorWithLayout::Wrap(std::move(parallel_tensor),
                               *target_parallel_device, *target_layout)
            .value();
  }

  RecordInShapeLayoutCache(*packed_tensor);
  TFE_TensorHandle* output =
      MakeLayoutTensorHandle(context, std::move(packed_tensor), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return output;
}

TFE_TensorHandle* DTensorDevice::SparsePack(
    TFE_Context* context, int num_inputs, TFE_TensorHandle** indices,
    TFE_TensorHandle** values, TFE_TensorHandle** shapes,
    const std::string& string_layout, TF_Status* status) {
  StatusOr<Layout> target_layout = Layout::FromString(string_layout);
  if (!target_layout.ok()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Failed to parse layout from string layout");
    return nullptr;
  }
  const Mesh& target_mesh = target_layout->mesh();
  const MeshWithParallelDevice* target_parallel_device =
      mesh_to_device_map_[target_mesh].get();
  if (target_parallel_device == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 absl::StrCat("Required mesh : ", target_mesh.ToString(),
                              "is not registered with DTensor ")
                     .c_str());
    return nullptr;
  }

  TF_DataType tf_int64 = TF_INT64;
  // Verify rank and dtype of shapes.
  VerifyTensorRankAndDType(shapes, num_inputs, /*expected_rank=*/1,
                           /*expected_dtype=*/&tf_int64, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Verify rank and dtype of indices.
  VerifyTensorRankAndDType(indices, num_inputs, /*expected_rank=*/2,
                           /*expected_dtype=*/&tf_int64, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Verify rank of values.
  VerifyTensorRankAndDType(values, num_inputs, /*expected_rank=*/1,
                           /*expected_dtype=*/nullptr, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Compute the local shape from a shape tensor.
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> shape_tensor(
      TFE_TensorHandleResolve(shapes[0], status), TF_DeleteTensor);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(
        status, TF_GetCode(status),
        absl::StrCat("Error resolving the tensor handle of shape tensor"
                     ". Original message: ",
                     TF_Message(status))
            .c_str());
    return nullptr;
  }
  int shape_tensor_size = TFE_TensorHandleDim(shapes[0], 0, status);
  if (TF_GetCode(status) != TF_OK || shape_tensor_size <= 0) {
    TF_SetStatus(status, TF_GetCode(status),
                 absl::StrCat("Error computing the num dims of shape tensor",
                              TF_Message(status))
                     .c_str());
    return nullptr;
  }

  const int64_t* data =
      static_cast<int64_t*>(TF_TensorData(shape_tensor.get()));
  std::vector<int64_t> local_shape(data, data + shape_tensor_size);
  if (local_shape.size() != target_layout->rank()) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        absl::StrCat(
            "Packed layout should have the same rank as the rank for each "
            "component. The rank of each component is: ",
            local_shape.size(),
            ", while layout has rank: ", target_layout->rank(),
            "\nLayout: ", target_layout->ToString(), "\n")
            .c_str());
    return nullptr;
  }

  // Create the SparseTensorWithLayout.
  std::unique_ptr<TensorWithLayout> packed_tensor;
  if (is_remote_mesh(target_parallel_device->mesh_config())) {
    // Create a dummy SparseTensorWithLayout.
    packed_tensor = SparseTensorWithLayout::Dummy(
        local_shape, *target_parallel_device, target_layout.value());
  } else {
    // Parse the indices, values, and dense_shape tensors and put them into
    // parallel tensors, and then pack it into a single SparseTensorWithLayout.
    auto local_devices = target_parallel_device->mesh_config().local_devices();

    std::vector<parallel_device::TensorHandlePtr> indices_components;
    std::vector<parallel_device::TensorHandlePtr> values_components;
    std::vector<parallel_device::TensorHandlePtr> dense_shapes_components;

    // Just a nice trick to make code cleaner to pack each of indices, values,
    // shapes.
    std::vector<std::vector<parallel_device::TensorHandlePtr>*> components{
        &indices_components, &values_components, &dense_shapes_components};
    std::vector<TFE_TensorHandle**> input_vectors{indices, values, shapes};
    for (int component_index = 0; component_index < 3; ++component_index) {
      components[component_index]->reserve(num_inputs);
      TFE_TensorHandle** inputs = input_vectors[component_index];
      for (int i = 0; i < num_inputs; ++i) {
        const char* input_device =
            TFE_TensorHandleDeviceName(inputs[i], status);
        if (TF_GetCode(status) != TF_OK) return nullptr;
        if (name_ == input_device) {
          TF_SetStatus(status, TF_INVALID_ARGUMENT,
                       "Does not support packing a Tensor that is already on "
                       "dtensor device.");
          return nullptr;
        }

        components[component_index]->emplace_back(TFE_TensorHandleCopyToDevice(
            inputs[i], context, local_devices[i].c_str(), status));
        if (TF_GetCode(status) != TF_OK) return nullptr;
      }
    }
    std::unique_ptr<parallel_device::ParallelTensor> parallel_indices_tensor =
        parallel_device::ParallelTensor::FromTensorHandles(
            target_parallel_device->parallel_device(),
            std::move(indices_components), status);

    std::unique_ptr<parallel_device::ParallelTensor> parallel_values_tensor =
        parallel_device::ParallelTensor::FromTensorHandles(
            target_parallel_device->parallel_device(),
            std::move(values_components), status);

    std::unique_ptr<parallel_device::ParallelTensor>
        parallel_dense_shapes_tensor =
            parallel_device::ParallelTensor::FromTensorHandles(
                target_parallel_device->parallel_device(),
                std::move(dense_shapes_components), status);

    if (TF_GetCode(status) != TF_OK) return nullptr;
    packed_tensor =
        SparseTensorWithLayout::Wrap(std::move(parallel_indices_tensor),
                                     std::move(parallel_values_tensor),
                                     std::move(parallel_dense_shapes_tensor),
                                     *target_parallel_device,
                                     target_layout.value(), local_shape)
            .value();
  }

  RecordInShapeLayoutCache(*packed_tensor);
  TFE_TensorHandle* output =
      MakeLayoutTensorHandle(context, std::move(packed_tensor), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return output;
}

bool DTensorDevice::IsSparseDTensor(TFE_Context* context,
                                    TFE_TensorHandle* input,
                                    TF_Status* status) {
  const char* input_device = TFE_TensorHandleDeviceName(input, status);
  if (input_device != name_) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        "DTensorSparseUnpack expects a tensor placed on the DTensor device.");
    return false;
  }
  TensorWithLayout* t = reinterpret_cast<TensorWithLayout*>(
      TFE_TensorHandleDevicePointer(input, status));
  if (TF_GetCode(status) != TF_OK) return false;
  return t->tensor_type() == TensorType::kSparse;
}

void DTensorDevice::UpdateOutputLayoutsWithSameShapePolicy(
    const std::vector<PartialTensorShape>& global_output_shapes,
    const absl::flat_hash_set<Mesh>& input_meshes, absl::string_view op_name,
    tensorflow::Graph* graph, std::vector<const Layout*>* output_layouts,
    TF_Status* status) {
  if (!same_shape_policy_enabled_) return;
  // Simply do not hint if inputs span across multiple meshes.
  if (input_meshes.size() > 1) return;

  for (Node* node : graph->op_nodes()) {
    if (!node->IsRetval()) {
      continue;
    }
    int output_index;
    RETURN_C_STATUS_IF_NOT_OK(
        GetNodeAttr(node->attrs(), "index", &output_index), status);
    if (output_layouts->at(output_index)) {
      continue;
    }

    const auto& global_output_shape = global_output_shapes.at(output_index);
    const Layout* layout = nullptr;
    // TODO(b/180022708): This is useful information, we should be
    // able to hint to layout propagation without making it a hard
    // requirement
    //
    // Special cases at the moment:
    // - Relayout needs an exemption.
    // - VarHandleOp does not need hint. VarHandleOp has scalar shape so layout
    //   is trivial. On the other hande, downstream system "thinks' Variable has
    //   shape same as the pointing value. So, providing a layout based on
    //   VarHandleOp (scalar) might confuse the downstream system.
    if (op_name != std::string("Relayout") &&
        op_name != std::string("VarHandleOp")) {
      // TODO(b/162009702): Support matching between partially-known shapes.
      if (global_output_shape.IsFullyDefined()) {
        gtl::InlinedVector<int64, 4> shape_vector(
            global_output_shape.dim_sizes());
        auto layout_iterator =
            shape_layout_cache_.find(FingerprintShape(shape_vector));
        if (layout_iterator != shape_layout_cache_.end() &&
            layout_iterator->second.is_unique) {
          // We have a cached layout for this shape. Send it to MLIR.
          layout = &layout_iterator->second.layout;
          VLOG(3) << op_name << ": found a cached layout for shape "
                  << global_output_shape.DebugString() << ": \""
                  << layout->ToString() << "\"";
          if (input_meshes.empty() &&
              layout->mesh() != default_mesh_->mesh_config()) {
            VLOG(3) << "But we can't infer a input mesh and cached layout: "
                    << "mesh \"" << (layout->mesh().ToString()) << " "
                    << "is different than the default mesh : \""
                    << default_mesh_->mesh_config().ToString() << "\"\n"
                    << "Not applying the cached layout.";
          } else if (!input_meshes.empty() &&
                     layout->mesh() != *input_meshes.begin()) {
            VLOG(3)
                << "But the layout mesh is different than the executing mesh: "
                << "\"" << (*input_meshes.begin()).ToString() << "\"\n"
                << "Not applying the cached layout.";
          } else {
            (*output_layouts)[output_index] = layout;
            node->AddAttr(kDefaultLayoutAttr, layout->ToString());
          }
        } else if (layout_iterator == shape_layout_cache_.end()) {
          VLOG(3) << op_name << ": no cached layout found for shape "
                  << global_output_shape.DebugString();
        } else {
          VLOG(3) << op_name << ": found multiple layouts for shape "
                  << global_output_shape.DebugString();
        }
      } else {
        VLOG(3) << op_name
                << ": not applying same-shape-same-layout due to "
                   "not-fully-known shape "
                << global_output_shape.DebugString();
      }
    }
  }
}

std::unordered_map<std::string, int>
DTensorDevice::GetFunctionCacheHitAndMissCount(TFE_Context* context,
                                               TF_Status* status) const {
  return function_compilation_hits_and_misses_;
}

// From `graph` containing computation for all meshes, extract/select
// computation for mesh specified in `function`. Returned graph is a cloned
// graph with ops only for single mesh execution.
StatusOr<std::unique_ptr<Graph>> SelectGraphToExecute(
    const TranslatedFunction& function, const Graph& graph,
    std::string* stateful_partitioned_call_name) {
  auto new_graph = std::make_unique<Graph>(graph.flib_def());
  CopyGraph(graph, new_graph.get());
  std::vector<Node*> arg_nodes;
  std::vector<Node*> retval_nodes;
  for (Node* node : new_graph->nodes()) {
    if (node->IsArg()) arg_nodes.emplace_back(node);
    if (node->IsRetval()) retval_nodes.emplace_back(node);
  }

  // Remove irrelevant function calls.
  for (Node* node : new_graph->nodes()) {
    if (node->op_def().name() != "StatefulPartitionedCall") continue;

    if (node->name() != function.node_to_execute->name()) {
      // Remove function call that does not match mesh specification and all
      // output retval nodes connected to the function call node.
      std::queue<Node*> nodes_to_remove;
      nodes_to_remove.push(node);
      while (!nodes_to_remove.empty()) {
        Node* n = nodes_to_remove.front();
        for (const Edge* out_edge : n->out_edges()) {
          if (out_edge->IsControlEdge()) continue;
          Node* out_node = out_edge->dst();
          if (!out_node->IsSink()) nodes_to_remove.push(out_node);
        }
        if (n->IsRetval()) {
          auto pos = std::find(retval_nodes.begin(), retval_nodes.end(), n);
          TF_RET_CHECK(pos != retval_nodes.end());
          retval_nodes.erase(pos);
        }
        nodes_to_remove.pop();
        new_graph->RemoveNode(n);
      }
    }
  }

  *stateful_partitioned_call_name = function.node_to_execute->name();
  VLOG(1) << "Selected call " << *stateful_partitioned_call_name;

  // Remove unused arg nodes in graph.
  for (auto it = arg_nodes.begin(); it != arg_nodes.end(); it++) {
    Node* arg_node = *it;
    bool arg_unused = true;
    for (const Edge* e : arg_node->out_edges()) {
      if (e->dst()->IsOp()) {
        arg_unused = false;
      }
    }
    if (!arg_unused) continue;

    new_graph->RemoveNode(arg_node);
    arg_nodes.erase(it--);
  }

  // Reset index attributes for arg and retval nodes.
  for (Node* n : new_graph->nodes()) {
    // Reset arg node index attributes to its position within all the arg
    // nodes. This should just be increasing from 0 to n where n
    // is the total number of arguments. Note that this definition to
    // the `index` attribute is different from the definition we set in
    // PrepareGraphForMLIR.
    // This attribute is needed for each arg node when converting a Graph to
    // a FunctionDef.
    if (n->IsArg()) {
      auto pos = std::find(arg_nodes.begin(), arg_nodes.end(), n);
      TF_RET_CHECK(pos != arg_nodes.end());
      const int new_index = std::distance(arg_nodes.begin(), pos);
      n->AddAttr("index", new_index);
    }

    // Reset retval nodes index attributes.
    if (n->IsRetval()) {
      auto retval_pos = std::find(retval_nodes.begin(), retval_nodes.end(), n);
      TF_RET_CHECK(retval_pos != retval_nodes.end());
      const int new_index = std::distance(retval_nodes.begin(), retval_pos);
      n->AddAttr("index", new_index);
    }
  }

  VLOG(4) << tensorflow::DumpGraphToFile("selected_graph_to_execute_",
                                         *new_graph);

  return new_graph;
}

// Adds processed graph to run for each mesh computation in
// `execution_functions` to function definition library.
Status AddExecutionFunctionDefsToFunctionDefLibrary(
    const absl::flat_hash_set<Node*>& control_ret_nodes, TFE_Context* context,
    const Graph& graph, ExecutionFunctions* execution_functions) {
  // Note: We use node name instead of node pointer for comparison because
  // node address in the new graph is different with the original graph.
  absl::flat_hash_set<std::string> control_ret_names;
  for (auto* n : control_ret_nodes) {
    control_ret_names.emplace(n->name());
  }
  for (TranslatedFunction& function : execution_functions->function_list) {
    std::string selected_call_node_name;
    // TODO(bfontain): We should just try to call the functions directly rather
    // than wrap
    // Construct graph that executes only computation for `function`.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Graph> new_graph,
        SelectGraphToExecute(function, graph, &selected_call_node_name));
    VLOG(4) << tensorflow::DumpGraphToFile("selected_graph_", *new_graph);

    // Add unique identifier based on the function we are executing to the
    // function/graph and convert graph to functiondef.
    NameAttrList func;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(function.node_to_execute->attrs(), "f", &func));

    static std::atomic<int64_t> unique_function_number(0);
    function.translated_function_name =
        absl::StrCat(func.name(), "_", unique_function_number.fetch_add(1));
    auto control_ret_node_names =
        [&control_ret_names, &selected_call_node_name](
            const Node* node) -> absl::optional<std::string> {
      // Add the stateful partitioned call node as a control return as we need
      // to process any control deps inside the inner function.
      if (control_ret_names.contains(node->name()) ||
          node->name() == selected_call_node_name) {
        return node->name();
      }
      return absl::nullopt;
    };

    tensorflow::FunctionDef to_run;
    TF_RETURN_IF_ERROR(tensorflow::GraphToFunctionDef(
        *new_graph, function.translated_function_name, control_ret_node_names,
        &to_run));

    for (const auto& out : to_run.signature().output_arg()) {
      function.output_dtypes.emplace_back(static_cast<TF_DataType>(out.type()));
    }

    AddDTensorFunctionAttr(to_run);
    TF_RETURN_IF_ERROR(tensorflow::unwrap(context)->AddFunctionDef(to_run));
  }

  return OkStatus();
}

void DTensorDevice::LowerToSPMDFunction(
    TFE_Context* context, const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation, const TFE_OpAttrs* attributes,
    const int num_outputs, const ExecutionFunctions** execution_functions,
    TF_Status* status) {
  profiler::TraceMe activity(
      [&] { return "DTensorDevice::LowerToSPMDFunction"; },
      profiler::TraceMeLevel::kInfo);
  FunctionLibraryDefinition* flib_def =
      tensorflow::unwrap(context)->FuncLibDef();
  auto graph(std::make_unique<tensorflow::Graph>(flib_def));
  NameAttrList eager_attributes;
  ASSIGN_OR_RETURN_C_STATUS(eager_attributes, FetchAttributes(attributes),
                            status);

  std::vector<PartialTensorShape> global_output_shapes;
  std::vector<const Layout*> output_layouts;
  const FunctionDef* function_def = doperation.function_def;
  if (!function_def) {
    // Output layouts of an eager op (e.g. fill) must be inferred before cache
    // key computation, since they might depend on the current DTensorDevice
    // state.
    Status s = PrepareGraphForMlir(
        function_manager_, inputs, doperation, *flib_def, eager_attributes,
        default_layout_, graph.get(), &global_output_shapes, &output_layouts);
    RETURN_C_STATUS_IF_NOT_OK(s, status);

    // Finds all meshes the inputs are lied on.
    absl::flat_hash_set<Mesh> input_meshes;
    for (const TensorWithLayout* tensor : inputs) {
      if (!tensor->layout().mesh().IsEmpty()) {
        input_meshes.insert(tensor->layout().mesh());
      }
    }
    // Currently we only provide layout hints for op-by-op, since
    // they interact badly with layout propagation.
    UpdateOutputLayoutsWithSameShapePolicy(global_output_shapes, input_meshes,
                                           doperation.name, graph.get(),
                                           &output_layouts, status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  std::pair<tensorflow::Fprint128, const ExecutionFunctions*>
      cache_key_and_func = function_manager_.GetCachedFunction(
          doperation, eager_attributes, inputs, output_layouts);
  *execution_functions = cache_key_and_func.second;
  if (*execution_functions != nullptr) {
    function_compilation_hits_and_misses_["hit"]++;
    return;
  } else if (function_def) {
    function_compilation_hits_and_misses_["miss"]++;
    LOG(INFO) << "DTensor cache key lookup missed for " << doperation.name
              << ". DTensor is (re-)computing its SPMD transformation.";
  }

  // It includes remote devices when the coordination service is enabled.
  const auto device_list = tensorflow::unwrap(context)->ListAllTfDevices();
  DeviceSet device_set;
  for (const auto device : device_list) device_set.AddDevice(device);

  if (function_def) {
    ASSIGN_OR_RETURN_C_STATUS(auto device_name_to_mesh_device,
                              PipelineSubMeshes(context), status);
    const bool is_pipelining_function = !device_name_to_mesh_device.empty();
    // For a multi-mesh function for pipelining, we take a different execution
    // path. Call the partitioner to lower and partition the graph into multiple
    // sub functions to execute (one per sub mesh).
    if (is_pipelining_function) {
      ASSIGN_OR_RETURN_C_STATUS(
          ExecutionFunctions functions,
          PipeliningPartitionerRun(&device_name_to_mesh_device, flib_def,
                                   &pass_runner_, *doperation.function_def,
                                   eager_attributes, inputs, device_set,
                                   num_outputs),
          status);
      *execution_functions = function_manager_.AddCachedFunction(
          doperation, cache_key_and_func.first, std::move(functions));
      return;
    }
    // Output layouts of a function are inferred by MLIR lowering. They are
    // not necessary for cache key computation, so run PrepareGraphForMlir after
    // cache key computation to reduce the overheads of running the same
    // function multiple times.
    Status s = PrepareGraphForMlir(
        function_manager_, inputs, doperation, *flib_def, eager_attributes,
        default_layout_, graph.get(), &global_output_shapes, &output_layouts);
    RETURN_C_STATUS_IF_NOT_OK(s, status);
  }

  absl::flat_hash_set<Node*> control_ret_nodes;
  // Run DTensor MLIR passes that convert input graph to SPMD version.
  {
    profiler::TraceMe activity([&] { return "DTensorDevice::RunMLIRPasses"; },
                               profiler::TraceMeLevel::kInfo);
    RETURN_C_STATUS_IF_NOT_OK(
        pass_runner_.RunOnGraph(device_set, doperation.is_func(), flib_def,
                                &graph, control_ret_nodes,
                                cache_key_and_func.first),
        status);
  }
  VLOG(4) << tensorflow::DumpGraphToFile("after_mlir_spmd_lowering", *graph,
                                         flib_def);
  if (flib_def->Contains(kLoadEmbeddingFn)) {
    Status s = InsertFunctionForTPUEmbeddingCheckpoint(
        status, graph.get(), inputs, kLoadEmbeddingFn);
    RETURN_C_STATUS_IF_NOT_OK(s, status);
  }

  // After MLIR transformations, exactly one StatefulPartitionedCall op is
  // returned for mesh cluster in computation. Identity all functions to execute
  // for each mesh and relevant input and output information.
  ASSIGN_OR_RETURN_C_STATUS(
      ExecutionFunctions functions,
      IdentifyAllFunctionsToExecute(*graph.get(), global_output_shapes),
      status);

  // In order to ensure that all resource assign operations as well as side
  // effecting ops are executed, we add identity ops before function outputs
  // with control rets.
  RETURN_C_STATUS_IF_NOT_OK(MaybeInsertIdentityNodes(function_def, graph.get()),
                            status);

  VLOG(4) << tensorflow::DumpGraphToFile("after_post_processing_graph", *graph,
                                         flib_def);

  RETURN_C_STATUS_IF_NOT_OK(
      AddExecutionFunctionDefsToFunctionDefLibrary(control_ret_nodes, context,
                                                   *graph.get(), &functions),
      status);
  functions.num_device_ids = 1;
  if (function_def) {
    for (TranslatedFunction& function : functions.function_list) {
      functions.function_mesh_fingerprint =
          FingerprintCat64(functions.function_mesh_fingerprint,
                           function.function_mesh.GlobalFingerprint());
    }
  }

  *execution_functions = function_manager_.AddCachedFunction(
      doperation, cache_key_and_func.first, std::move(functions));
}

void DTensorDevice::ExecuteFunctionAndWait(
    TFE_Context* context, const TranslatedFunction* function_ptr,
    const MeshWithParallelDevice* parallel_device_mesh,
    const std::vector<parallel_device::ParallelTensor*>& parallel_inputs,
    const int64_t step_id, const TFE_OpAttrs* attributes, TF_Status* status) {
  const std::string mesh_str = function_ptr->function_mesh.ToString();
  VLOG(4) << "Launching computation for mesh : " << mesh_str;
  parallel_device_mesh->parallel_device().StartExecute(
      context,
      /*inputs=*/parallel_inputs,
      /*operation_name=*/function_ptr->translated_function_name.c_str(),
      /*attributes=*/attributes,
      /*expected_max_outputs=*/function_ptr->local_output_shapes.size(),
      /*cancellation_manager=*/*cancellation_manager_,
      /*step_id=*/step_id);

  VLOG(4) << "Joining computation result from mesh : " << mesh_str;
  parallel_device_mesh->parallel_device().Join(
      function_ptr->local_output_shapes, status);
  VLOG(4) << "Joining status: " << TF_Message(status);
  if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_CANCELLED) {
    LOG(ERROR) << "Encountered error while executing function: "
               << function_ptr->translated_function_name
               << " for mesh : " << mesh_str
               << " / error : " << TF_Message(status);
  }

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> async_wait_status(
      TF_NewStatus(), TF_DeleteStatus);
  AsyncWait(context, async_wait_status.get());
  TF_Code error_code = TF_GetCode(async_wait_status.get());
  if (error_code != TF_OK && error_code != TF_CANCELLED) {
    LOG(ERROR) << "Async status: " << TF_Message(async_wait_status.get());
  }
}

void DTensorDevice::ExecuteRegularOperation(
    TFE_Context* context, const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation, const TFE_OpAttrs* attributes,
    int* num_outputs, TFE_TensorHandle** outputs, TF_Status* status) {
  const ExecutionFunctions* execution_functions = nullptr;

  LowerToSPMDFunction(context, inputs, doperation, attributes, *num_outputs,
                      &execution_functions, status);
  if (TF_GetCode(status) != TF_OK) return;

  // Update input layouts for resource arguments.
  for (const TranslatedFunction& function :
       execution_functions->function_list) {
    for (const auto& entry : function.resource_input_layouts) {
      // TODO(hthu): Add an TensorWithLayout in the inputs vector at location 0
      // for DeviceId. This is done as the first arg is always DeviceId, and it
      // isn't mapped to input Tensors.
      const int resource_index_to_update = entry.first - 1;
      inputs[resource_index_to_update]->UpdateLayout(entry.second, status);
      if (TF_GetCode(status) != TF_OK) {
        RETURN_STATUS(status, TF_GetCode(status),
                      absl::StrCat("Attempt to update layout input arg: ",
                                   resource_index_to_update,
                                   ". Original message: ", TF_Message(status))
                          .c_str());
      }
    }
  }

  int num_global_outputs = 0;

  std::map<std::string, const MeshWithParallelDevice*>
      function_name_and_mesh_mapping;
  absl::flat_hash_set<std::string> excluded_fn_names;
  std::unique_ptr<const TranslatedFunction> epu_fn_ptr, load_embedding_ptr;
  for (const TranslatedFunction& function :
       execution_functions->function_list) {
    StatusOr<Mesh> maybe_converted_mesh = function.function_mesh;
    if (function.function_mesh.is_epu_mesh()) {
      maybe_converted_mesh = function.function_mesh.ToDeviceType("CPU");
    }

    if (!maybe_converted_mesh.ok()) {
      RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                    absl::StrCat("Failed to convert mesh, get error: ",
                                 maybe_converted_mesh.status().error_message())
                        .c_str());
    }
    const Mesh& mesh = *maybe_converted_mesh;
    const MeshWithParallelDevice* parallel_device_mesh =
        mesh_to_device_map_.contains(mesh) ? mesh_to_device_map_[mesh].get()
                                           : default_mesh_;
    if (parallel_device_mesh == nullptr) {
      RETURN_STATUS(status, TF_INTERNAL,
                    "required mesh is not registered with DTensor device");
    }
    function_name_and_mesh_mapping[function.translated_function_name] =
        parallel_device_mesh;

    if (function.function_mesh.is_epu_mesh()) {
      if (epu_fn_ptr != nullptr) {
        RETURN_STATUS(status, TF_INTERNAL,
                      "There are more than one function defined on EPU mesh.");
      }
      epu_fn_ptr = std::make_unique<const TranslatedFunction>(function);
      excluded_fn_names.insert(function.translated_function_name);
    }
    if (absl::StartsWith(function.translated_function_name, kLoadEmbeddingFn)) {
      if (load_embedding_ptr != nullptr) {
        RETURN_STATUS(status, TF_INTERNAL,
                      "There are more than one function defined on EPU mesh.");
      }
      load_embedding_ptr = std::make_unique<const TranslatedFunction>(function);
      excluded_fn_names.insert(function.translated_function_name);
    }
  }

  // Compute the step_id based on the function_mesh_fingerprint and the
  // corresponding function execution counter.
  uint64 function_mesh_fingerprint =
      execution_functions->function_mesh_fingerprint;
  if (func_mesh_fingerprint_to_step_counter_.contains(
          function_mesh_fingerprint)) {
    func_mesh_fingerprint_to_step_counter_.at(function_mesh_fingerprint)++;
  } else {
    func_mesh_fingerprint_to_step_counter_.insert(
        {function_mesh_fingerprint, 0});
  }
  const uint64 step_id = FingerprintCat64(
      function_mesh_fingerprint,
      func_mesh_fingerprint_to_step_counter_.at(function_mesh_fingerprint));

  // Execute excluded functions in sequence.
  if (epu_fn_ptr != nullptr) {
    ExecuteFunctionAndWait(
        context,
        /*function_ptr=*/epu_fn_ptr.get(),
        /*parallel_device_mesh=*/
        function_name_and_mesh_mapping[epu_fn_ptr->translated_function_name],
        /*parallel_inputs=*/{}, /*step_id=*/step_id, /*attributes=*/attributes,
        /*status=*/status);
  }

  if (load_embedding_ptr != nullptr) {
    StatusOr<std::vector<parallel_device::ParallelTensor*>> parallel_inputs =
        PrepareEmbeddingInputs(inputs);
    if (!parallel_inputs.ok()) {
      RETURN_STATUS(status, TF_INTERNAL,
                    parallel_inputs.status().error_message().c_str());
    }
    ExecuteFunctionAndWait(
        context,
        /*function_ptr=*/load_embedding_ptr.get(),
        /*parallel_device_mesh=*/
        function_name_and_mesh_mapping[load_embedding_ptr
                                           ->translated_function_name],
        /*parallel_inputs=*/*parallel_inputs, /*step_id=*/step_id,
        /*attributes=*/attributes, /*status=*/status);
  }

  // Extract the global parallel inputs and flatten SparseTensors
  // into the three component tensors.
  std::vector<parallel_device::ParallelTensor*> global_parallel_inputs;
  std::vector<parallel_device::ParallelTensor*> global_parallel_sparse_inputs;
  absl::flat_hash_set<int> global_sparse_input_indices;
  for (auto input : inputs) {
    if (input->tensor_type() == TensorType::kSparse) {
      SparseTensorWithLayout* sparse_input =
          dynamic_cast<SparseTensorWithLayout*>(input);
      global_parallel_sparse_inputs.push_back(sparse_input->indices());
      global_parallel_sparse_inputs.push_back(sparse_input->dense_shapes());
      global_parallel_sparse_inputs.push_back(sparse_input->values());
    } else {
      global_parallel_inputs.push_back(input->tensor());
    }
  }
  // Insert SparseTensor components to the end, this is because
  // in the MLIR handling of SparseTensors, we place SparseTensor components
  // to the end of the main func arguments for a fixed ordering.
  global_parallel_inputs.insert(global_parallel_inputs.end(),
                                global_parallel_sparse_inputs.begin(),
                                global_parallel_sparse_inputs.end());

  // Execute all functions in parallel.
  for (const TranslatedFunction& function :
       execution_functions->function_list) {
    const Mesh& mesh = function.function_mesh;
    const std::string& translated_function_name =
        function.translated_function_name;

    num_global_outputs += function.local_output_shapes.size();

    if (is_remote_mesh(mesh) ||
        (excluded_fn_names.find(translated_function_name) !=
         excluded_fn_names.end())) {
      // Skip execution for a translated function has remote mesh or when it is
      // excluded.
      continue;
    }

    const MeshWithParallelDevice* parallel_device_mesh =
        function_name_and_mesh_mapping[translated_function_name];

    // Gather the local inputs for this function.
    std::vector<parallel_device::ParallelTensor*> parallel_inputs;
    parallel_inputs.reserve(inputs.size() + 1);
    auto input_mapping = function.input_index_map;

    // We sort here because by this time, the function graph we are executing
    // is a reduced version of the main function, that includes the
    // StatefulPartitionedCall that we are executing for this mesh.
    // Thus, the ordering is the same as the main function ordering, which
    // is sorted increasingly.
    std::sort(input_mapping.begin(), input_mapping.end());

    for (const int global_index : input_mapping) {
      auto input_index = global_index - execution_functions->num_device_ids;

      if (global_index < execution_functions->num_device_ids) {
        parallel_inputs.push_back(
            parallel_device_mesh->DeviceIDs(context, status));
        if (TF_GetCode(status) != TF_OK) return;
      } else {
        parallel_inputs.push_back(global_parallel_inputs[input_index]);
      }
    }

    VLOG(4) << "Launching computation for mesh : " << mesh.ToString();
    parallel_device_mesh->parallel_device().StartExecute(
        context, parallel_inputs, translated_function_name.c_str(), attributes,
        /*expected_max_outputs=*/function.local_output_shapes.size(),
        *cancellation_manager_, /*step_id=*/step_id);
  }

  *num_outputs = num_global_outputs;
  std::vector<std::unique_ptr<TensorWithLayout>> typed_outputs;
  typed_outputs.resize(num_global_outputs);

  // Join all mesh computation together.
  // TODO(b/177932563): Expose cancel logic to handle failures.
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> join_status(
      TF_NewStatus(), TF_DeleteStatus);
  for (const TranslatedFunction& function :
       execution_functions->function_list) {
    // Skip execution for a function when it's excluded.
    if (excluded_fn_names.contains(function.translated_function_name)) {
      continue;
    }
    const Mesh& mesh = function.function_mesh;
    const MeshWithParallelDevice* parallel_device_mesh =
        function_name_and_mesh_mapping[function.translated_function_name];

    std::vector<std::unique_ptr<TensorWithLayout>> output_with_layout;
    output_with_layout.reserve(function.output_index_map.size());
    if (is_remote_mesh(mesh)) {
      // Create dummy outputs on a remote mesh.
      for (int i = 0; i < function.output_index_map.size(); ++i) {
        const auto dim_sizes = function.local_output_shapes.at(i).dim_sizes();
        std::vector<int64_t> local_shape =
            std::vector<int64_t>(dim_sizes.begin(), dim_sizes.end());
        TF_DataType dtype =
            static_cast<TF_DataType>(function.output_dtypes.at(i));
        auto remote_output =
            TensorWithLayout::Dummy(local_shape, dtype, *parallel_device_mesh,
                                    function.output_layouts[i]);
        output_with_layout.push_back(std::move(remote_output));
      }
    } else {
      VLOG(4) << "Joining computation result from mesh : " << mesh.ToString();
      auto result = parallel_device_mesh->parallel_device().Join(
          function.local_output_shapes, status);
      if (TF_GetCode(join_status.get()) != TF_OK &&
          // Preserve the first failure we see, but only if it is a real failure
          // and not a cancellation (which was probably triggered by the error
          // we want to propagate).
          (TF_GetCode(status) == TF_OK ||
           TF_GetCode(join_status.get()) != TF_CANCELLED)) {
        continue;
      }
      if (TF_GetCode(status) != TF_OK) {
        if (TF_GetCode(status) != TF_CANCELLED) {
          LOG(ERROR) << "Encountered error while executing function: "
                     << function.translated_function_name
                     << " for mesh : " << mesh.ToString()
                     << " / error : " << TF_Message(status);
        }
        TF_SetStatus(join_status.get(), TF_GetCode(status), TF_Message(status));
        continue;
      }

      for (int i = 0; i < result->size(); ++i) {
        ASSIGN_OR_RETURN_C_STATUS(
            auto local_output,
            TensorWithLayout::Wrap(std::move((*result)[i]),
                                   *parallel_device_mesh,
                                   function.output_layouts[i]),
            status);
        output_with_layout.push_back(std::move(local_output));
      }
    }

    for (int i = 0; i < function.output_index_map.size(); ++i) {
      // TODO(b/162744844): Generalize this pattern so that the extraction is
      // not special cased.
      if (function.shape_output_metadata.find(i) !=
          function.shape_output_metadata.end()) {
        output_with_layout[i]->set_input_layout_for_shape_op_result(
            function.shape_output_metadata.at(i));
      }

      RecordInShapeLayoutCache(*output_with_layout[i]);
      typed_outputs[function.output_index_map[i]] =
          std::move(output_with_layout[i]);
    }
  }
  if (TF_GetCode(join_status.get()) != TF_OK) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> async_wait_status(
        TF_NewStatus(), TF_DeleteStatus);
    AsyncWait(context, async_wait_status.get());
    TF_Code error_code = TF_GetCode(async_wait_status.get());
    if (error_code != TF_OK && error_code != TF_CANCELLED) {
      // Ignore the AsyncWait() status return since we already have a bad status
      // to propagate. We've just canceled a bunch of operations, so we expect
      // cancellation status returns. We'll log anything else just to be safe.
      LOG(ERROR) << "Error executing " << doperation.name << " "
                 << TF_Message(async_wait_status.get());
    }

    TF_SetStatus(status, TF_GetCode(join_status.get()),
                 TF_Message(join_status.get()));
    return;
  }
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "Executed " << doperation.name << ", got "
              << typed_outputs.size() << " outputs:";
    for (const std::unique_ptr<TensorWithLayout>& output : typed_outputs) {
      LOG(INFO) << "  " << output->DebugString();
    }
  }
  if (doperation.name == std::string("VarHandleOp")) {
    // For new variables, set the dereferenced shape/dtype so we can pass it in
    // as _handle_dtype and _handle_shape in the future.
    //
    // Note that VarHandleOps generated by `tf.Variable` objects are always run
    // eagerly, which is almost all of the op's usage in TF2. Theoretically a
    // user could run it in a tf.function via tf.raw_ops.VarHandleOp, return it
    // from that function, and add it as an input to another, and it would
    // currently be missing handle information.
    if (typed_outputs.size() != 1) {
      RETURN_STATUS(status, TF_INTERNAL,
                    "Expected one output from VarHandleOp");
    }
    NameAttrList name_and_attrs;
    ASSIGN_OR_RETURN_C_STATUS(name_and_attrs, FetchAttributes(attributes),
                              status);

    typed_outputs[0]->UpdateShapeAndDType(
        name_and_attrs.attr().at("shape").shape(),
        name_and_attrs.attr().at("dtype").type(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  for (int i = 0; i < *num_outputs; ++i) {
    outputs[i] =
        MakeLayoutTensorHandle(context, std::move(typed_outputs[i]), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
}

void DTensorDevice::Execute(const TFE_Op* original_op, int* num_outputs,
                            TFE_TensorHandle** outputs, TF_Status* status) {
  TFE_Context* context = TFE_OpGetContext(original_op, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* operation_name = TFE_OpGetName(original_op, status);
  if (TF_GetCode(status) != TF_OK) return;
  const TFE_OpAttrs* attributes = TFE_OpGetAttrs(original_op);
  int num_inputs = TFE_OpGetFlatInputCount(original_op, status);
  if (TF_GetCode(status) != TF_OK) return;
  std::vector<TFE_TensorHandle*> inputs_vector;
  inputs_vector.reserve(num_inputs);
  for (int input_index = 0; input_index < num_inputs; ++input_index) {
    TFE_TensorHandle* input =
        TFE_OpGetFlatInput(original_op, input_index, status);
    if (TF_GetCode(status) != TF_OK) return;
    inputs_vector.push_back(input);
  }
  TFE_TensorHandle** inputs = inputs_vector.data();

  DTensorOperation dtensor_operation{};
  dtensor_operation.name = operation_name;
  {
    dtensor_operation.function_def =
        tensorflow::unwrap(context)->FindFunctionDef(operation_name);
  }

  // First handle DTensor-specific virtual operations.
  bool is_op_handled = false;
  MaybeHandleDTensorCustomOps(operation_name, num_inputs, attributes, context,
                              inputs, num_outputs, outputs, &is_op_handled,
                              status);
  if (is_op_handled) return;

  // This isn't a special op, so we'll defer to TFE_Execute to actually execute
  // it, but we'll also run DTensor MLIR passes and propagate the layout.
  std::vector<TensorWithLayout*> typed_inputs;
  std::vector<std::unique_ptr<TensorWithLayout>> inputs_with_no_layout;

  // Record a unique mesh identified through all inputs that's already on
  // DTensor device. If we can identify a single mesh, the same mesh is used as
  // the mesh to broadcast non-dtensor inputs.
  absl::flat_hash_set<Mesh> input_meshes;
  std::vector<int> not_on_device_input_indices;

  typed_inputs.resize(num_inputs);
  for (int j = 0; j < num_inputs; ++j) {
    TFE_TensorHandle* input = inputs[j];
    const char* input_device = TFE_TensorHandleDeviceName(input, status);
    if (TF_GetCode(status) != TF_OK) return;
    if (name_ != input_device) {
      not_on_device_input_indices.push_back(j);
      continue;
    }
    // Handle input which is on DTensor device already.
    TensorWithLayout* t = reinterpret_cast<TensorWithLayout*>(
        TFE_TensorHandleDevicePointer(input, status));
    if (TF_GetCode(status) != TF_OK) return;

    // VarHandleOp runs on empty mesh, and that isn't registered with device.
    if (!t->layout().mesh().IsEmpty()) {
      input_meshes.insert(t->layout().mesh());
    }
    // Remote mesh inputs are not able to be read and evaluated.
    if (!is_remote_mesh(t->layout().mesh()) && !t->const_value().has_value()) {
      std::optional<NodeDef> const_value =
          ExtractSmallTensorValue(context, input, t->layout(), status);
      if (TF_GetCode(status) != TF_OK) return;
      if (const_value.has_value()) {
        t->set_const_value(const_value.value());
      }
    }
    typed_inputs[j] = t;
  }

  // If a unique mesh is identified across all inputs, we use that mesh as the
  // mesh to broadcast to. Otherwise we fallback to default mesh.
  const MeshWithParallelDevice* broadcast_mesh =
      input_meshes.size() == 1
          ? mesh_to_device_map_[*input_meshes.begin()].get()
          : default_mesh_;
  if (!broadcast_mesh) {
    RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                  "No mesh has been registered to DTensor. Use copy_to_mesh to "
                  "explicit specify a mesh instead.");
  }
  for (int not_on_device_input_index : not_on_device_input_indices) {
    TFE_TensorHandle* input = inputs[not_on_device_input_index];
    // DTensor creation should be explicit, with some exceptions for usability
    // (scalars/shapes/slice specs/etc.) Here we do some trivial validation to
    // enforce this rule.
    int num_dims = TFE_TensorHandleNumDims(input, status);
    if (TF_GetCode(status) != TF_OK) return;
    int64_t num_elements = TFE_TensorHandleNumElements(input, status);
    if (TF_GetCode(status) != TF_OK) return;
    TF_DataType dtype = TFE_TensorHandleDataType(input);
    const bool small_int_tensor = num_elements < kSmallTensorThreshold &&
                                  (dtype == TF_INT32 || dtype == TF_INT64);
    if (!(num_dims == 0 || dtype == TF_STRING || small_int_tensor)) {
      std::vector<int64_t> tensor_shape(TensorShapeAsVector(input, status));
      if (TF_GetCode(status) != TF_OK) return;
      RETURN_STATUS(
          status, TF_UNIMPLEMENTED,
          absl::StrCat(
              "The op/function ", operation_name,
              " got a regular tensor for input ", not_on_device_input_index,
              " (shape ", ShapeToDebugString(tensor_shape),
              ") but was expecting a DTensor. Currently only scalars and "
              "small integer/string tensors are auto-broadcast to "
              "DTensors. For other tensors, please use copy_to_mesh to "
              "make a DTensor explicitly; note that this may be slow if it "
              "happens frequently.")
              .c_str());
    }
    // Construct temporary TensorWithLayout objects for inputs that didn't
    // have any to start. These are owned by the `inputs_with_no_layout`
    // vector, whereas the input `TFE_TensorHandle`s maintain ownership for
    // inputs that already had layouts (and therefor had TensorWithLayout
    // objects).
    std::unique_ptr<TensorWithLayout> wrapper = TensorWithLayout::Broadcast(
        context, input, *broadcast_mesh, name_, status);
    if (TF_GetCode(status) != TF_OK) return;
    if (!ShouldFoldInputArgument(dtensor_operation.name,
                                 /*input_index=*/not_on_device_input_index)) {
      wrapper->reset_const_value();
    }
    typed_inputs[not_on_device_input_index] = wrapper.get();
    inputs_with_no_layout.emplace_back(wrapper.release());
  }

  ExecuteRegularOperation(context, typed_inputs, dtensor_operation, attributes,
                          num_outputs, outputs, status);
}

void ExecuteOnDTensorDevice(const TFE_Op* original_op, int* num_outputs,
                            TFE_TensorHandle** outputs, TF_Status* status,
                            void* device_info) {
  DTensorDevice* dev = reinterpret_cast<DTensorDevice*>(device_info);
  dev->Execute(original_op, num_outputs, outputs, status);
}

void DeleteDTensorDevice(void* device_info) {
  delete static_cast<DTensorDevice*>(device_info);
}

TFE_TensorHandle* CopyToDTensorDevice(TFE_Context* context,
                                      TFE_TensorHandle* tensor,
                                      TF_Status* status, void* device_info) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "Trying to copy a tensor on to a DTensor mesh without a layout "
               "(use the CopyToMesh op for now).");
  return nullptr;
}

TFE_TensorHandle* CopyFromDTensorDevice(TFE_Context* context,
                                        TFE_TensorHandle* tensor,
                                        const char* target_device_name,
                                        TF_Status* status, void* device_info) {
  TensorWithLayout* typed_input = reinterpret_cast<TensorWithLayout*>(
      TFE_TensorHandleDevicePointer(tensor, status));
  if (!tensorflow::dtensor::Layout(typed_input->layout()).IsFullyReplicated()) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Trying to copy a non-replicated DTensor is not supported.");
    return nullptr;
  }
  if (typed_input->tensor()->dtype() == TF_RESOURCE) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Trying to copy a DTensor resource handle is not supported.");
    return nullptr;
  }
  DTensorDevice* dev = reinterpret_cast<DTensorDevice*>(device_info);
  // Since operations are executed asynchronously, the operation which should
  // produce the tensor we're trying to copy off the DTensor device may be
  // canceled due to a failure on another device. If so, we want to report the
  // failure that caused the cancellation, not the cancellation itself. This
  // requires blocking waiting for other devices to flush their execution
  // queues.
  // Note that we also only need to sync the threads on the parallel_device()
  // directly, or a context level sync might cause unintentional deadlocks when
  // grabbing locks on other threads.
  dev->AsyncWait(context, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TFE_TensorHandleCopySharingTensor(typed_input->get_tensor(0), status);
}

void AllocateDTensorDevice(absl::string_view device_name,
                           TFE_CustomDevice* device, void** device_info) {
  device->copy_tensor_to_device = &CopyToDTensorDevice;
  device->copy_tensor_from_device = &CopyFromDTensorDevice;
  device->delete_device = &DeleteDTensorDevice;
  device->execute = &ExecuteOnDTensorDevice;
  *device_info = new DTensorDevice(device_name);
}

void AddMesh(const std::string& serialized_mesh, void* device_info,
             bool is_async, bool is_host_mesh, TF_Status* status) {
  auto mesh_config_or_status = Mesh::FromString(serialized_mesh);
  if (!mesh_config_or_status.ok()) {
    TF_SetStatus(status, TF_INTERNAL,
                 absl::StrCat("Failed to parse mesh config. ",
                              mesh_config_or_status.status().error_message())
                     .c_str());
    return;
  }
  auto mesh_config = mesh_config_or_status.value();
  std::vector<std::string> underlying_devices;
  underlying_devices.insert(underlying_devices.end(),
                            mesh_config.local_devices().begin(),
                            mesh_config.local_devices().end());
  // DTensor uses multi-client setup which doesn't use remote eager, so we can
  // enable eager async execution in ParallelDevice.
  std::unique_ptr<tensorflow::parallel_device::ParallelDevice> parallel(
      new tensorflow::parallel_device::ParallelDevice(underlying_devices,
                                                      is_async));

  std::string composite_device_name;
  if (absl::StartsWith(mesh_config.name(), kPipelineMeshNamePrefix)) {
    composite_device_name = std::string(
        absl::StripPrefix(mesh_config.name(), kPipelineMeshNamePrefix));
  }

  auto mesh = std::make_unique<MeshWithParallelDevice>(
      std::move(mesh_config), std::move(parallel), composite_device_name);
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->AddMesh(std::move(mesh), is_host_mesh);
}

void ExperimentalSetDefaultLayout(const std::string& serialized_layout,
                                  void* device_info, TF_Status* status) {
  StatusOr<Layout> layout = Layout::FromString(serialized_layout);
  if (!layout.ok()) {
    RETURN_STATUS(status, TF_INTERNAL, layout.status().error_message().c_str());
  }
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->SetDefaultLayout(layout.value());
}

void ExperimentalClearDefaultLayout(void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->ClearDefaultLayout();
}

void ExperimentalSetDefaultMesh(const std::string& serialized_mesh,
                                void* device_info, TF_Status* status) {
  StatusOr<Mesh> mesh = Mesh::FromString(serialized_mesh);
  if (!mesh.ok()) {
    RETURN_STATUS(status, TF_INTERNAL, mesh.status().error_message().c_str());
  }
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->SetDefaultMesh(mesh.value());
}

void ExperimentalClearDefaultMesh(void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->ClearDefaultMesh();
}

void SetSameShapePolicy(void* device_info, bool enabled) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->SetSameShapePolicy(enabled);
}

void SetTPUCoreIDs(const std::string& mesh_name,
                   const std::vector<int>& tpu_core_ids, void* device_info,
                   TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  RETURN_C_STATUS_IF_NOT_OK(device->SetTPUCoreIDs(mesh_name, tpu_core_ids),
                            status);
}

void ClearTPUCoreIDs(void* device_info) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  device->ClearTPUCoreIDs();
}

std::vector<std::vector<int>> TPUCoreIDsToLocations(
    TFE_Context* context, const std::vector<int>& tpu_core_ids,
    void* device_info) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->TPUCoreIDsToLocations(context, tpu_core_ids);
}

std::vector<int> TPUCoreLocationsToIDs(
    TFE_Context* context,
    const std::vector<std::vector<int>>& tpu_core_locations,
    void* device_info) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->TPUCoreLocationsToIDs(context, tpu_core_locations);
}

TFE_TensorHandle* Pack(TFE_Context* context, int num_inputs,
                       TFE_TensorHandle** inputs,
                       const std::string& string_layout, void* device_info,
                       TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->Pack(context, num_inputs, inputs, string_layout, status);
}

std::vector<TFE_TensorHandle*> Unpack(TFE_Context* context,
                                      TFE_TensorHandle* input,
                                      void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->Unpack(context, input, status);
}

std::string FetchLayout(TFE_Context* context, TFE_TensorHandle* input,
                        void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->FetchLayout(context, input, status);
}

TFE_TensorHandle* SparsePack(TFE_Context* context, int num_inputs,
                             TFE_TensorHandle** indices,
                             TFE_TensorHandle** values,
                             TFE_TensorHandle** shapes,
                             const std::string& string_layout,
                             void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->SparsePack(context, num_inputs, indices, values, shapes,
                            string_layout, status);
}

bool IsSparseDTensor(TFE_Context* context, TFE_TensorHandle* input,
                     void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->IsSparseDTensor(context, input, status);
}

std::unordered_map<std::string, int> GetFunctionCacheHitAndMissCount(
    TFE_Context* context, void* device_info, TF_Status* status) {
  DTensorDevice* device = reinterpret_cast<DTensorDevice*>(device_info);
  return device->GetFunctionCacheHitAndMissCount(context, status);
}
}  // namespace dtensor
}  // namespace tensorflow
