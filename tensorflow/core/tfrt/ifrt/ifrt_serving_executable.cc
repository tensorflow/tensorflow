/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/extract_callback.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/pack_inputs_pass.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/mlir/tfrt/utils/export.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/layout.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dump.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_device_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/platform/tstring.h"
#include "tsl/profiler/lib/traceme.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace {

using StaticShapeMap =
    absl::flat_hash_map<size_t /*original_arg_idx*/,
                        tensorflow::TensorShape /*static_shape*/>;

// Per-element byte size of a pack-eligible tensor element type. Returns -1
// for unsupported types (sub-byte, complex, string, resource).
int64_t ElementByteSize(mlir::Type elt) {
  if (elt.isIntOrFloat()) {
    int64_t bw = elt.getIntOrFloatBitWidth();
    if (bw < 8 || bw % 8 != 0) return -1;
    return bw / 8;
  }
  // TF quantized types: storage is plain (u)int{8,16,32}.
  if (llvm::isa<mlir::TF::Quint8Type, mlir::TF::Qint8Type>(elt)) return 1;
  if (llvm::isa<mlir::TF::Quint16Type, mlir::TF::Qint16Type>(elt)) return 2;
  if (llvm::isa<mlir::TF::Qint32Type>(elt)) return 4;
  return -1;
}

// Plan for packing a request's host-side input tensors into one (or more)
// coalesced host->device transfers.
//
// Both vectors are parallel to the request input list (i.e., the `inputs`
// argument of IfrtServingExecutable::Execute). For input i:
//   pack_group_ids[i] == -1  -> transfer this input individually.
//   pack_group_ids[i] >= 0   -> fuse into the named pack group at
//                               `pack_offsets[i]` bytes from the group's host
//                               buffer base.
struct PackPlan {
  std::vector<int64_t> pack_group_ids;
  std::vector<int64_t> pack_offsets;
};

struct PackPlanOptions {
  // Inputs whose byte size exceeds this threshold are not packed. Default
  // chosen to keep packed buffer footprint reasonable while still catching the
  // common case of many small categorical/scalar features.
  int64_t size_threshold_bytes = 131072;
  // Alignment applied to each input's offset within its pack group, in bytes.
  // 16 matches typical HLO host-buffer alignment expectations.
  int64_t alignment_bytes = 16;
};

// Per-element byte size for a dtype, or -1 if the dtype is not pack-eligible.
// Pack-eligible = whole-byte numeric storage with no host-side
// reinterpretation. Excludes strings, resources, variants, complex types, and
// sub-byte ints (where simply memcpying the host bytes would corrupt
// neighbors).
int64_t DtypeByteSize(tensorflow::DataType dtype) {
  switch (dtype) {
    case DT_BOOL:
    case DT_INT8:
    case DT_UINT8:
    case DT_QINT8:
    case DT_QUINT8:
      return 1;
    case DT_INT16:
    case DT_UINT16:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_HALF:
    case DT_BFLOAT16:
      return 2;
    case DT_INT32:
    case DT_UINT32:
    case DT_FLOAT:
    case DT_QINT32:
      return 4;
    case DT_INT64:
    case DT_UINT64:
    case DT_DOUBLE:
      return 8;
    default:
      return -1;
  }
}

int64_t RoundUpToAlignment(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

// Layout-fussy TF ops (Filter 4). Packing an operand that feeds one of these
// triggers an on-device relayout copy after the bitcast_convert unpack. On
// TPU especially, that copy can dominate the H2D launch savings.
const absl::flat_hash_set<absl::string_view>& LayoutFussyTfOps() {
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>> kSet({
      "tf.MatMul",
      "tf.BatchMatMul",
      "tf.BatchMatMulV2",
      "tf.BatchMatMulV3",
      "tf.Conv2D",
      "tf.Conv2DBackpropFilter",
      "tf.Conv2DBackpropInput",
      "tf.Conv3D",
      "tf.DepthwiseConv2dNative",
      "tf.MaxPool",
      "tf.MaxPool3D",
      "tf.AvgPool",
      "tf.AvgPool3D",
      "tf.Einsum",
  });
  return *kSet;
}

bool FeedsLayoutFussyOp(mlir::Value input) {
  llvm::SmallVector<mlir::Value, 4> worklist;
  worklist.push_back(input);
  llvm::SmallPtrSet<mlir::Value, 4> visited;

  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;

    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* owner = use.getOwner();
      llvm::StringRef name = owner->getName().getStringRef();
      if (LayoutFussyTfOps().contains(name)) return true;

      if (name == "tf.Identity" || name == "tf.Cast" || name == "tf.Reshape" ||
          name == "tf.ExpandDims" || name == "tf.Squeeze" ||
          name == "tf.StopGradient" || name == "tf.Bitcast") {
        for (mlir::Value res : owner->getResults()) {
          worklist.push_back(res);
        }
      }
    }
  }
  return false;
}

absl::StatusOr<PackPlan> ComputePackPlanFromInputs(
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices, mlir::func::FuncOp main_func,
    PackPlanOptions options = {}) {
  if (options.alignment_bytes <= 0 || options.size_threshold_bytes < 0) {
    return absl::InvalidArgumentError(
        "PackPlanOptions: alignment_bytes must be > 0 and "
        "size_threshold_bytes must be >= 0.");
  }

  const int num_inputs = static_cast<int>(dtypes_and_shapes.size());
  PackPlan plan;
  plan.pack_group_ids.assign(num_inputs, /*not_packed=*/-1);
  plan.pack_offsets.assign(num_inputs, /*default_offset=*/0);

  absl::flat_hash_set<int> variable_set(variable_arg_indices.begin(),
                                        variable_arg_indices.end());

  absl::flat_hash_map<int64_t, int64_t> next_offset_per_group;
  for (int i = 0; i < num_inputs; ++i) {
    // Filter 1: variable args resolve via on-device NamedArray, not H2D.
    if (variable_set.contains(i)) continue;

    // Filter 2: dtype pack-eligibility.
    int64_t bytes_per_elt = DtypeByteSize(dtypes_and_shapes[i].dtype);
    if (bytes_per_elt <= 0) continue;

    // Byte size from the concrete request shape. num_elements() returns -1
    // for unknown rank, which shouldn't happen for tensors fed into Execute().
    int64_t num_elements = dtypes_and_shapes[i].shape.num_elements();
    if (num_elements < 0) continue;
    int64_t bytes = num_elements * bytes_per_elt;

    // Filter 3: economic threshold. Large transfers already amortize per-launch
    // overhead; packing them grows the staging buffer without latency benefit.
    if (bytes > options.size_threshold_bytes) continue;

    // Filter 4: bypass MLIR-level packing for layout-fussy ops, but tell the
    // executor to batch transfer them via arena slicing.
    if (main_func && static_cast<unsigned>(i) < main_func.getNumArguments() &&
        FeedsLayoutFussyOp(main_func.getArgument(i))) {
      plan.pack_group_ids[i] = -2;
      continue;
    }

    int64_t group_id = bytes_per_elt;
    plan.pack_group_ids[i] = group_id;
    int64_t& next_offset = next_offset_per_group[group_id];
    plan.pack_offsets[i] = next_offset;
    next_offset =
        RoundUpToAlignment(next_offset + bytes, options.alignment_bytes);
  }

  return plan;
}

bool IsSingleDevice(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  return compile_metadata.num_replicas() == 1 &&
         compile_metadata.num_cores_per_replica() == 1;
}

absl::StatusOr<std::vector<DtypeAndShape>> BuildDtypeAndShape(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const StaticShapeMap& static_shapes_map,
    const IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry) {
  std::vector<DtypeAndShape> dtypes_and_shapes;
  dtypes_and_shapes.reserve(inputs.size());

  int variable_arg_index = 0;
  for (int i = 0; i < inputs.size(); i++) {
    std::optional<tensorflow::TensorShape> static_shape;
    auto it = static_shapes_map.find(i);
    if (it != static_shapes_map.end()) {
      static_shape = it->second;
    }
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      // Get already loaded variable tensor.
      TF_ASSIGN_OR_RETURN(auto dtype_and_shape,
                          ifrt_restore_tensor_registry.GetDtypeAndShape(
                              inputs[i].scalar<tsl::tstring>()()));
      dtype_and_shape.static_shape = std::move(static_shape);
      dtypes_and_shapes.push_back(std::move(dtype_and_shape));

      variable_arg_index++;
    } else {
      dtypes_and_shapes.push_back(
          DtypeAndShape{.dtype = inputs[i].dtype(),
                        .shape = inputs[i].shape(),
                        .static_shape = std::move(static_shape)});
    }
  }
  return dtypes_and_shapes;
}

// Returns the device assignment from the given IFRT devices list.
absl::StatusOr<xla::DeviceAssignment> GetRuntimeXlaDeviceAssignment(
    const xla::ifrt::DeviceListRef& device_list, int num_replicas,
    int num_cores_per_replica) {
  const int num_devices = num_replicas * num_cores_per_replica;
  const absl::Span<xla::ifrt::Device* const> devices = device_list->devices();
  if (devices.size() != num_devices) {
    return absl::InternalError(
        absl::StrCat("Device assignment has ", devices.size(),
                     " devices, but expected ", num_devices));
  }
  xla::DeviceAssignment da(num_replicas, num_cores_per_replica);
  int device_index = 0;
  for (int replica_idx = 0; replica_idx < num_replicas; replica_idx++) {
    for (int core_idx = 0; core_idx < num_cores_per_replica;
         core_idx++, device_index++) {
      da(replica_idx, core_idx) = devices[device_index]->Id().value();
      VLOG(3) << "Added IFRT device id: " << da(replica_idx, core_idx);
    }
  }
  return da;
}

static constexpr absl::string_view kDeviceAssignmentAttr = "device_assignment";
static constexpr absl::string_view kEntryFuncName = "main";

absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedDevices(
    mlir::ModuleOp module, const xla::ifrt::Client& ifrt_client,
    int num_replicas, int num_cores_per_replica) {
  auto op = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!op) {
    return absl::InternalError("Could not find entry function in MLIR Module.");
  }

  auto device_assignment_attr =
      op->getAttrOfType<mlir::ArrayAttr>(kDeviceAssignmentAttr);
  std::optional<std::vector<int>> device_assignment_attr_val;

  if (device_assignment_attr && !device_assignment_attr.getValue().empty()) {
    std::vector<int> coords;
    coords.reserve(num_replicas * num_cores_per_replica);
    for (auto coord_attr : device_assignment_attr.getValue()) {
      auto coord_attr_val = mlir::dyn_cast<mlir::IntegerAttr>(coord_attr);
      if (!coord_attr_val) {
        return absl::InternalError(
            llvm::formatv("Device assignment attribute is not an integer: {0}",
                          device_assignment_attr)
                .str());
      }
      coords.push_back(coord_attr_val.getInt());
    }
    device_assignment_attr_val = std::move(coords);
  }
  return GetAssignedIfrtDevices(ifrt_client, num_replicas,
                                num_cores_per_replica,
                                device_assignment_attr_val);
}

absl::StatusOr<
    absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>>
GetHostCallbackModulesAndRemoveHostFuncs(mlir::ModuleOp module) {
  absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>
      host_callback_modules;
  llvm::DenseSet<mlir::TF::XlaHostComputeOp> xla_host_compute_ops;
  module->walk(
      [&](mlir::TF::XlaHostComputeOp op) { xla_host_compute_ops.insert(op); });
  for (auto& op : xla_host_compute_ops) {
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> host_callback_module,
                        ExtractCallbackModule(module, op.getKey().str()));
    auto [_, inserted] = host_callback_modules.insert(
        {op.getKey().str(), std::move(host_callback_module)});
    if (!inserted) {
      return absl::FailedPreconditionError(
          absl::StrCat("Duplicate host callback key: ", op.getKey().str()));
    }
    auto func = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
        module, op.getKeyAttr());
    if (!func) {
      return absl::InternalError(
          absl::StrCat("symbol not found: ", op.getKey().str()));
    }
    func->erase();
  }
  return host_callback_modules;
}

// Retrieves static shapes for the inputs. For arguments specified in
// `static_shape_arg_map`, it extracts the actual shapes from the
// corresponding input tensors.
absl::StatusOr<StaticShapeMap> GetStaticShapesFromInputs(
    absl::Span<const tensorflow::Tensor> inputs,
    const absl::flat_hash_map<size_t, size_t>& static_shape_arg_map) {
  StaticShapeMap static_shapes;
  for (const auto& [original_arg_idx, static_shape_arg_idx] :
       static_shape_arg_map) {
    if (static_shape_arg_idx >= inputs.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Static shape arg index out of bound: got index ",
          static_shape_arg_idx, " with inputs size ", inputs.size()));
    }
    const auto& static_shape_tensor = inputs[static_shape_arg_idx];
    if (static_shape_tensor.dims() != 1) {
      return absl::InvalidArgumentError("Static shape tensor must be 1D");
    }
    if (static_shape_tensor.NumElements() > 2) {
      return absl::InvalidArgumentError("Static shape must be 1D or 2D");
    }
    tensorflow::TensorShape static_shape;
    if (static_shape_tensor.dtype() == tensorflow::DT_INT32) {
      auto flat = static_shape_tensor.flat<int32_t>();
      for (size_t k = 0; k < flat.size(); ++k) {
        static_shape.AddDim(flat(k));
      }
    } else if (static_shape_tensor.dtype() == tensorflow::DT_INT64) {
      auto flat = static_shape_tensor.flat<int64_t>();
      for (size_t k = 0; k < flat.size(); ++k) {
        static_shape.AddDim(flat(k));
      }
    } else {
      return absl::InternalError("Static shape tensor must be int32 or int64");
    }
    static_shapes.insert({original_arg_idx, std::move(static_shape)});
  }
  return static_shapes;
}

// Extracts the `tf._static_shape_arg_idx` attributes from the entry function
// of the MLIR module and returns them as a map.
absl::flat_hash_map<size_t, size_t> GetStaticShapeArgMap(
    mlir::ModuleOp module, absl::string_view signature_name) {
  absl::flat_hash_map<size_t, size_t> static_shape_arg_map;
  auto entry_func_op = module.lookupSymbol<mlir::func::FuncOp>(signature_name);
  if (!entry_func_op) {
    entry_func_op = module.lookupSymbol<mlir::func::FuncOp>("main");
  }
  if (entry_func_op) {
    for (size_t i = 0; i < entry_func_op.getNumArguments(); ++i) {
      if (auto arg_attr = entry_func_op.getArgAttrOfType<mlir::IntegerAttr>(
              i, "tf._static_shape_arg_idx")) {
        static_shape_arg_map[i] = arg_attr.getInt();
      }
    }
  }
  return static_shape_arg_map;
}

// If the input tensor's rank differs from `reshaped_shape` (e.g., due to
// compiler flattening), attempt to reshape the input to match the rank of
// `reshaped_shape`. The whole shape does not have to match, because the input
// is expected to be a prefix of the reshaped shape in case of static shape.
// Currently, only flattening to 1D is supported since it suffices the only use
// case we need to support.
absl::StatusOr<tensorflow::Tensor> MaybeReshapeInputForStaticShape(
    const tensorflow::Tensor& input,
    const tensorflow::TensorShape& reshaped_shape) {
  if (input.dims() > 2 || reshaped_shape.dims() > 2) {
    return absl::UnimplementedError(absl::StrCat(
        "MaybeReshapeInputForStaticShape only supports input and reshaped "
        "shapes with at most 2 dimensions. Got input dims: ",
        input.dims(), ", reshaped_shape dims: ", reshaped_shape.dims()));
  }
  if (input.dims() == reshaped_shape.dims()) {
    return input;
  }
  if (input.dims() == 1 && reshaped_shape.dims() == 2) {
    return absl::UnimplementedError(absl::StrCat(
        "Input shape rank is 1 but reshaped shape rank is 2. Got input shape: ",
        input.shape().DebugString(),
        ", reshaped_shape: ", reshaped_shape.DebugString()));
  }
  tensorflow::TensorShape flattened_shape;
  flattened_shape.AddDim(input.NumElements());
  tensorflow::Tensor flattened;
  if (!flattened.CopyFrom(input, flattened_shape)) {
    return absl::InternalError(absl::StrCat(
        "Failed to flatten input tensor to match static shape rank 1. ",
        "Input shape: ", input.shape().DebugString()));
  }
  return flattened;
}

}  // namespace

absl::StatusOr<std::unique_ptr<IfrtServingExecutable>>
IfrtServingExecutable::Create(
    int64_t program_id, absl::string_view model_name,
    absl::string_view signature_name, mlir::OwningOpRef<mlir::ModuleOp> module,
    std::shared_ptr<xla::ifrt::Client> client,
    tsl::thread::ThreadPool* thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    const IfrtRestoreTensorRegistry* ifrt_restore,
    tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
    tensorflow::DeviceMgr* device_mgr,
    tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    IfrtServingCoreSelector* ifrt_serving_core_selector,
    std::variant<tsl::protobuf::Message*,
                 xla::CompileOptions::EnvironmentOptionOverrides>
        compilation_env_or_overrides,
    TfToHloCompiler* tf_to_hlo_compiler,
    IfrtPersistentCompilationCache* persistent_compilation_cache,
    H2DTransferExecutorFactory* h2d_transfer_executor_factory) {
  if (h2d_transfer_executor_factory == nullptr) {
    return absl::InvalidArgumentError("H2DTransferExecutorFactory is null.");
  }
  TF_ASSIGN_OR_RETURN(
      tensorflow::tpu::TPUCompileMetadataProto original_compile_metadata,
      GetCompileMetadata(*module, *client));

  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::Device*> assigned_devices,
      GetAssignedDevices(*module, *client,
                         original_compile_metadata.num_replicas(),
                         original_compile_metadata.num_cores_per_replica()));

  absl::flat_hash_map<size_t, size_t> static_shape_arg_map =
      GetStaticShapeArgMap(*module, signature_name);

  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      client->MakeDeviceList(assigned_devices));
  auto executable = absl::WrapUnique(new IfrtServingExecutable(
      program_id, model_name, signature_name, std::move(module), client,
      thread_pool, ifrt_loaded_variable_registry, ifrt_restore,
      checkpoint_loader_queue, device_mgr, std::move(shape_representation_fn),
      ifrt_serving_core_selector, std::move(original_compile_metadata),
      std::move(device_list), std::move(static_shape_arg_map),
      compilation_env_or_overrides, tf_to_hlo_compiler,
      persistent_compilation_cache, h2d_transfer_executor_factory));

  return executable;
}

absl::StatusOr<std::vector<tensorflow::FunctionDef>> BuildFunctionDef(
    mlir::ModuleOp module) {
  std::vector<tensorflow::FunctionDef> function_defs;

  // Sets `export_tf_original_func_name` to false so that ExportFunctionDef
  // does not rename the function back to the original function name. This
  // allows calling the function by the function name in the MLIR module.
  TF_RETURN_IF_ERROR(ExportFunctionDefs(
      module,
      [&](tensorflow::FunctionDef function_def) {
        function_defs.push_back(std::move(function_def));
        return absl::OkStatus();
      },
      /*export_tf_original_func_name=*/false));

  return function_defs;
}

// Host callback info for one host callback.
struct HostCallbackBuilderInfo {
  tensorflow::tf2xla::HostTransferMetadata device_to_host;
  tensorflow::tf2xla::HostTransferMetadata host_to_device;
};

absl::StatusOr<absl::flat_hash_map<std::string, HostCallbackBuilderInfo>>
GroupHostCallbackByKey(
    const tf2xla::HostComputeMetadata& host_compute_metadata) {
  absl::flat_hash_map<std::string, HostCallbackBuilderInfo> host_callbacks;

  for (const auto& device_to_host : host_compute_metadata.device_to_host()) {
    auto& host_callback = host_callbacks[device_to_host.key()];
    host_callback.device_to_host = device_to_host;
  }
  for (const auto& host_to_device : host_compute_metadata.host_to_device()) {
    auto& host_callback = host_callbacks[host_to_device.key()];
    host_callback.host_to_device = host_to_device;
  }
  return host_callbacks;
}

// TODO: shape propagation in module
absl::StatusOr<xla::HostCallback> BuildHostCallback(
    absl::string_view key, const HostCallbackBuilderInfo& builder_info,
    mlir::ModuleOp callback_module, tensorflow::DeviceMgr* device_mgr,
    std::vector<std::unique_ptr<TfHostCallback>>& tf_host_callbacks) {
  LOG_EVERY_N_SEC(INFO, 20) << "BuildHostCallback for key: " << key;

  DCHECK(device_mgr);
  xla::HostCallback host_callback;
  std::vector<DtypeAndShape> operand_type_and_shapes;
  std::vector<DtypeAndShape> result_type_and_shapes;

  auto to_xla_shape = [](tensorflow::DataType data_type,
                         const tensorflow::TensorShapeProto& shape)
      -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_ASSIGN_OR_RETURN(tensorflow::TensorShape tensor_shape,
                        tensorflow::TensorShape::BuildTensorShape(shape));

    if (absl::Status status = tensorflow::TensorShapeToXLAShape(
            data_type, tensor_shape, &xla_shape);
        status.ok()) {
      return xla_shape;
    } else {
      return status;
    }
  };

  operand_type_and_shapes.reserve(builder_info.device_to_host.metadata_size());
  result_type_and_shapes.reserve(builder_info.host_to_device.metadata_size());
  for (const auto& metadata : builder_info.device_to_host.metadata()) {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        to_xla_shape(metadata.type(), metadata.shape()));
    uint16_t channel_id = static_cast<uint16_t>(metadata.channel_id());
    LOG_EVERY_N_SEC(INFO, 20) << "Channel id: " << channel_id;
    host_callback.operands.push_back(
        {.channel_id = channel_id, .shape = shape});
    operand_type_and_shapes.push_back(
        DtypeAndShape{.dtype = metadata.type(), .shape = metadata.shape()});
  }

  for (const auto& metadata : builder_info.host_to_device.metadata()) {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        to_xla_shape(metadata.type(), metadata.shape()));
    uint16_t channel_id = static_cast<uint16_t>(metadata.channel_id());
    LOG_EVERY_N_SEC(INFO, 20) << "Channel id: " << channel_id;
    host_callback.results.push_back(
        {.channel_id = channel_id, .shape = std::move(shape)});
    result_type_and_shapes.push_back(
        DtypeAndShape{.dtype = metadata.type(), .shape = metadata.shape()});
  }

  TF_ASSIGN_OR_RETURN(std::vector<tensorflow::FunctionDef> function_defs,
                      BuildFunctionDef(callback_module));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfHostCallback> tf_host_callback,
      TfHostCallback::Create(function_defs, key, operand_type_and_shapes,
                             result_type_and_shapes, device_mgr));

  host_callback.callback = [tf_host_callback = tf_host_callback.get()](
                               void** output, void** input) {
    return tf_host_callback->Call(input, output);
  };

  tf_host_callbacks.push_back(std::move(tf_host_callback));
  return host_callback;
}

absl::StatusOr<std::vector<xla::HostCallback>> BuildHostCallbacks(
    const tf2xla::HostComputeMetadata& host_compute_metadata,
    absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>
        host_callback_modules,
    tensorflow::DeviceMgr* device_mgr,
    std::vector<std::unique_ptr<TfHostCallback>>& tf_host_callbacks) {
  TF_ASSIGN_OR_RETURN(auto host_callback_maps,
                      GroupHostCallbackByKey(host_compute_metadata));

  std::vector<xla::HostCallback> host_callbacks;
  host_callbacks.reserve(host_callback_maps.size());
  for (const auto& [entry_function, builder_info] : host_callback_maps) {
    auto host_callback_module_it = host_callback_modules.find(entry_function);
    if (host_callback_module_it == host_callback_modules.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Host callback module not found for key: ", entry_function));
    }
    TF_ASSIGN_OR_RETURN(auto host_callback,
                        BuildHostCallback(entry_function, builder_info,
                                          *host_callback_module_it->second,
                                          device_mgr, tf_host_callbacks));
    host_callbacks.push_back(std::move(host_callback));
  }

  return host_callbacks;
}

absl::Status EncodeLayout(absl::Span<const xla::Shape> xla_input_shapes,
                          mlir::ModuleOp module) {
  auto op = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!op) {
    return absl::InternalError("Could not find entry function in MLIR Module.");
  }

  if (op.getNumArguments() != xla_input_shapes.size()) {
    return absl::InternalError(
        absl::StrCat("Argument count mismatch: ", op.getNumArguments(), " vs ",
                     xla_input_shapes.size()));
  }

  mlir::OpBuilder builder(op.getContext());
  for (int i = 0; i < xla_input_shapes.size(); ++i) {
    const auto& shape = xla_input_shapes[i];
    if (shape.has_layout()) {
      auto attr = builder.getStringAttr(shape.layout().ToString());
      op.setArgAttr(i, "mhlo.layout_mode", attr);
    }
  }
  return absl::OkStatus();
}

absl::Status IfrtServingExecutable::PopulateInvariantMetadata(
    const Tf2HloResult& tf2hlo_result,
    xla::ifrt::LoadedExecutableRef ifrt_executable,
    std::vector<std::unique_ptr<TfHostCallback>> host_callbacks,
    const xla::ifrt::Topology* topology,
    CachedExecutableBundle& executable_bundle) {
  executable_bundle.ifrt_input_dtypes.reserve(
      tf2hlo_result.compile_metadata.args().size());
  executable_bundle.reshaped_input_tensors.reserve(
      tf2hlo_result.compile_metadata.args().size());
  executable_bundle.ifrt_input_shapes.reserve(
      tf2hlo_result.compile_metadata.args().size());
  executable_bundle.xla_input_shapes.reserve(
      tf2hlo_result.compile_metadata.args().size());
  executable_bundle.xla_input_layouts.reserve(
      tf2hlo_result.compile_metadata.args().size());
  executable_bundle.byte_strides.reserve(
      tf2hlo_result.compile_metadata.args().size());

  TF_ASSIGN_OR_RETURN(auto parameter_layouts,
                      ifrt_executable->GetParameterLayouts());

  const xla::PjRtTopologyDescription* pjrt_topology = nullptr;
  if (!tf2hlo_result.xla_input_shapes.empty() && topology) {
    pjrt_topology = topology->description().get();
  }

  for (int i = 0; i < tf2hlo_result.compile_metadata.args().size(); ++i) {
    const auto& arg = tf2hlo_result.compile_metadata.args(i);
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, ToIfrtDType(arg.dtype()));
    executable_bundle.ifrt_input_dtypes.push_back(ifrt_dtype);
    TF_ASSIGN_OR_RETURN(auto reshaped_tensor,
                        tensorflow::TensorShape::BuildTensorShape(arg.shape()));

    xla::ifrt::Shape ifrt_shape = ToIfrtShape(reshaped_tensor);
    executable_bundle.ifrt_input_shapes.push_back(
        std::make_shared<xla::ifrt::Shape>(std::move(ifrt_shape)));
    executable_bundle.reshaped_input_tensors.push_back(
        std::move(reshaped_tensor));

    if (!tf2hlo_result.xla_input_shapes.empty()) {
      auto shape_ptr =
          std::make_shared<xla::Shape>(tf2hlo_result.xla_input_shapes[i]);
      if (pjrt_topology) {
        // Canonicalize the `xla::Shape`. For example, on TPUs, this could set
        // the layout or tile dimensions depending on the memory space.
        const xla::Shape& request_shape = tf2hlo_result.xla_input_shapes[i];
        if (auto shape = pjrt_topology->MakeCanonicalShapeForMemorySpace(
                pjrt_topology->GetDefaultMemorySpaceKindId(), request_shape,
                request_shape.has_layout() ? &request_shape.layout() : nullptr);
            shape.ok()) {
          shape_ptr = std::make_shared<xla::Shape>(*std::move(shape));
        }
      }
      executable_bundle.xla_input_shapes.push_back(std::move(shape_ptr));
    } else {
      executable_bundle.xla_input_shapes.push_back(nullptr);
    }

    executable_bundle.byte_strides.push_back(
        GetByteStrides(arg.dtype(),
                       executable_bundle.reshaped_input_tensors.back())
            .value_or(absl::InlinedVector<int64_t, 4>()));

    // Create device shape with backend-optimized layout. The layouts from
    // `GetParameterLayouts()` are the physical formats expected by the
    // compiled program, which may include hardware-specific tiling or padding.
    int compiled_arg_idx = i;
    xla::ifrt::LayoutRef layout_ref;
    bool is_packed = false;
    if (!executable_bundle.pack_group_ids.empty()) {
      if (executable_bundle.pack_group_ids[i] >= 0) {
        is_packed = true;
      } else {
        int num_packed_before = 0;
        for (int j = 0; j < i; ++j) {
          if (executable_bundle.pack_group_ids[j] >= 0) {
            num_packed_before++;
          }
        }
        compiled_arg_idx = i - num_packed_before;
      }
    }

    if (!is_packed && compiled_arg_idx < parameter_layouts.size()) {
      const auto& layout = parameter_layouts[compiled_arg_idx];
      if (layout && layout->xla_layout().minor_to_major_size() ==
                        reshaped_tensor.dims()) {
        layout_ref = xla::ifrt::PjRtLayout::Create(layout);
      }
    }
    executable_bundle.xla_input_layouts.push_back(std::move(layout_ref));
  }

  executable_bundle.ifrt_executable = std::move(ifrt_executable);
  executable_bundle.compile_metadata =
      std::move(tf2hlo_result.compile_metadata);
  executable_bundle.host_callbacks = std::move(host_callbacks);

  executable_bundle.arg_hlo_shardings.reserve(
      executable_bundle.compile_metadata.args().size());

  for (const auto& arg : executable_bundle.compile_metadata.args()) {
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(arg.sharding()));
    executable_bundle.arg_hlo_shardings.push_back(hlo_sharding);
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::ShardingRef ifrt_sharding,
        ToIfrtSharding(*ifrt_client_, hlo_sharding, assigned_device_list_));
    executable_bundle.arg_ifrt_shardings.push_back(std::move(ifrt_sharding));
  }

  if (UsePortableExecution()) {
    // For core selection, the device is selected at runtime. We pre-calculate
    // the sharding for each addressable device to avoid doing it on the
    // critical path. The map is keyed by device ID.
    for (xla::ifrt::Device* device : ifrt_client_->addressable_devices()) {
      executable_bundle.portable_single_device_shardings.emplace(
          device->Id(), xla::ifrt::SingleDeviceSharding::Create(
                            device, xla::ifrt::MemoryKind()));
    }
  }

  executable_bundle.retval_hlo_shardings.reserve(
      executable_bundle.compile_metadata.retvals().size());
  for (const auto& retvals : executable_bundle.compile_metadata.retvals()) {
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(retvals.sharding()));
    executable_bundle.retval_hlo_shardings.push_back(hlo_sharding);
  }

  // Populate compiled parameter shapes and layouts post-packing.
  executable_bundle.compiled_xla_input_shapes.reserve(parameter_layouts.size());
  executable_bundle.compiled_xla_input_layouts.reserve(
      parameter_layouts.size());
  executable_bundle.compiled_ifrt_input_dtypes.reserve(
      parameter_layouts.size());
  executable_bundle.compiled_ifrt_input_shapes.reserve(
      parameter_layouts.size());

  const int unpacked_size = tf2hlo_result.compile_metadata.args().size();
  int packed_inputs_count = 0;
  if (!executable_bundle.pack_group_ids.empty()) {
    for (int i = 0; i < unpacked_size; ++i) {
      if (executable_bundle.pack_group_ids[i] >= 0) {
        packed_inputs_count++;
      }
    }
  }
  const int total_individuals = unpacked_size - packed_inputs_count;

  std::vector<int64_t> unique_group_ids;
  if (!executable_bundle.pack_group_ids.empty()) {
    for (int i = 0; i < unpacked_size; ++i) {
      if (executable_bundle.pack_group_ids[i] >= 0) {
        if (std::find(unique_group_ids.begin(), unique_group_ids.end(),
                      executable_bundle.pack_group_ids[i]) ==
            unique_group_ids.end()) {
          unique_group_ids.push_back(executable_bundle.pack_group_ids[i]);
        }
      }
    }
    std::sort(unique_group_ids.begin(), unique_group_ids.end());
  }

  for (int c = 0; c < parameter_layouts.size(); ++c) {
    std::shared_ptr<const xla::Shape> c_xla_shape = nullptr;
    xla::ifrt::LayoutRef c_xla_layout;
    xla::ifrt::DType c_ifrt_dtype(xla::ifrt::DType::kInvalid);
    std::shared_ptr<const xla::ifrt::Shape> c_ifrt_shape = nullptr;

    if (c < total_individuals) {
      int individual_count = 0;
      int original_idx = -1;
      for (int i = 0; i < unpacked_size; ++i) {
        bool matches_individual = true;
        if (!executable_bundle.pack_group_ids.empty() &&
            executable_bundle.pack_group_ids[i] >= 0) {
          matches_individual = false;
        }
        if (matches_individual) {
          if (individual_count == c) {
            original_idx = i;
            break;
          }
          individual_count++;
        }
      }
      if (original_idx >= 0 &&
          original_idx < executable_bundle.xla_input_shapes.size()) {
        c_xla_shape = executable_bundle.xla_input_shapes[original_idx];
      }
      if (parameter_layouts[c]) {
        c_xla_layout = xla::ifrt::PjRtLayout::Create(parameter_layouts[c]);
      }
      if (original_idx >= 0 &&
          original_idx < executable_bundle.ifrt_input_dtypes.size()) {
        c_ifrt_dtype = executable_bundle.ifrt_input_dtypes[original_idx];
        c_ifrt_shape = executable_bundle.ifrt_input_shapes[original_idx];
      }
    } else {
      int group_idx = c - total_individuals;
      int64_t gid = unique_group_ids[group_idx];
      int64_t total_packed_size = 0;
      for (int i = 0; i < unpacked_size; ++i) {
        if (!executable_bundle.pack_group_ids.empty() &&
            executable_bundle.pack_group_ids[i] == gid) {
          int64_t bytes_per_elt =
              DtypeByteSize(tf2hlo_result.compile_metadata.args(i).dtype());
          int64_t num_elements =
              executable_bundle.reshaped_input_tensors[i].num_elements();
          if (bytes_per_elt > 0 && num_elements > 0) {
            int64_t offset = executable_bundle.pack_offsets[i];
            total_packed_size = std::max(total_packed_size,
                                         offset + num_elements * bytes_per_elt);
          }
        }
      }
      xla::PrimitiveType primitive_type;
      xla::ifrt::DType ifrt_type(xla::ifrt::DType::kInvalid);
      int64_t element_count = total_packed_size / gid;
      if (gid == 1) {
        primitive_type = xla::PrimitiveType::S8;
        ifrt_type = xla::ifrt::DType(xla::ifrt::DType::kS8);
      } else if (gid == 2) {
        primitive_type = xla::PrimitiveType::S16;
        ifrt_type = xla::ifrt::DType(xla::ifrt::DType::kS16);
      } else if (gid == 4) {
        primitive_type = xla::PrimitiveType::S32;
        ifrt_type = xla::ifrt::DType(xla::ifrt::DType::kS32);
      } else if (gid == 8) {
        primitive_type = xla::PrimitiveType::S64;
        ifrt_type = xla::ifrt::DType(xla::ifrt::DType::kS64);
      } else {
        return absl::InternalError(
            "Unsupported pack group gid, must be 1, 2, 4 or 8");
      }

      xla::Shape packed_shape =
          xla::ShapeUtil::MakeShape(primitive_type, {element_count});
      if (parameter_layouts[c]) {
        *packed_shape.mutable_layout() = parameter_layouts[c]->xla_layout();
      } else {
        *packed_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0});
      }

      c_xla_shape = std::make_shared<xla::Shape>(packed_shape);
      if (parameter_layouts[c]) {
        c_xla_layout = xla::ifrt::PjRtLayout::Create(parameter_layouts[c]);
      }
      c_ifrt_dtype = ifrt_type;
      c_ifrt_shape = std::make_shared<const xla::ifrt::Shape>(
          xla::ifrt::Shape({element_count}));
    }

    executable_bundle.compiled_xla_input_shapes.push_back(
        std::move(c_xla_shape));
    executable_bundle.compiled_xla_input_layouts.push_back(
        std::move(c_xla_layout));
    executable_bundle.compiled_ifrt_input_dtypes.push_back(c_ifrt_dtype);
    executable_bundle.compiled_ifrt_input_shapes.push_back(
        std::move(c_ifrt_shape));
  }

  return absl::OkStatus();
}

absl::StatusOr<IfrtServingExecutable::SharedCachedExecutableBundle>
IfrtServingExecutable::CreateExecutableSynchronously(
    mlir::OwningOpRef<mlir::ModuleOp> module_copy,
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices) {
  TF_ASSIGN_OR_RETURN(auto host_callback_modules,
                      GetHostCallbackModulesAndRemoveHostFuncs(*module_copy));
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("module_for_bridge_phase2", *module_copy);
  }

  Tf2HloArg tf2hlo_arg{
      .module = module_copy.get(),
      .input_dtypes_and_shapes = std::vector<DtypeAndShape>(
          dtypes_and_shapes.begin(), dtypes_and_shapes.end()),
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = signature_name(),
      .compile_metadata = compile_metadata,
      .shape_representation_fn = shape_representation_fn_,
      .platform_name = ifrt_client_->platform_name(),
  };

  // Only get device topology for clients that implement GetTopologyForDevices.
  if (tf2hlo_arg.platform_name != xla::CudaName() &&
      !absl::StartsWith(ifrt_client_->runtime_type(), "proxy/")) {
    TF_ASSIGN_OR_RETURN(
        tf2hlo_arg.topology,
        ifrt_client_->GetTopologyForDevices(assigned_device_list_));
  }

  TF_ASSIGN_OR_RETURN(Tf2HloResult tf2hlo_result,
                      persistent_compilation_cache_->LookupTf2HloResultOrCreate(
                          tf2hlo_arg, tf_to_hlo_compiler_));
  if (VLOG_IS_ON(1)) {
    xla::DumpHloModuleProtoIfEnabled(tf2hlo_result.hlo_module_proto,
                                     "before_ifrt_serialization");
  }

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_hlo_module,
      ::xla::ConvertHloToStablehloWithOptions(
          *module_copy->getContext(), &tf2hlo_result.hlo_module_proto,
          /*import_all_computations=*/false));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_bridge_phase2",
                                 mlir_hlo_module.get());
  }

  // IFRT pack-inputs: if the propagator stamped a plan on the atom @main of
  // module_copy (TF-dialect side, pre-tf2hlo), invoke PackInputsPass on the
  // post-tf2hlo stablehlo module to rewrite the executable's signature to
  // take a single packed tensor<Nxi8> in place of the group-marked operands.
  //
  // Reading the plan from module_copy (not mlir_hlo_module) because tf2hlo
  // doesn't carry the propagator's func-level attrs onto the HLO output.
  std::vector<SliceInfo> slices;
  // Temporarily disable reading plan from IR.
  /*
  if (auto main_func = module_copy->lookupSymbol<mlir::func::FuncOp>("main")) {
    auto group_ids_attr =
        main_func->getAttrOfType<mlir::ArrayAttr>(kIfrtPackGroupIdsAttr);
    auto offsets_attr =
        main_func->getAttrOfType<mlir::ArrayAttr>(kIfrtPackOffsetsAttr);
    if (group_ids_attr && offsets_attr &&
        group_ids_attr.size() == offsets_attr.size() &&
        static_cast<size_t>(main_func.getNumArguments()) ==
            group_ids_attr.size()) {
      auto hlo_main_func =
          mlir_hlo_module->lookupSymbol<mlir::func::FuncOp>("main");
      if (hlo_main_func) {
        for (size_t i = 0; i < group_ids_attr.size(); ++i) {
          int64_t gid =
              llvm::cast<mlir::IntegerAttr>(group_ids_attr[i]).getInt();
          if (gid < 0) continue;
          int64_t offset =
              llvm::cast<mlir::IntegerAttr>(offsets_attr[i]).getInt();
          // Read type from the HLO module, where bounded dynamic shapes have
          // been specialized to static.
          auto arg_type = llvm::dyn_cast<mlir::RankedTensorType>(
              hlo_main_func.getArgument(i).getType());
          if (!arg_type || !arg_type.hasStaticShape()) continue;
          int64_t bytes_per_elt = ElementByteSize(arg_type.getElementType());
          if (bytes_per_elt < 0) continue;
          int64_t bytes = arg_type.getNumElements() * bytes_per_elt;
          slices.push_back({static_cast<unsigned>(i), offset, bytes, gid});
        }
      }
    }
  }
  */

  if (slices.empty()) {
    // Build per-arg DtypeAndShape from concrete compile_metadata. This is
    // the path that lights up for vanilla dynamic-shape models (e.g.
    // recommenders with `<?x?>` quantized inputs).
    std::vector<DtypeAndShape> args_ds;
    args_ds.reserve(compile_metadata.args_size());
    for (int i = 0; i < compile_metadata.args_size(); ++i) {
      const auto& arg = compile_metadata.args(i);
      DtypeAndShape ds;
      ds.dtype = arg.dtype();
      if (!tensorflow::TensorShape::BuildTensorShape(arg.shape(), &ds.shape)
               .ok()) {
        ds.shape = tensorflow::TensorShape();
      }
      args_ds.push_back(std::move(ds));
    }
    auto main_func =
        module_copy->lookupSymbol<mlir::func::FuncOp>(signature_name());
    if (!main_func)
      main_func = module_copy->lookupSymbol<mlir::func::FuncOp>("main");

    if (absl::EqualsIgnoreCase(ifrt_client_->platform_name(), "tpu")) {
      TF_ASSIGN_OR_RETURN(
          PackPlan plan,
          ComputePackPlanFromInputs(args_ds, variable_arg_indices, main_func));
      LOG_EVERY_N_SEC(INFO, 20)
          << "IFRT Pack-Inputs: Computed plan from inputs. "
          << "Args size: " << plan.pack_group_ids.size();
      for (size_t i = 0; i < plan.pack_group_ids.size(); ++i) {
        if (plan.pack_group_ids[i] >= 0) {
          LOG_EVERY_N_SEC(INFO, 20)
              << "  Arg " << i << " -> Group " << plan.pack_group_ids[i]
              << " Offset " << plan.pack_offsets[i];
        }
      }
      for (int i = 0; i < static_cast<int>(plan.pack_group_ids.size()); ++i) {
        if (plan.pack_group_ids[i] < 0) continue;
        const int64_t num_elements = args_ds[i].shape.num_elements();
        const int64_t bytes_per_elt =
            tensorflow::DataTypeSize(args_ds[i].dtype);
        if (num_elements <= 0 || bytes_per_elt <= 0) continue;
        slices.push_back({static_cast<unsigned>(i), plan.pack_offsets[i],
                          num_elements * bytes_per_elt,
                          plan.pack_group_ids[i]});
      }
      if (!slices.empty()) {
        LOG_EVERY_N_SEC(INFO, 20)
            << "CreateExecutableSynchronously: computed pack plan from "
            << "compile_metadata (no IR-time plan); " << slices.size()
            << " slice(s) of " << compile_metadata.args_size() << " args";
      }
    }
  }

  if (!tf2hlo_result.xla_input_shapes.empty()) {
    TF_RETURN_IF_ERROR(
        EncodeLayout(tf2hlo_result.xla_input_shapes, mlir_hlo_module.get()));
  }

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_layout_encoding_layout_mode",
                                 mlir_hlo_module.get());
  }

  if (!slices.empty()) {
    if (auto hlo_main =
            mlir_hlo_module->lookupSymbol<mlir::func::FuncOp>("main")) {
      if (hlo_main.getNumArguments() == compile_metadata.args_size()) {
        LOG_EVERY_N_SEC(INFO, 20)
            << "IFRT Pack-Inputs: Duplicate Shape Refinement on "
               "mlir_hlo_module main function for program_id "
            << program_id_;
        bool refine_any = false;
        for (int i = 0; i < compile_metadata.args_size(); ++i) {
          auto cur_type = llvm::dyn_cast<mlir::RankedTensorType>(
              hlo_main.getArgument(i).getType());
          if (!cur_type) {
            LOG_EVERY_N_SEC(INFO, 20)
                << "IFRT Pack-Inputs: Skip argument " << i << " for program_id "
                << " due to unranked type"
                << hlo_main.getArgument(i)
                       .getType()
                       .getTypeID()
                       .getAsOpaquePointer();
          }
          if (!cur_type || cur_type.hasStaticShape()) {
            LOG_EVERY_N_SEC(INFO, 20)
                << "IFRT Pack-Inputs: Skip argument " << i << " for program_id "
                << bool(cur_type) << " "
                << (cur_type ? cur_type.hasStaticShape() : false);
            continue;
          }
          const auto& shape_proto = compile_metadata.args(i).shape();
          if (static_cast<int>(shape_proto.dim_size()) != cur_type.getRank()) {
            LOG_EVERY_N_SEC(INFO, 20)
                << "IFRT Pack-Inputs: Skip argument " << i << " for program_id "
                << " due to rank mismatch: " << shape_proto.dim_size() << " vs "
                << cur_type.getRank();
            continue;
          }
          llvm::SmallVector<int64_t> new_shape;
          bool ok = true;
          for (const auto& dim : shape_proto.dim()) {
            if (dim.size() < 0) {
              LOG_EVERY_N_SEC(INFO, 20)
                  << "IFRT Pack-Inputs: Skip argument " << i
                  << " for program_id "
                  << " due to negative dimension size: " << dim.size();
              ok = false;
              break;
            }
            new_shape.push_back(dim.size());
            LOG_EVERY_N_SEC(INFO, 20)
                << "IFRT Pack-Inputs: Argument " << i << " for program_id "
                << program_id_ << " has dimension size: " << dim.size();
          }
          if (!ok) {
            LOG_EVERY_N_SEC(INFO, 20)
                << "IFRT Pack-Inputs: Skip argument " << i << " for program_id "
                << " due to invalid dimension size";
            continue;
          }
          hlo_main.getArgument(i).setType(mlir::RankedTensorType::get(
              new_shape, cur_type.getElementType()));
          refine_any = true;
        }
        if (refine_any) {
          LOG_EVERY_N_SEC(INFO, 20)
              << "IFRT Pack-Inputs: Refine argument types on hlo_main "
                 "for program_id "
              << program_id_;
          llvm::SmallVector<mlir::Type, 4> new_input_types;
          for (const auto& arg : hlo_main.getArguments()) {
            new_input_types.push_back(arg.getType());
          }
          hlo_main.setType(mlir::FunctionType::get(
              mlir_hlo_module->getContext(), new_input_types,
              hlo_main.getFunctionType().getResults()));
          LOG_EVERY_N_SEC(INFO, 20)
              << "IFRT Pack-Inputs: Refined argument shapes on hlo_main "
                 "for program_id "
              << program_id_;
        }
      } else {
        LOG_EVERY_N_SEC(INFO, 20)
            << "IFRT Pack-Inputs: Skip argument type refinement on "
               "hlo_main for program_id "
            << program_id_
            << " due to argument size mismatch: " << hlo_main.getNumArguments()
            << " vs " << compile_metadata.args_size();
      }
    }

    LOG_EVERY_N_SEC(INFO, 20)
        << "IFRT Pack-Inputs: Invoking RefineShapesPass and "
           "PackInputsPass with "
        << slices.size() << " slices on submodule for program_id "
        << program_id_;
    mlir::PassManager pm(mlir_hlo_module->getContext());
    pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
    pm.addPass(CreatePackInputsPass(slices));
    if (mlir::failed(pm.run(*mlir_hlo_module))) {
      return absl::InternalError(
          "PackInputsPass failed on stablehlo atom module");
    }
    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("ifrt_after_pack_inputs",
                                   mlir_hlo_module.get());
    }
  }

  const int num_replicas = tf2hlo_result.compile_metadata.num_replicas();
  const int num_partitions =
      tf2hlo_result.compile_metadata.num_cores_per_replica();

  LOG_EVERY_N_SEC(INFO, 20) << " Number of replcas is " << num_replicas
                            << " and num_partitions is " << num_partitions;

  if (num_replicas > 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only support single replica, but replica number is ",
                     num_replicas, " and num_partitions is ", num_partitions));
  }

  xla::CompileOptions xla_compile_options;

  if (std::holds_alternative<tsl::protobuf::Message*>(
          compilation_env_or_overrides_)) {
    tsl::protobuf::Message* compilation_environment_proto_ =
        std::get<tsl::protobuf::Message*>(compilation_env_or_overrides_);
    if (compilation_environment_proto_) {
      tsl::protobuf::Message* comp_env_copy =
          compilation_environment_proto_->New();
      comp_env_copy->CopyFrom(*compilation_environment_proto_);
      TF_RETURN_IF_ERROR(
          xla_compile_options.executable_build_options.mutable_comp_envs()
              ->AddEnv(
                  absl::WrapUnique<tsl::protobuf::Message>(comp_env_copy)));
    }
  } else if (std::holds_alternative<
                 xla::CompileOptions::EnvironmentOptionOverrides>(
                 compilation_env_or_overrides_)) {
    xla_compile_options.env_option_overrides =
        std::get<xla::CompileOptions::EnvironmentOptionOverrides>(
            compilation_env_or_overrides_);
  } else {
    return absl::NotFoundError(
        "Either compilation_environment_proto or env_option_overrides is "
        "expected.");
  }

  xla_compile_options.executable_build_options.set_num_replicas(num_replicas);
  xla_compile_options.executable_build_options.set_num_partitions(
      num_partitions);

  xla_compile_options.executable_build_options.set_use_spmd_partitioning(
      original_compile_metadata_.use_spmd_for_xla_partitioning());
  xla_compile_options.executable_build_options.set_use_shardy_partitioner(
      compile_metadata.use_shardy_partitioner());
  xla_compile_options.parameter_is_tupled_arguments = false;
  // Use portable execution for single device + core selection.
  if (UsePortableExecution()) {
    xla_compile_options.compile_portable_executable = true;
  } else {
    TF_ASSIGN_OR_RETURN(
        xla::DeviceAssignment da,
        GetRuntimeXlaDeviceAssignment(assigned_device_list_, num_replicas,
                                      num_partitions));
    LOG_EVERY_N_SEC(INFO, 20) << "Device assignment :" << da.ToString();
    xla_compile_options.executable_build_options.set_device_assignment(da);
  }

  std::vector<std::unique_ptr<TfHostCallback>> tf_host_callbacks;
  TF_ASSIGN_OR_RETURN(auto host_callbacks,
                      BuildHostCallbacks(tf2hlo_result.host_compute_metadata,
                                         std::move(host_callback_modules),
                                         device_mgr_, tf_host_callbacks));

  std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
      loaded_host_callbacks;
  loaded_host_callbacks.reserve(host_callbacks.size());
  for (const auto& host_callback : host_callbacks) {
    loaded_host_callbacks.push_back(
        tsl::MakeRef<xla::ifrt::PjRtHostSendAndRecvLoadedHostCallback>(
            ifrt_client_.get(),
            std::make_unique<xla::HostCallback>(host_callback)));
  }
  auto hlo_program =
      std::make_unique<xla::ifrt::HloProgram>(mlir_hlo_module.get());
  SharedCachedExecutableBundle executable_bundle =
      std::make_shared<CachedExecutableBundle>();

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutableRef ifrt_executable,
      persistent_compilation_cache_->LookupLoadedExecutableOrCreate(
          std::move(hlo_program), assigned_device_list_, xla_compile_options,
          loaded_host_callbacks, ifrt_client_.get(),
          [&](std::unique_ptr<xla::ifrt::Program> program,
              std::unique_ptr<xla::ifrt::CompileOptions> options)
              -> absl::StatusOr<xla::ifrt::LoadedExecutableRef> {
            return ifrt_client_->GetDefaultCompiler()
                ->CompileAndLoad(std::move(program), std::move(options))
                .Await();
          }));

  if (!slices.empty()) {
    const int n_args = tf2hlo_result.compile_metadata.args_size();
    executable_bundle->pack_group_ids.assign(n_args, -1);
    executable_bundle->pack_offsets.assign(n_args, 0);
    for (const auto& s : slices) {
      // Use the original input index s.arg_index to match the lookup in
      // ExecuteCore, which uses the original index to access pack_group_ids.
      // Using compiled_arg_idx caused a mismatch by shifting indices.
      if (s.arg_index < n_args) {
        executable_bundle->pack_group_ids[s.arg_index] = s.group_id;
        executable_bundle->pack_offsets[s.arg_index] = s.start;
      }
    }
    LOG_EVERY_N_SEC(INFO, 20)
        << "IFRT Pack-Inputs: Stored bundle plan. "
        << "pack_group_ids size: " << executable_bundle->pack_group_ids.size();
    for (size_t i = 0; i < executable_bundle->pack_group_ids.size(); ++i) {
      if (executable_bundle->pack_group_ids[i] >= 0) {
        LOG_EVERY_N_SEC(INFO, 20) << "  Arg " << i << " -> Group "
                                  << executable_bundle->pack_group_ids[i];
      }
    }
  }

  TF_RETURN_IF_ERROR(PopulateInvariantMetadata(
      tf2hlo_result, std::move(ifrt_executable), std::move(tf_host_callbacks),
      tf2hlo_arg.topology.get(), *executable_bundle));

  return executable_bundle;
}

absl::Status IfrtServingExecutable::LoadAndRegisterVariableOnExecutable(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const xla::ifrt::DeviceListRef& device_list,
    IfrtServingExecutable::CachedExecutableBundle* executable_bundle) {
  TF_RETURN_IF_ERROR(AsyncLoadIfrtArray(inputs, variable_arg_indices,
                                        *executable_bundle, device_list));

  std::vector<int> device_ids;
  device_ids.reserve(device_list->size());
  for (xla::ifrt::Device* device : device_list->devices()) {
    device_ids.push_back(device->Id().value());
  }

  for (const int i : variable_arg_indices) {
    if (inputs[i].dtype() != tensorflow::DT_STRING ||
        !tensorflow::TensorShapeUtils::IsScalar(inputs[i].shape())) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected a scalar tensor as loaded variable array key, "
                       "but got type ",
                       inputs[i].dtype(), " and shape ",
                       inputs[i].shape().DebugString(), " at index ", i));
    }
    std::string tensor_name = inputs[i].scalar<tsl::tstring>()();
    IfrtLoadedVariableRegistry::Key key{
        .device_ids = device_ids,
        .input_name = tensor_name,
        .hlo_sharding = executable_bundle->arg_hlo_shardings[i],
        .shape_on_device = executable_bundle->xla_input_shapes[i],
    };
    TF_ASSIGN_OR_RETURN(auto loaded_variable,
                        ifrt_loaded_variable_registry_.GetLoadedVariable(key));
    executable_bundle->variable_arrays.emplace(key, std::move(loaded_variable));
  }
  return absl::OkStatus();
}

absl::StatusOr<tsl::Future<IfrtServingExecutable::SharedCachedExecutableBundle>>
IfrtServingExecutable::LookUpOrCreateExecutable(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices,
    const xla::ifrt::DeviceListRef& device_list) {

  tsl::Promise<SharedCachedExecutableBundle> promise;
  tsl::Future<SharedCachedExecutableBundle> future;
  mlir::OwningOpRef<mlir::ModuleOp> module_copy;
  tsl::Future<SharedCachedExecutableBundle> existing_bundle_future;
  {
    absl::MutexLock lock(mutex_);

    const auto it = executable_bundles_.find(KeyView{dtypes_and_shapes});
    if (it != executable_bundles_.end()) {
      return it->second;
    }

    if (is_frozen_ || tf_to_hlo_compiler_->IsXlaCompilationDisabled()) {
      // Build a description of the requested (offending) input shapes.
      std::string requested_shapes_str;
      for (size_t i = 0; i < dtypes_and_shapes.size(); ++i) {
        absl::StrAppend(
            &requested_shapes_str, "[", i,
            "]: ", dtypes_and_shapes[i].GetShapeForCompilation().DebugString());
        if (i + 1 < dtypes_and_shapes.size()) {
          absl::StrAppend(&requested_shapes_str, ", ");
        }
      }
      // Build a description of all cached (already compiled) shape sets.
      std::string cached_shapes_str;
      int key_idx = 0;
      for (const auto& [key, unused_future] : executable_bundles_) {
        absl::StrAppend(&cached_shapes_str, "  compiled[", key_idx++, "]: {");
        for (size_t i = 0; i < key.input_shapes.size(); ++i) {
          absl::StrAppend(&cached_shapes_str,
                          key.input_shapes[i].DebugString());
          if (i + 1 < key.input_shapes.size()) {
            absl::StrAppend(&cached_shapes_str, ", ");
          }
        }
        absl::StrAppend(&cached_shapes_str, "}\n");
      }
      tsl::Future<SharedCachedExecutableBundle> frozen_future(
          absl::FailedPreconditionError(absl::StrCat(
              "Cannot compile for new input shapes. Either the executable is "
              "already frozen: ",
              is_frozen_,
              " or XLA compilation disabled by ScopedTpuCompileDisabler: ",
              tf_to_hlo_compiler_->IsXlaCompilationDisabled(),
              ". Requested input shapes: {", requested_shapes_str,
              "}. Number of already compiled shape sets: ",
              executable_bundles_.size(),
              cached_shapes_str.empty()
                  ? ""
                  : absl::StrCat(". Already compiled:\n", cached_shapes_str))));
      return frozen_future;
    }

    // Only create promise and future when cache missed.
    std::tie(promise, future) =
        tsl::MakePromise<SharedCachedExecutableBundle>();

    std::vector<tensorflow::TensorShape> input_shapes;
    input_shapes.reserve(dtypes_and_shapes.size());
    for (const auto& dtype_and_shape : dtypes_and_shapes) {
      input_shapes.push_back(dtype_and_shape.GetShapeForCompilation());
    }
    Key key = {.input_shapes = std::move(input_shapes)};

    executable_bundles_.emplace(std::move(key), future);
    // Clone the module to avoid race condition between Freeze() and
    // compilation.
    module_copy = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
  }

  LOG_EVERY_N_SEC(INFO, 20) << "Cache missed. Building executable";

  tensorflow::tpu::TPUCompileMetadataProto compile_metadata =
      original_compile_metadata_;

  // b/469105465: Add test coverage for core selection in execution.
  if (UsePortableExecution()) {
    // Clear device_assignment because portable execution doesn't allow device
    // assignment.
    compile_metadata.clear_device_assignment();
  }

  TF_RETURN_IF_ERROR(
      UpdateCompileMetadata(compile_metadata, dtypes_and_shapes));

  std::vector<SliceInfo> slice_info;
  for (int i = 0; i < compile_metadata.args().size(); ++i) {
    const auto& arg = compile_metadata.args(i);
    std::string shape_str = arg.shape().DebugString();
    LOG_EVERY_N_SEC(INFO, 20) << "<<<<<<<<<<<<<<< arg " << i << ": "
                              << shape_str << " " << arg.kind();
  }

  absl::StatusOr<SharedCachedExecutableBundle> executable_bundle =
      CreateExecutableSynchronously(std::move(module_copy), compile_metadata,
                                    dtypes_and_shapes, variable_arg_indices);

  if (!executable_bundle.ok()) {
    promise.Set(executable_bundle.status());
    return executable_bundle.status();
  }

  if (UsePortableExecution()) {
    // If core selector is enabled, we load variables on all cores.
    for (const auto& device : ifrt_client_->addressable_devices()) {
      TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef selected_device_list,
                          ifrt_client_->MakeDeviceList({device}));
      TF_RETURN_IF_ERROR(LoadAndRegisterVariableOnExecutable(
          inputs, variable_arg_indices, selected_device_list,
          (*executable_bundle).get()));
    }
  } else {
    TF_RETURN_IF_ERROR(LoadAndRegisterVariableOnExecutable(
        inputs, variable_arg_indices, device_list, (*executable_bundle).get()));
  }

  promise.Set(std::move(executable_bundle));

  // Here is a immediate return as promise is already set.
  return future;
}

void IfrtServingExecutable::Freeze() {
  LOG_EVERY_N_SEC(INFO, 20)
      << "Freezing executable. Program id: " << program_id_;
  absl::MutexLock lock(mutex_);
  is_frozen_ = true;
  module_ = nullptr;
}

bool IfrtServingExecutable::UsePortableExecution() {
  // TODO(b/335247101) Add a check that the core selector must be non-null if
  // it is a single-device program after core selection in Ifrt is stable.
  return IsSingleDevice(original_compile_metadata_) &&
         ifrt_serving_core_selector_;
}

absl::StatusOr<IfrtServingExecutable::ExecutionInfo>
IfrtServingExecutable::ExecuteCore(absl::Span<const tensorflow::Tensor> inputs,
                                   absl::Span<const int> variable_arg_indices,
                                   absl::Span<const int64_t> pack_group_ids,
                                   absl::Span<const int64_t> pack_offsets) {
  tsl::profiler::TraceMe traceme("IfrtServingExecutable::Execute");

  LOG_EVERY_N_SEC(INFO, 20)
      << "IFRT pack-inputs coalescing is "
      << (pack_group_ids.empty() ? "INACTIVE" : "ACTIVE")
      << " for program_id=" << program_id_ << " inputs=" << inputs.size();

  // Determine whether the caller passed a usable pack-inputs plan.
  bool is_packed = false;
  if (!pack_group_ids.empty()) {
    for (int64_t g : pack_group_ids) {
      if (g >= 0) {
        is_packed = true;
        LOG_EVERY_N_SEC(INFO, 20)
            << "IFRT pack-inputs coalescing is ACTIVE for program_id="
            << program_id_ << " inputs=" << inputs.size();
        break;
      }
    }
    LOG_EVERY_N_SEC(INFO, 20)
        << "IFRT pack-inputs coalescing is "
        << (is_packed ? "ACTIVE" : "INACTIVE")
        << " for program_id=" << program_id_ << " inputs=" << inputs.size();
    if (is_packed) {
      if (pack_group_ids.size() != inputs.size() ||
          pack_offsets.size() != inputs.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "pack annotation size mismatch: inputs=", inputs.size(),
            " group_ids=", pack_group_ids.size(),
            " offsets=", pack_offsets.size()));
      }
    }
  }

  for (int i = 1; i < variable_arg_indices.size(); i++) {
    if (variable_arg_indices[i] <= variable_arg_indices[i - 1]) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Expected variable_arg_indices in ascending order. But subsequence "
          "starting at ",
          i - 1, ": (", variable_arg_indices[i - 1], ", ",
          variable_arg_indices[i], ")", " is not in ascending order"));
    }
  }

  if (!variable_arg_indices.empty() &&
      inputs.size() <= variable_arg_indices.back()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected at most ", inputs.size(), " inputs, but got up to ",
        variable_arg_indices.back(), " variables."));
  }

  // Ensure the variable tensor holds a valid key: a scalar string tensor.
  for (const int i : variable_arg_indices) {
    if (inputs[i].dtype() != tensorflow::DT_STRING ||
        !tensorflow::TensorShapeUtils::IsScalar(inputs[i].shape())) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected a scalar tensor as loaded variable array key, "
                       "but got type ",
                       inputs[i].dtype(), " and shape ",
                       inputs[i].shape().DebugString(), " at index ", i));
    }
  }

  TF_ASSIGN_OR_RETURN(StaticShapeMap static_shapes_map,
                      GetStaticShapesFromInputs(inputs, static_shape_arg_map_));
  TF_ASSIGN_OR_RETURN(
      std::vector<DtypeAndShape> dtypes_and_shapes,
      BuildDtypeAndShape(inputs, variable_arg_indices, static_shapes_map,
                         ifrt_restore_tensor_registry_));

  // `device_reservation` should be alive before the end of the execution.
  tsl::DeviceReservation device_reservation(kNoCoreSelectedIndex, nullptr);
  xla::ifrt::DeviceListRef device_list;
  if (UsePortableExecution()) {
    device_reservation =
        ifrt_serving_core_selector_->ReserveDevice(program_id_);
    TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(
                            device_reservation.device_index())));
    TF_ASSIGN_OR_RETURN(device_list, ifrt_client_->MakeDeviceList({device}));
  } else {
    device_list = assigned_device_list_;
  }
  TF_ASSIGN_OR_RETURN(
      tsl::Future<SharedCachedExecutableBundle> executable_bundle_future,
      LookUpOrCreateExecutable(inputs, dtypes_and_shapes, variable_arg_indices,
                               device_list));
  TF_ASSIGN_OR_RETURN(SharedCachedExecutableBundle executable_bundle,
                      executable_bundle_future.Await());

  if (!is_packed && !executable_bundle->pack_group_ids.empty()) {
    if (executable_bundle->pack_group_ids.size() != inputs.size() ||
        executable_bundle->pack_offsets.size() != inputs.size()) {
      return absl::InternalError(absl::StrCat(
          "Bundle pack annotation size mismatch: inputs=", inputs.size(),
          " group_ids=", executable_bundle->pack_group_ids.size(),
          " offsets=", executable_bundle->pack_offsets.size()));
    }
    pack_group_ids = absl::MakeConstSpan(executable_bundle->pack_group_ids);
    pack_offsets = absl::MakeConstSpan(executable_bundle->pack_offsets);
    LOG_EVERY_N_SEC(INFO, 20)
        << "IFRT Pack-Inputs: Adopted plan for execution. "
        << "pack_group_ids size: " << pack_group_ids.size();
    for (size_t i = 0; i < pack_group_ids.size(); ++i) {
      if (pack_group_ids[i] >= 0) {
        LOG_EVERY_N_SEC(INFO, 20)
            << "  Input " << i << " -> Group " << pack_group_ids[i]
            << " Offset " << pack_offsets[i];
      }
    }
    for (int64_t g : pack_group_ids) {
      if (g >= 0) {
        is_packed = true;
        break;
      }
    }
    LOG_EVERY_N_SEC(INFO, 20) << "Adopted bundle pack-inputs plan (sync path); "
                              << "pack_group_ids.size=" << pack_group_ids.size()
                              << " is_packed=" << is_packed;
  }

  // With packing, the executable's signature is rewritten to be SHORTER than
  // the original input list. The arity check must allow this.
  if (!is_packed && executable_bundle->compile_metadata.args().size() !=
                        dtypes_and_shapes.size()) {
    return absl::InternalError(absl::StrCat(
        "Expected ", executable_bundle->compile_metadata.args().size(),
        " but got ", dtypes_and_shapes.size(), " arguments"));
  }

  // Determine the effective device IDs for this execution.
  std::vector<int> portable_device_ids;
  if (UsePortableExecution()) {
    portable_device_ids.reserve(device_list->size());
    for (xla::ifrt::Device* device : device_list->devices()) {
      portable_device_ids.push_back(device->Id().value());
    }
  }
  absl::Span<const int> effective_device_ids =
      UsePortableExecution() ? portable_device_ids : assigned_device_ids_;
  int variable_arg_index = 0;
  std::vector<tsl::Future<xla::ifrt::ArrayRef>> variable_args;
  variable_args.reserve(variable_arg_indices.size());

  LOG_EVERY_N_SEC(INFO, 20)
      << "<<<< IFRT Pack-Inputs: Using H2DTransferExecutor";
  // Create H2DTransferExecutor
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<H2DTransferExecutor> executor,
      h2d_transfer_executor_factory_->CreateH2DTransferExecutor(*ifrt_client_));

  // Prepare InputHandles
  std::vector<InputHandle> handles;
  handles.reserve(inputs.size());

  std::vector<int> handle_to_input_idx;
  handle_to_input_idx.reserve(inputs.size());

  absl::flat_hash_map<xla::ifrt::Device*, absl::flat_hash_set<int64_t>>
      seen_groups_per_device;
  int packed_inputs_count = 0;

  // Find total individuals (variables + kept non-packed inputs)
  if (is_packed) {
    for (int i = 0; i < inputs.size(); ++i) {
      if (pack_group_ids[i] >= 0) {
        packed_inputs_count++;
      }
    }
  }
  const int total_individuals = inputs.size() - packed_inputs_count;
  LOG_EVERY_N_SEC(INFO, 20) << "<<<< total_individuals: " << total_individuals;

  // Collect unique group IDs to sort them
  std::vector<int64_t> unique_group_ids;
  if (is_packed) {
    for (int i = 0; i < inputs.size(); ++i) {
      if (pack_group_ids[i] >= 0) {
        if (std::find(unique_group_ids.begin(), unique_group_ids.end(),
                      pack_group_ids[i]) == unique_group_ids.end()) {
          unique_group_ids.push_back(pack_group_ids[i]);
        }
      }
    }
    std::sort(unique_group_ids.begin(), unique_group_ids.end());
  }

  variable_arg_index = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      // Variables are handled separately.
      IfrtLoadedVariableRegistry::KeyView key_view(
          effective_device_ids, inputs[i].scalar<tsl::tstring>()(),
          executable_bundle->arg_hlo_shardings[i],
          executable_bundle->xla_input_shapes[i]);
      auto it = executable_bundle->variable_arrays.find(key_view);
      if (it == executable_bundle->variable_arrays.end()) {
        return absl::InternalError(absl::StrCat(
            "Variable array not found for key: ", key_view.input_name));
      }
      variable_args.push_back((*it).second.array);
      variable_arg_index++;
      continue;
    }

    int64_t gid = !pack_group_ids.empty() ? pack_group_ids[i] : -1;
    int64_t offset = !pack_group_ids.empty() ? pack_offsets[i] : 0;

    bool is_representative = false;
    if (gid >= 0) {
      for (auto* device :
           executable_bundle->arg_ifrt_shardings[i]->devices()->devices()) {
        if (seen_groups_per_device[device].insert(gid).second) {
          is_representative = true;
        }
      }
    }

    tensorflow::Tensor reshaped = inputs[i];
    if (gid < 0) {
      // Individual tensor. Apply reshaping for static shapes if needed.
      const tensorflow::TensorShape& reshaped_shape =
          executable_bundle->reshaped_input_tensors[i];
      if (reshaped.shape() != reshaped_shape) {
        if (dtypes_and_shapes[i].static_shape.has_value()) {
          TF_ASSIGN_OR_RETURN(reshaped, MaybeReshapeInputForStaticShape(
                                            reshaped, reshaped_shape));
        } else {
          if (!reshaped.CopyFrom(inputs[i], reshaped_shape)) {
            return absl::InternalError("Failed to reshape tensor");
          }
        }
      }
    }

    xla::ifrt::ShardingRef handle_ifrt_sharding;
    std::shared_ptr<const xla::Shape> expected_packed_xla_shape = nullptr;
    xla::ifrt::DType handle_ifrt_dtype =
        executable_bundle->ifrt_input_dtypes[i];
    std::shared_ptr<const xla::ifrt::Shape> handle_ifrt_shape =
        executable_bundle->ifrt_input_shapes[i];
    int sharding_idx = -1;

    if (is_representative) {
      // Find the index of this group in sorted unique_group_ids
      auto it =
          std::find(unique_group_ids.begin(), unique_group_ids.end(), gid);
      int group_idx = std::distance(unique_group_ids.begin(), it);

      sharding_idx = total_individuals + group_idx;
      DCHECK_LT(sharding_idx,
                executable_bundle->compiled_xla_input_shapes.size());

      handle_ifrt_sharding = executable_bundle->arg_ifrt_shardings[i];
      expected_packed_xla_shape =
          executable_bundle->compiled_xla_input_shapes[sharding_idx];
      handle_ifrt_dtype =
          executable_bundle->compiled_ifrt_input_dtypes[sharding_idx];
      handle_ifrt_shape =
          executable_bundle->compiled_ifrt_input_shapes[sharding_idx];
    } else {
      handle_ifrt_sharding = executable_bundle->arg_ifrt_shardings[i];
    }

    InputHandle handle{
        .tensor = std::move(reshaped),
        .ifrt_dtype = handle_ifrt_dtype,
        .ifrt_shape = handle_ifrt_shape,
        .input_xla_shape = executable_bundle->xla_input_shapes[i],
        .device_list = device_list,
        .ifrt_sharding = std::move(handle_ifrt_sharding),
        .xla_input_layout =
            is_representative
                ? executable_bundle->compiled_xla_input_layouts[sharding_idx]
                : executable_bundle->xla_input_layouts[i],
        .byte_strides = executable_bundle->byte_strides[i],
        .pack_group_id = gid,
        .pack_offset = offset,
        .is_pack_group_representative = is_representative,
        .expected_packed_xla_shape = std::move(expected_packed_xla_shape),
    };

    handles.push_back(std::move(handle));
    handle_to_input_idx.push_back(i);
  }

  // Schedule and Run transfers
  TF_ASSIGN_OR_RETURN(auto future,
                      executor->ScheduledH2DTransfers(handles, thread_pool_));
  TF_RETURN_IF_ERROR(executor->RunH2DTransfers());
  TF_ASSIGN_OR_RETURN(std::vector<xla::ifrt::ArrayRef> arrays, future.Await());

  // Assemble transfer_result
  std::vector<xla::ifrt::ArrayRef> transfer_result;
  transfer_result.resize(executable_bundle->compile_metadata.args().size());

  absl::btree_map<int64_t, xla::ifrt::ArrayRef> group_arrays_map;

  variable_arg_index = 0;
  int kept_individual_idx = 0;
  int handle_idx = 0;

  for (int i = 0; i < inputs.size(); i++) {
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      TF_ASSIGN_OR_RETURN(auto array_ref,
                          variable_args[variable_arg_index].Await());
      transfer_result[kept_individual_idx] = std::move(array_ref);
      variable_arg_index++;
      kept_individual_idx++;
    } else {
      int64_t gid = !pack_group_ids.empty() ? pack_group_ids[i] : -1;
      LOG_EVERY_N_SEC(INFO, 10)
          << "IFRT Execute: input " << i << " has pack_group_id=" << gid;
      if (gid < 0) {
        xla::ifrt::ArrayRef array = arrays[handle_idx++];
        transfer_result[kept_individual_idx] = std::move(array);
        kept_individual_idx++;
      } else {
        if (handles[handle_idx].is_pack_group_representative) {
          xla::ifrt::ArrayRef array = arrays[handle_idx];
          group_arrays_map[gid] = array;
        }
        handle_idx++;
      }
    }
  }

  // Populate packed groups at the end of transfer_result.
  for (auto& [gid, array] : group_arrays_map) {
    if (kept_individual_idx >= transfer_result.size()) {
      return absl::InternalError("transfer_result overflow during packing");
    }
    transfer_result[kept_individual_idx++] = std::move(array);
  }

  transfer_result.resize(kept_individual_idx);

  if (is_packed) {
    DCHECK_EQ(transfer_result.size(), kept_individual_idx)
        << "Built " << transfer_result.size() << " ifrt args but expected "
        << kept_individual_idx;
  } else {
    DCHECK_EQ(transfer_result.size(),
              executable_bundle->compile_metadata.args().size())
        << "Built " << transfer_result.size()
        << " ifrt args but executable expects "
        << executable_bundle->compile_metadata.args().size()
        << "; pack-inputs plumbing likely out of sync.";
  }

  LOG_EVERY_N_SEC(INFO, 10)
      << "IFRT Execute: inputs.size()=" << inputs.size()
      << ", compile_metadata.args().size()="
      << executable_bundle->compile_metadata.args().size()
      << ", transfer_result.size()=" << transfer_result.size()
      << ", variable_args.size()=" << variable_args.size()
      << ", group_arrays_map.size()=" << group_arrays_map.size();

  // LOG_EVERY_N_SEC(INFO, 20)<< "Start Execution";

  std::optional<xla::ifrt::DeviceListRef> execution_device_list;
  if (UsePortableExecution()) {
    execution_device_list = device_list;
  }
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult execution_result,
      [&]() -> absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult> {
        tsl::profiler::TraceMe traceme("Execute");
        return executable_bundle->ifrt_executable->Execute(
            absl::MakeSpan(transfer_result), /*options=*/{.fill_status = true},
            std::move(execution_device_list));
      }());

  if (executable_bundle->compile_metadata.retvals().size() !=
      execution_result.outputs.size()) {
    return absl::InternalError(absl::StrCat(
        "Expect ", executable_bundle->compile_metadata.retvals().size(),
        " but got ", execution_result.outputs.size(), " outputs"));
  }

  return ExecutionInfo{
      .execution_result = std::move(execution_result),
      .executable_bundle = std::move(executable_bundle),
      .device_list = std::move(device_list),
      .transfer_result = std::move(transfer_result),
      .device_reservation = std::make_shared<tsl::DeviceReservation>(
          std::move(device_reservation)),
  };
}

absl::StatusOr<std::vector<tensorflow::Tensor>> IfrtServingExecutable::Execute(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    absl::Span<const int64_t> pack_group_ids,
    absl::Span<const int64_t> pack_offsets) {
  TF_ASSIGN_OR_RETURN(
      ExecutionInfo exec_info,
      ExecuteCore(inputs, variable_arg_indices, pack_group_ids, pack_offsets));

  TF_RETURN_IF_ERROR(exec_info.execution_result.status.Await());

  std::vector<tsl::Future<tensorflow::Tensor>> output_futures;
  output_futures.reserve(exec_info.execution_result.outputs.size());
  for (int i = 0; i < exec_info.execution_result.outputs.size(); ++i) {
    output_futures.push_back(MakeTensorFromArray(
        *ifrt_client_, *exec_info.execution_result.outputs[i],
        exec_info.executable_bundle->retval_hlo_shardings[i],
        exec_info.device_list, thread_pool_));
  }

  tsl::Future<std::vector<tensorflow::Tensor>> joined_outputs =
      tsl::JoinFutures(absl::MakeSpan(output_futures));

  // If there are no outputs, JoinFutures returns an invalid future.
  // Return early to avoid calling Await() on it, which would crash.
  if (!joined_outputs.IsValid()) {
    return std::vector<tensorflow::Tensor>();
  }

  return joined_outputs.Await();
}

absl::StatusOr<tsl::Future<std::vector<tensorflow::Tensor>>>
IfrtServingExecutable::ExecuteAsync(absl::Span<const tensorflow::Tensor> inputs,
                                    absl::Span<const int> variable_arg_indices,
                                    absl::Span<const int64_t> pack_group_ids,
                                    absl::Span<const int64_t> pack_offsets) {
  TF_ASSIGN_OR_RETURN(
      ExecutionInfo exec_info,
      ExecuteCore(inputs, variable_arg_indices, pack_group_ids, pack_offsets));

  std::vector<tsl::Future<tensorflow::Tensor>> output_futures;
  output_futures.reserve(exec_info.execution_result.outputs.size());
  for (int i = 0; i < exec_info.execution_result.outputs.size(); ++i) {
    output_futures.push_back(MakeTensorFromArray(
        *ifrt_client_, *exec_info.execution_result.outputs[i],
        exec_info.executable_bundle->retval_hlo_shardings[i],
        exec_info.device_list, thread_pool_));
  }

  if (output_futures.empty()) {
    // Special case for no returns: we must wait for status.
    // Map creates a promise internally, but it's only for this rare case.
    tsl::Future<> exec_status = exec_info.execution_result.status;
    return exec_status.Map<std::vector<tensorflow::Tensor>>(
        [exec_info =
             std::move(exec_info)]() -> std::vector<tensorflow::Tensor> {
          return std::vector<tensorflow::Tensor>();
        });
  }

  tsl::Future<> status_future = std::move(exec_info.execution_result.status);
  tsl::Future<std::vector<tensorflow::Tensor>> final_future =
      tsl::JoinFutures(absl::MakeSpan(output_futures));

  // Fast path: if both are ready, offload cleanup to thread pool immediately.
  if (final_future.IsReady() && status_future.IsReady()) {
    thread_pool_.Schedule([exec_info = std::move(exec_info)]() mutable {});
    return final_future;
  }

  return status_future.Map([final_future = std::move(final_future),
                            exec_info = std::move(exec_info),
                            thread_pool = &thread_pool_]() mutable
                               -> tsl::Future<std::vector<tensorflow::Tensor>> {
    return final_future.Map(
        [exec_info = std::move(exec_info),
         thread_pool](std::vector<tensorflow::Tensor> outputs) mutable {
          // Offload cleanup to thread pool to avoid blocking execution
          // thread.
          thread_pool->Schedule(
              [exec_info = std::move(exec_info)]() mutable {});
          return outputs;
        });
  });
}

absl::Status IfrtServingExecutable::AsyncLoadIfrtArray(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const CachedExecutableBundle& executable_bundle,
    const xla::ifrt::DeviceListRef& devices) {
  if (executable_bundle.xla_input_shapes.size() != inputs.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", executable_bundle.xla_input_shapes.size(),
                     " input shapes, but got ", inputs.size(), " inputs"));
  }
  for (const int i : variable_arg_indices) {
    // Validation for variable inputs is handled upstream in the Execute()
    // method.
    std::string tensor_name = inputs[i].scalar<tsl::tstring>()();
    // TODO(b/339521818): Add test cases for OpSharding on variables.
    VariableDeviceShardingConfig sharding_config{
        .hlo_sharding = executable_bundle.arg_hlo_shardings[i],
    };
    for (xla::ifrt::Device* device : devices->devices()) {
      sharding_config.device_ids.push_back(device->Id().value());
    }
    xla::ifrt::LayoutRef layout_ref = executable_bundle.xla_input_layouts[i];
    std::shared_ptr<const xla::Shape> shape_on_device =
        executable_bundle.xla_input_shapes[i];

    TF_RETURN_IF_ERROR(
        ifrt_serving::AsyncLoadRestoredTensorAsIfrtLoadedVariable(
            tensor_name, ifrt_client_, thread_pool_,
            ifrt_restore_tensor_registry_, ifrt_loaded_variable_registry_,
            checkpoint_loader_queue_, sharding_config, std::move(layout_ref),
            std::move(shape_on_device), devices));
  }
  return absl::OkStatus();
}
}  // namespace ifrt_serving
}  // namespace tensorflow
