/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/execute.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/common_runtime/arg_ret_placement.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/small_constants_optimizer.h"
#include "tensorflow/core/common_runtime/eager/summary_optimizer.h"
#include "tensorflow/core/common_runtime/int32_fulltype.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"

// clang-format on

#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/compiler/jit/defs.h"
#include "xla/tsl/util/env_var.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/statusor.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_copy_node.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/util.h"

#ifdef INTEL_MKL
#include "tensorflow/core/graph/mkl_graph_util.h"
#endif

namespace tensorflow {

namespace {

constexpr char kEnabled[] = "enabled";
constexpr char kDisabled[] = "disabled";

auto* function_compile_counter =
    monitoring::Counter<2>::New("/tensorflow/core/tf_function_compile",
                                "The number of times that TF function is "
                                "called for different compilation options.",
                                "device", "compilation_option");
auto* top_level_jit_compilation_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/tf_top_level_jit_compilation",
    "The number of times a top-level JIT-compiled function is called.",
    "device");

bool SendAsProtosWhenPossible() {
  static bool send_as_protos_when_possible = []() {
    bool result;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_SEND_AS_PROTOS_WHEN_POSSIBLE",
                                        false, &result));
    return result;
  }();
  return send_as_protos_when_possible;
}

const string& DeviceNameOrUnspecified(Device* device) {
  static string* unspecified_string = new string("<unspecified>");
  return (device == nullptr) ? *unspecified_string : device->name();
}

// Returns whether a kernel should be cached.
bool KernelCacheEnabled(const OpDef& op_def) {
  if (data::DatasetOpKernel::IsDatasetOp(op_def)) {
    return false;
  }
  // TODO(b/162540360): Revisit a way to mark kernels as uncachable once we have
  // 5+ kernels to exclude.
  return true;
}

// This function expects *handle to point to an existing tensor handle that is
// currently on "handle_device", but where the operation expects that input to
// reside on "expected_input_device".  The function will arrange for this
// transfer to happen and will return OK on success and will storage a new
// handle to the equivalent tensor on the correct device in "*result".  Or if an
// error is encountered, it will return a non-OK status and set "*result" to
// nullptr.
//
// `op_device` is passed in explicitly because `op->device()` might be
// unset and we might have selected some specific device to run this op on.
Status CopyInputToExpectedDevice(EagerContext* ctx, EagerOperation* op,
                                 Device* op_device,
                                 TensorHandle* handle,  // op->Inputs()[i]
                                 int i, Device* handle_device,
                                 Device* expected_input_device,
                                 TensorHandle** result) {
  VLOG(6) << "Expected input device: " << expected_input_device->name()
          << "; handle_device: " << handle_device->name();
  // Should only be called when these don't match
  DCHECK(expected_input_device != handle_device);
  *result = nullptr;
  const string& op_device_name = DeviceNameOrUnspecified(op_device);

  switch (ctx->GetDevicePlacementPolicy()) {
    case DEVICE_PLACEMENT_SILENT_FOR_INT32:
      // TODO(xpan): See if we could bubble python related error up
      // to python level.
      if (handle->dtype == DT_INT32) {
        // Note: enabling silent copies of int32 tensors to match behavior
        // of graph mode.
        break;
      }
      VLOG(6) << "DevicePlacementPolicy: DEVICE_PLACEMENT_SILENT_FOR_INT32 but "
                 "input type is not INT32.";
      TF_FALLTHROUGH_INTENDED;
    case DEVICE_PLACEMENT_EXPLICIT:
      // tf.identity is allowed to copy, as indicated in the error message
      // below.
      if (op->Name() == "Identity" ||
          op->Name() == "IdentityN"
          // Constants start on CPU:0 and are copied via EagerConst to the
          // current device.
          || op->Name() == "_EagerConst") {
        break;
      }
      return errors::InvalidArgument(
          "Tensors on conflicting devices:"
          " cannot compute ",
          op->Name(), " as input #", i, " was expected to be on ",
          expected_input_device->name(), " but is actually on ",
          handle_device->name(), " (operation running on ", op_device_name, ")",
          " Tensors can be copied explicitly using:"
          " `with tf.device(device_name): x = tf.identity(x)`"
          " or transparently copied by using"
          " tf.config.experimental.set_device_policy('silent')."
          " Copying tensors between devices may slow down your model");
    case DEVICE_PLACEMENT_WARN:
      LOG(WARNING) << "before computing " << op->Name() << " input #" << i
                   << " was expected to be on " << expected_input_device->name()
                   << " but is actually on " << handle_device->name()
                   << " (operation running on " << op_device_name
                   << "). This triggers a copy which can be a performance "
                      "bottleneck.";
      break;
    case DEVICE_PLACEMENT_SILENT:  // Do nothing.
      break;
  }
  // We are only here if the policy is warn or silent copies, so we should
  // trigger a copy.
  TensorHandle* result_handle = nullptr;
  tsl::profiler::TraceMe activity(
      [&] {
        return absl::StrCat("_Send input ", i, " from ", handle_device->name(),
                            " to ", expected_input_device->name());
      },
      tsl::profiler::TraceMeLevel::kInfo);
  Status status =
      EagerCopyToDevice(handle, ctx, &op->Executor(), expected_input_device,
                        /* mirror= */ true, &result_handle);
  activity.Stop();
  if (!status.ok()) {
    return Status(
        status.code(),
        absl::StrCat("Failed copying input tensor from ", handle_device->name(),
                     " to ", expected_input_device->name(), " in order to run ",
                     op->Name(), ": ", status.message()));
  }

  *result = result_handle;

  return absl::OkStatus();
}

// `op_device_name` the name of the device on which the op will run, if any.
// For functions running using function library runtime, the device can be
// unspecified.
Status ValidateInputTypeAndPlacement(
    EagerContext* ctx, EagerOperation* op,
    const core::RefCountPtr<KernelAndDevice>& kernel) {
  tsl::profiler::TraceMe activity("ValidateInputTypeAndPlacement",
                                  tsl::profiler::TraceMeLevel::kInfo);
  const int n_inputs = op->Inputs().size();
  if (kernel->num_inputs() != n_inputs) {
    return errors::InvalidArgument("expected ", kernel->num_inputs(),
                                   " inputs, got ", n_inputs);
  }
  const bool is_function = kernel->IsFunction();
  if (n_inputs > 0) {
    const DataType* input_types = &kernel->input_dtypes()[0];
    const absl::InlinedVector<TensorHandle*, 4>* handles;
    TF_RETURN_IF_ERROR(op->TensorHandleInputs(&handles));
    for (int i = 0; i < n_inputs; ++i) {
      TensorHandle* handle = (*handles)[i];
      Device* expected_device = kernel->InputDevice(i);
      if (!kernel->IsFunction() && handle->Type() == TensorHandle::PACKED) {
        // Extract a handle on the op device from a packed input.
        // This happens when a function is marked for XLA compilation.
        // MaybePackInputTensor guarantees that a primitive op has no packed
        // input at this point.
        for (int j = 0; j < handle->NumPackedHandles(); ++j) {
          TensorHandle* h = nullptr;
          TF_RETURN_IF_ERROR(handle->ExtractPackedHandle(j, &h));
          if ((h->op_device() != nullptr) &&
              (h->op_device()->name() == op->DeviceName())) {
            op->UpdateInput(i, h);
            handle = h;
            break;
          }
        }
      }
      Device* handle_device = handle->DeviceOrHostCPU(*ctx);
      const bool maybe_copy =
          !is_function || handle->Type() != TensorHandle::REMOTE;
      VLOG(6) << "!is_function: " << !is_function;
      VLOG(6) << "handle->Type(): " << handle->Type();
      // If the input is already on the right device, then nothing to do.
      if (expected_device != handle_device && maybe_copy) {
        TF_RETURN_IF_ERROR(CopyInputToExpectedDevice(ctx, op, kernel->device(),
                                                     handle, i, handle_device,
                                                     expected_device, &handle));
        op->UpdateInput(i, handle);
        // Unref handle since it has a ref as an input now
        handle->Unref();
      }
      if (handle->dtype != input_types[i]) {
        return errors::InvalidArgument(
            "cannot compute ", op->Name(), " as input #", i, "(zero-based)",
            " was expected to be a ", DataTypeString(input_types[i]),
            " tensor but is a ", DataTypeString(handle->dtype), " tensor");
      }
    }
  }
  return absl::OkStatus();
}

Status GetOutputDTypes(EagerOperation* op, DataTypeVector* output_dtypes) {
  const auto& node_def = op->MutableAttrs()->BuildNodeDef();
  const OpDef* op_def = nullptr;

  const FunctionDef* function_def = op->GetFunctionDef();
  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(op->Name().c_str(), &op_def));
  }

  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, *op_def, output_dtypes));

  return absl::OkStatus();
}

const KernelDef* GetKernelDef(const EagerOperation& op, const NodeDef* node_def,
                              const Device* op_device) {
  if (node_def == nullptr || op_device == nullptr) return nullptr;
  const KernelDef* kernel_def = nullptr;
  Status s = FindKernelDef(DeviceType(op_device->device_type()), *node_def,
                           &kernel_def,
                           /*kernel_class_name=*/nullptr);
  if (!s.ok()) return nullptr;
  return kernel_def;
}

bool IsHostMemoryArg(const EagerOperation& op, const NodeDef* node_def,
                     const Device* op_device, const KernelDef* kernel_def,
                     const int port_id) {
  if (op.is_function()) return false;
  if (node_def == nullptr) return false;
  if (kernel_def == nullptr || op_device == nullptr) return false;
  const auto& host_memory_args = kernel_def->host_memory_arg();
  const OpDef& op_def = OpRegistry::Global()->LookUp(op.Name())->op_def;
  const int arg_id = OpPortIdToArgId(*node_def, op_def.input_arg(), port_id);
  // Fail if argument ID not found.
  if (arg_id < 0) {
    return false;
  }
  return std::find(host_memory_args.begin(), host_memory_args.end(),
                   op_def.input_arg(arg_id).name()) != host_memory_args.end();
}

Status GetDeviceForInput(const EagerOperation& op, const EagerContext& ctx,
                         const bool is_host_memory_arg,
                         TensorHandle* tensor_handle, Device** result) {
  Device* cpu_device = ctx.HostCPU();
  string device_name;
  if (tensor_handle->Type() != TensorHandle::LOCAL) {
    Device* device = tensor_handle->device();
    device_name = device != nullptr ? device->name() : cpu_device->name();
    *result = (device == nullptr ? cpu_device : device);
  } else if (tensor_handle->dtype == DT_RESOURCE) {
    // Use the resource's actual device because it is the device that will
    // influence partitioning the multi-device function.
    const Tensor* tensor;
    // TODO(fishx): Avoid blocking here.
    TF_RETURN_IF_ERROR(tensor_handle->Tensor(&tensor));
    if (tensor->NumElements() == 0) {
      return errors::InvalidArgument("Empty resource handle");
    }
    const ResourceHandle& handle = tensor->flat<ResourceHandle>()(0);
    device_name = handle.device();

    Device* input_device;
    TF_RETURN_IF_ERROR(
        ctx.FindDeviceFromName(device_name.c_str(), &input_device));
    *result = input_device;
  } else {
    Device* device = tensor_handle->device();
    const bool is_tpu = device != nullptr && device->device_type() == "TPU";
    // int32 return values can be placed on TPUs.
    // int32 retrun values can be placed on device for eager operations.
    FullTypeDef ft = tensor_handle->FullType();
    const bool use_host_memory =
        is_tpu || (!op.is_function() && device != cpu_device &&
                   !is_host_memory_arg)
            ? MTypeFromDTypeIntsOnDevice(tensor_handle->dtype)
            : MTypeFromDType(tensor_handle->dtype);
    if (use_host_memory) {
      Int32FulltypePass int32_ft("GetDeviceForInput");
      TF_RETURN_IF_ERROR(int32_ft.Int32FullTypeForTensor(
          tensor_handle->dtype, &ft, /*set_only_int32=*/false));
      VLOG(2)
          << "Full type information with TFT_SHAPE_TENSOR for int32 for eager '"
          << tensor_handle->DebugString();
    }
    TF_RETURN_IF_ERROR(
        tensorflow::full_type::CheckMemoryType(use_host_memory, ft));
    if (use_host_memory) {
      *result = cpu_device;
    } else {
      // Eager ops executing as functions should have their preferred inputs set
      // to the op's device. This allows us to avoid expensive D2H copies if a
      // mirror of the tensor already exists on the op's device.
      if (!op.is_function() && device != cpu_device && !is_host_memory_arg) {
        device = std::get<Device*>(op.Device());
      }
      *result = (device == nullptr ? cpu_device : device);
    }
  }
  return absl::OkStatus();
}

// Appends a TensorShape object to Fprint128 hash.
// For best performance, we would like to avoid dynamic memory allocation in
// this function.
// If "shape" has unknown rank, we attach "?" to hashed content; otherwise we
// attach every dim size to hashed content.
void AppendTensorShapeToFingerprint(const PartialTensorShape& shape,
                                    Fprint128* fingerprint) {
  if (shape.unknown_rank()) {
    char c = '?';
    *fingerprint = tsl::FingerprintCat128(*fingerprint, c);
  } else {
    for (int i = 0; i < shape.dims(); i++) {
      int64_t dim = shape.dim_size(i);
      *fingerprint = tsl::FingerprintCat128(*fingerprint, dim);
    }
  }
}

Status GetFuncAttr(const EagerOperation* op, const EagerContext& ctx,
                   const char* attr_name, bool* value) {
  Status status = op->Attrs().Get(attr_name, value);
  if (status.ok()) {
    VLOG(2) << "Caller explicitly specifies " << attr_name
            << (value ? "=true " : "=false, ") << op->DebugString();
    return absl::OkStatus();
  }

  const FunctionDef* function_def = op->GetFunctionDef();
  if (function_def == nullptr) {
    return errors::NotFound("Failed to find function '", op->Name(), "'");
  }

  status = GetNodeAttr(AttrSlice(&function_def->attr()), attr_name, value);
  if (status.ok()) {
    VLOG(2) << "Function definition explicitly specifies " << attr_name
            << (value ? "=true" : "=false");
    return absl::OkStatus();
  }
  return status;
}

// Checks if `op` is a function and contains TPU replication ops.  If `op` does,
// then `has_tpu_replication` is set to true.  Other `has_tpu_replication` is
// set to false.
Status HasTPUReplication(const EagerOperation& op, const EagerContext& ctx,
                         bool* has_tpu_replication) {
  *has_tpu_replication = false;
  if (!op.is_function()) {
    return absl::OkStatus();
  }

  const FunctionDef* function_def = op.GetFunctionDef();
  if (function_def == nullptr) {
    return errors::NotFound("Failed to find function '", op.Name(), "'");
  }
  for (const NodeDef& node : function_def->node_def()) {
    if (node.op() == "TPUReplicateMetadata") {
      *has_tpu_replication = true;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

Status MustCompileWithXLA(const EagerOperation* op, const EagerContext& ctx,
                          bool* compile_with_xla) {
#if defined(PLUGGABLE_DEVICE_SUPPORTED_MACOS)
  *compile_with_xla = false;
#else
  if (!op->is_function()) {
    *compile_with_xla = false;
    return absl::OkStatus();
  }

  if (op->eager_func_params().has_value() &&
      op->eager_func_params().value().is_component_function) {
    // If the op is a component of a multi-device function, don't compile it
    // with XLA.
    *compile_with_xla = false;
    return absl::OkStatus();
  }

  Status status = GetFuncAttr(op, ctx, kXlaMustCompileAttr, compile_with_xla);
  if (status.ok()) {
    return absl::OkStatus();
  }

  // No explicit requests. Compile for XLA devices by default.
  if (op->GetDeviceParsedName().type == tensorflow::DEVICE_TPU ||
      op->GetDeviceParsedName().type == "XLA_GPU" ||
      op->GetDeviceParsedName().type == "XLA_CPU") {
    VLOG(2) << "Compiling " << op->Name()
            << " with XLA because it is running on an XLA device "
            << op->GetDeviceParsedName().type;
    *compile_with_xla = true;
  } else {
    *compile_with_xla = false;
  }
#endif

  return absl::OkStatus();
}

// Check if `op` has tf.StatefulPartitionedCall op with _XlaMustCompile, sets
// `has_jit_compile` and `device`.
Status HasNestedJitCompile(const EagerOperation& op, const EagerContext& ctx,
                           bool* has_jit_compile, string* device) {
  *has_jit_compile = false;

  const std::string kStatefulPartitionedCallOp = "StatefulPartitionedCall";
  const std::string kXlaMustCompile = "_XlaMustCompile";
  if (!op.is_function()) {
    return absl::OkStatus();
  }

  std::queue<std::string> function_names;
  function_names.push(op.Name());

  const FunctionLibraryDefinition* func_lib_def = op.FuncLibDef();

  while (!function_names.empty()) {
    const string& function_name = function_names.front();

    const FunctionDef* function_def = func_lib_def->Find(function_name);
    if (function_def == nullptr) {
      return errors::NotFound("Failed to find function '", function_name, "'");
    }
    function_names.pop();
    for (const NodeDef& node : function_def->node_def()) {
      if (node.op() == kStatefulPartitionedCallOp) {
        auto attr = node.attr().find(kXlaMustCompile);
        if (attr != node.attr().end() && attr->second.b() == true) {
          *has_jit_compile = true;
          auto device_attr = node.attr().find("device");
          if (device_attr != node.attr().end()) {
            *device = device_attr->second.s();
          }
          return absl::OkStatus();
        } else {
          auto attr = node.attr().find("f");
          if (attr != node.attr().end() &&
              !attr->second.func().name().empty()) {
            function_names.push(attr->second.func().name());
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

string CanonicalizeDeviceType(std::string_view device_type) {
  string canonical_device_type = "Unknown";
  if (device_type == "XLA_CPU" || device_type == tensorflow::DEVICE_CPU) {
    canonical_device_type = tensorflow::DEVICE_CPU;
  }
  if (device_type == "XLA_GPU" || device_type == tensorflow::DEVICE_GPU) {
    canonical_device_type = tensorflow::DEVICE_GPU;
  }
  if (device_type == tensorflow::DEVICE_TPU) {
    canonical_device_type = tensorflow::DEVICE_TPU;
  }
  return canonical_device_type;
}

Status UpdateCompileCounter(const EagerOperation* op, const EagerContext& ctx,
                            bool compile_with_xla, bool has_tpu_replication) {
  if (has_tpu_replication) {
    function_compile_counter->GetCell(tensorflow::DEVICE_TPU, kEnabled)
        ->IncrementBy(1);
    return absl::OkStatus();
  }

  string device_type = CanonicalizeDeviceType(op->GetDeviceParsedName().type);
  string compilation_option = kDisabled;
  if (!compile_with_xla) {
    bool nested_jit_compile;
    string device;
    TF_RETURN_IF_ERROR(
        HasNestedJitCompile(*op, ctx, &nested_jit_compile, &device));
    if (nested_jit_compile) {
      if (!device.empty()) {
        tsl::DeviceNameUtils::ParsedName device_parsed_name;
        if (!DeviceNameUtils::ParseFullName(device, &device_parsed_name)) {
          return errors::InvalidArgument("Malformed device specification: '",
                                         device);
        }
        VLOG(1) << "Compilation Device Type: " << device_parsed_name.type;

        function_compile_counter
            ->GetCell(CanonicalizeDeviceType(device_parsed_name.type), kEnabled)
            ->IncrementBy(1);
        return absl::OkStatus();
      } else {
        compilation_option = kEnabled;
      }
    }
  } else {
    // Top-level JIT compilation
    top_level_jit_compilation_counter->GetCell(device_type)->IncrementBy(1);
  }

  if (device_type == tensorflow::DEVICE_TPU || compile_with_xla) {
    compilation_option = kEnabled;
  }

  VLOG(1) << "Compilation Device Type: " << device_type;

  function_compile_counter->GetCell(device_type, compilation_option)
      ->IncrementBy(1);
  return absl::OkStatus();
}

Status VerifyWrappableInCallOp(const OpDef& opdef, EagerOperation* op) {
  absl::flat_hash_set<string> opdef_attrs;
  for (const auto& attr : opdef.attr()) {
    opdef_attrs.insert(attr.name());
  }
  const auto& node_def = op->MutableAttrs()->BuildNodeDef();
  for (const auto& attr : node_def.attr()) {
    if (opdef_attrs.find(attr.first) == opdef_attrs.end()) {
      return errors::Unimplemented("EagerOperation: ", op->Name(),
                                   " has a private attr '", attr.first, "'.");
    }
  }
  return absl::OkStatus();
}

using ProtoArgListType = protobuf::RepeatedPtrField<OpDef_ArgDef>;

string EscapeOrigName(const string& orig_name) {
  // Replace _ with __ in the original name to avoid name conflicts.
  return absl::StrReplaceAll(orig_name, {{"_", "__"}});
}

// Variadic args are flattened during wrapping. This utility returns the name
// of a flattened arg/attr.
string GetFlatName(const string orig_name, int index) {
  return absl::StrCat(EscapeOrigName(orig_name), "_", index);
}

// Builds the name of the wrapping FunctionDef for an eager op.
//
// For ops without variadic inputs/outputs, the name is simply __wrapped_OpType.
//
// For ops with variadic inputs/outputs, the arity of each variadic attr is
// encoded in the name. For example:
//
// IdentityN[T:[DT_FLOAT, DT_INT64]] -> __wrapped__IdentityN_T_2
// Concat[N:2, T:DT_FLOAT] -> __wrapped__Concat_N_2
Status BuildWrappedOpName(EagerOperation* op, const OpDef& opdef,
                          const AbstractOpAttrs* op_attrs, string* name) {
  string fname = absl::StrCat("__wrapped__", EscapeOrigName(op->Name()));
  // For every variadic arg in `args`, populates `attr_to_len` with
  // (attr_name, len(arg)).
  auto FillAttrToLen = [op_attrs, op](
                           const ProtoArgListType& args,
                           absl::btree_map<string, int>* attr_to_len) {
    for (const auto& arg : args) {
      if (!arg.type_list_attr().empty()) {
        gtl::InlinedVector<DataType, 4> type_list;
        TF_RETURN_IF_ERROR(
            op_attrs->GetTypeList(arg.type_list_attr(), &type_list));
        (*attr_to_len)[arg.type_list_attr()] = type_list.size();
      } else if (!arg.number_attr().empty()) {
        int64_t number_attr;
        if (!op_attrs->GetInt(arg.number_attr(), &number_attr)) {
          return errors::Internal("Unable to read attr ", arg.number_attr(),
                                  " for op ", op->Name());
        }
        (*attr_to_len)[arg.number_attr()] = number_attr;
      }
    }
    return absl::OkStatus();
  };
  absl::btree_map<string, int> attr_to_len;
  TF_RETURN_IF_ERROR(FillAttrToLen(opdef.input_arg(), &attr_to_len));
  TF_RETURN_IF_ERROR(FillAttrToLen(opdef.output_arg(), &attr_to_len));
  for (auto& name_len : attr_to_len) {
    absl::StrAppend(&fname, "_", name_len.first, "_", name_len.second);
  }
  // The NodeDef in the FunctionDef gets placed on `op-DeviceName()` to ensure
  // placement consistency with eager mode.
  // TODO(b/200153278): Ideally we would just forward the call op's device at
  // runtime but currently there is no way to do it so we incur the cost of
  // creating extra FunctionDefs.
  absl::StrAppend(&fname, "_device_", op->DeviceName());
  *name = fname;
  return absl::OkStatus();
}

// Validates the node def. This is required when running in eager op as function
// mode because this code path does not go through the _apply_op_helper's
// validation (which is reached when executing in graph mode)
// or the eager execution's validation (which is reached via the CreateOpKernel
// call).
Status ValidateOp(EagerOperation* op) {
  const NodeDef& node_def = op->MutableAttrs()->BuildNodeDef();
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));
  return ValidateNodeDef(node_def, *op_def);
}

// Builds the signature of the wrapping FunctionDef for an eager op.
//
// For ops without variadic inputs/outputs, the signature is the same as the
// OpDef of the original op.
//
// Variadic inputs/outputs get flattened since we do not support executing
// functions with variadic signatures.
//
// TODO(srbs): These examples should be tests.
//
// Examples:
//
// Mixed type list:
//
// op {
//   name: "IdentityN"
//   input_arg {
//     name: "input"
//     type_list_attr: "T"
//   }
//   output_arg {
//     name: "output"
//     type_list_attr: "T"
//   }
//   attr {
//     name: "T"
//     type: "list(type)"
//     has_minimum: true
//     minimum: 1
//   }
// }
//
// With two inputs T=[DT_FLOAT, DT_INT64] would convert to
//
// op {
//   name: "__wrapped__IdentityN_T_2"
//   input_arg {
//     name: "input_0"
//     type_attr: "T_0"
//   }
//   input_arg {
//     name: "input_1"
//     type_attr: "T_1"
//   }
//   output_arg {
//     name: "output_0"
//     type_attr: "T_0"
//   }
//   output_arg {
//     name: "output_1"
//     type_attr: "T_1"
//   }
//   attr {
//     name: "T_0"
//     type: "type"
//   }
//   attr {
//     name: "T_1"
//     type: "type"
//   }
//   attr {
//     name: "T"
//     type: "list(type)"
//     has_minimum: true
//     minimum: 1
//   }
// }
//
// Note that the list(type) attr is preserved so that it can get copied to the
// inner op via a placeholder. This allows additional verification.
//
// Single type list:
//
// op {
//   name: "ConcatV2"
//   input_arg {
//     name: "values"
//     type_attr: "T"
//     number_attr: "N"
//   }
//   attr {
//     name: "N"
//     type: "int"
//     has_minimum: true
//     minimum: 2
//   }
//   attr {
//     name: "T"
//     type: "type"
//   }
//   [axis, output, Tidx are simply copied]
// }
//
// With two inputs N=2 would convert to:
//
// op {
//   name: "__wrapped__ConcatV2_N_2"
//   input_arg {
//     name: "values_0"
//     type_attr: "T"
//   }
//   input_arg {
//     name: "values_1"
//     type_attr: "T"
//   }
//   attr {
//     name: "N"
//     type: "int"
//     has_minimum: true
//     minimum: 2
//   }
//   attr {
//     name: "T"
//     type: "type"
//   }
//   [axis, output, Tidx are simply copied]
// }
//
// Note that the N attr is preserved so that it can get copied to the
// inner op via a placeholder. This allows additional verification.
Status BuildWrappedOpSignature(EagerOperation* op, const OpDef& opdef,
                               const string& fname, OpDef& signature) {
  signature = opdef;
  signature.clear_input_arg();
  signature.clear_output_arg();
  signature.set_name(fname);
  auto op_attrs = op->GetOpAttrs();
  auto FillSignatureArgs = [op_attrs, op](
                               const ProtoArgListType& opdef_args,
                               ProtoArgListType* sig_args,
                               absl::flat_hash_set<string>& new_attrs) {
    for (const auto& arg : opdef_args) {
      if (!arg.type_list_attr().empty()) {
        gtl::InlinedVector<DataType, 4> type_list;
        TF_RETURN_IF_ERROR(
            op_attrs->GetTypeList(arg.type_list_attr(), &type_list));
        for (size_t i = 0; i < type_list.size(); i++) {
          auto arg_def = sig_args->Add();
          arg_def->set_name(GetFlatName(arg.name(), i));
          auto attr_name = GetFlatName(arg.type_list_attr(), i);
          new_attrs.insert(attr_name);
          arg_def->set_type_attr(std::move(attr_name));
        }
      } else if (!arg.number_attr().empty()) {
        int64_t number_attr;
        if (!op_attrs->GetInt(arg.number_attr(), &number_attr)) {
          return errors::Internal("Unable to read attr ", arg.number_attr(),
                                  " for op ", op->Name());
        }
        for (int64_t i = 0; i < number_attr; i++) {
          auto arg_def = sig_args->Add();
          arg_def->set_name(GetFlatName(arg.name(), i));
          if (!arg.type_attr().empty()) {
            arg_def->set_type_attr(arg.type_attr());
          } else {
            arg_def->set_type(arg.type());
          }
        }
      } else {
        auto arg_def = sig_args->Add();
        *arg_def = arg;
        arg_def->set_name(EscapeOrigName(arg.name()));
        if (!arg.type_attr().empty()) {
          // Don't escape: type attrs are still referenced by the original name.
          arg_def->set_type_attr(arg.type_attr());
        }
      }
    }
    return absl::OkStatus();
  };
  absl::flat_hash_set<string> new_attrs;
  TF_RETURN_IF_ERROR(FillSignatureArgs(
      opdef.input_arg(), signature.mutable_input_arg(), new_attrs));
  TF_RETURN_IF_ERROR(FillSignatureArgs(
      opdef.output_arg(), signature.mutable_output_arg(), new_attrs));
  for (auto& attr_name : new_attrs) {
    auto attr_def = signature.mutable_attr()->Add();
    attr_def->set_name(attr_name);
    attr_def->set_type("type");
  }
  return absl::OkStatus();
}

// For mixed type inputs "list(type)" we create new attributes in the signature
// for each element tensor (See examples in BuildWrappedOpSignature). Here
// we construct the values for those attributes and set them on the wrapped op.
Status AddMixedTypeListAttrs(EagerOperation* wrapped_op,
                             const AbstractOpAttrs* op_attrs,
                             const OpDef& opdef) {
  auto FillAttrsToAdd =
      [op_attrs](const ProtoArgListType& opdef_args,
                 absl::flat_hash_map<string, DataType>* attrs_to_add) {
        for (const auto& arg : opdef_args) {
          if (!arg.type_list_attr().empty()) {
            gtl::InlinedVector<DataType, 4> type_list;
            TF_RETURN_IF_ERROR(
                op_attrs->GetTypeList(arg.type_list_attr(), &type_list));
            for (size_t i = 0; i < type_list.size(); i++) {
              auto attr_name = GetFlatName(arg.type_list_attr(), i);
              (*attrs_to_add)[attr_name] = type_list[i];
            }
          }
        }
        return absl::OkStatus();
      };
  absl::flat_hash_map<string, DataType> attrs_to_add;
  TF_RETURN_IF_ERROR(FillAttrsToAdd(opdef.input_arg(), &attrs_to_add));
  TF_RETURN_IF_ERROR(FillAttrsToAdd(opdef.output_arg(), &attrs_to_add));
  for (auto& name_type : attrs_to_add) {
    TF_RETURN_IF_ERROR(
        wrapped_op->SetAttrType(name_type.first.data(), name_type.second));
  }
  // TODO(srbs): Rename all original attributes using EscapeOrigName.
  return absl::OkStatus();
}

// Maps the op's outputs to the function outputs. Mainly useful for variadic
// outputs which need to be flattened.
Status PopulateRetMap(FunctionDef* fdef, const AbstractOpAttrs* op_attrs,
                      const EagerOperation* op, const OpDef& opdef,
                      const OpDef& signature, const string& node_name) {
  int next_sig_output = 0;
  for (size_t i = 0; i < opdef.output_arg_size(); i++) {
    const auto& output_arg = opdef.output_arg(i);
    if (!output_arg.type_list_attr().empty()) {
      gtl::InlinedVector<DataType, 4> type_list;
      TF_RETURN_IF_ERROR(
          op_attrs->GetTypeList(output_arg.type_list_attr(), &type_list));
      for (int j = 0; j < type_list.size(); j++) {
        (*fdef->mutable_ret())[signature.output_arg(next_sig_output++).name()] =
            absl::StrCat(node_name, ":", output_arg.name(), ":", j);
      }
    } else if (!output_arg.number_attr().empty()) {
      int64_t number_attr;
      if (!op_attrs->GetInt(output_arg.number_attr(), &number_attr)) {
        return errors::Internal("Unable to read attr ",
                                output_arg.number_attr(), " for op ",
                                op->Name());
      }
      for (int j = 0; j < number_attr; j++) {
        (*fdef->mutable_ret())[signature.output_arg(next_sig_output++).name()] =
            absl::StrCat(node_name, ":", output_arg.name(), ":", j);
      }
    } else {
      (*fdef->mutable_ret())[signature.output_arg(next_sig_output++).name()] =
          absl::StrCat(node_name, ":", output_arg.name(), ":0");
    }
  }
  return absl::OkStatus();
}

#ifdef INTEL_MKL
inline void GetMKLNodeDef(NodeDef* ndef) {
  // All MKL eager ops have `_kernel` private attribute that needs to be set
  // to a fixed label.
  AttrValue attr_kernel;
  attr_kernel.set_s(mkl_op_registry::kMklNameChangeOpLabel);
  (*ndef->mutable_attr()).insert({"_kernel", attr_kernel});
}
#endif  // INTEL_MKL

Status WrapInCallOp(EagerOperation* op, EagerOperation** wrapped_op) {
  DCHECK(!op->is_function());
  const OpDef& opdef = OpRegistry::Global()->LookUp(op->Name())->op_def;
  // Raise an error for ops which don't support wrapping yet. This includes
  // ops with list inputs/outputs and ops with private attrs.
  // TODO(srbs): Support list inputs/outputs.
  TF_RETURN_IF_ERROR(VerifyWrappableInCallOp(opdef, op));

  // Build a FunctionDef containing op as a node and register with context.
  // TODO(srbs): Here we are unable to distinguish between a FunctionDef for
  // a wrapped eager op and an existing user defined function registered with
  // the context e.g. with something like
  // @tf.function
  // def __wrapped__Add(x, y):
  //   ...
  // This can be avoided by introducing a dict in EagerContext that stores a
  // mapping from the eager op's name to its unique FunctionDef name.
  auto op_attrs = op->GetOpAttrs();
  string fname;
  TF_RETURN_IF_ERROR(BuildWrappedOpName(op, opdef, op_attrs, &fname));
  if (!op->EagerContext().GetFunctionDef(fname)) {
    FunctionDef fdef;
    // Set signature.
    TF_RETURN_IF_ERROR(
        BuildWrappedOpSignature(op, opdef, fname, *fdef.mutable_signature()));
    // Add node.
    NodeDef* ndef = fdef.add_node_def();
    ndef->set_op(op->Name());
    ndef->set_name(op->Name());  // This could be anything.
    const auto& signature = fdef.signature();
    for (size_t i = 0; i < signature.input_arg_size(); i++) {
      ndef->add_input(absl::StrCat(fdef.signature().input_arg(i).name(), ":0"));
    }
    // TODO(srbs): Private attrs on the op are dropped here and applied to
    // the call op instead. If this causes problems we might have to copy those
    // attrs to this ndef. That would require updating fname to contain a hash
    // of such attributes.
    for (const auto& attr : opdef.attr()) {
      (*ndef->mutable_attr())[attr.name()].set_placeholder(attr.name());
    }
    // Set the device of this node to be the exact same one that eager mode
    // would have used.
    // TODO(b/200153278): Ideally we would just forward the call op's device at
    // runtime but currently there is no way to do it.
    ndef->set_device(op->DeviceName());

#ifdef INTEL_MKL
    if (IsMKLEnabled() &&
        absl::StartsWith(op->Name(), mkl_op_registry::kMklOpPrefix)) {
      GetMKLNodeDef(ndef);
    }
#endif  // INTEL_MKL

    // Set `ret` map.
    TF_RETURN_IF_ERROR(
        PopulateRetMap(&fdef, op_attrs, op, opdef, signature, ndef->name()));
    VLOG(1) << fdef.DebugString();
    TF_RETURN_IF_ERROR(op->EagerContext().AddFunctionDef(std::move(fdef)));
  }
  // Build the call op.
  auto& ctx = op->EagerContext();
  AbstractOperationPtr call_op(ctx.CreateOperation());
  TF_RETURN_IF_ERROR(call_op->Reset(fname.c_str(), op->DeviceName().c_str()));
  for (auto t : op->Inputs()) {
    TF_RETURN_IF_ERROR(call_op->AddInput(t));
  }
  *wrapped_op = down_cast<EagerOperation*>(call_op.release());
  // Attributes on the elementary eager operation are applied to the call op and
  // to the NodeDef inside the FunctionDef. This allows us to have a single
  // FunctionDef for different attribute values. When the function is
  // instantiated, these attributes get forwarded to the NodeDef. This is done
  // by setting the AttrValue.placeholder field for the NodeDef attrs.
  (*wrapped_op)->AddAttrs(op_attrs);
  return AddMixedTypeListAttrs(*wrapped_op, op_attrs, opdef);
}

// Necessary condition to place int args/retvals on device but not sufficient.
// For eager operations return values can be placed on the device for use
// by subsequent eager ops. E.g.
// with tf.device("/GPU:0"):
//   x = tf.random_uniform(shape=(2, 2), maxval=5, dtype=tf.int32)
//   y = tf.random_uniform(shape=(2, 2), maxval=5, dtype=tf.int32)
//   z = tf.bitwise.bitwise_and(x, y)
// In the above example `z` can use the outputs of `x` and `y` without needing
// an H2D copy if x and y are left on-device.
bool IntArgsAndRetvalsOnDevice(EagerOperation* op,
                               const KernelDef* kernel_def) {
  // We choose to leave `EagerConsts`
  // on HOST to avoid `shape` and other arguments that are traditionally pinned
  // to HostMemory from being placed on-device and then being copied to host via
  // an expensive D2H transfer.
  if (op->Name() == "_EagerConst") return false;

  // Check if any of the Op's output_arg(s) are pinned to Host.
  if (kernel_def == nullptr) return false;
  const OpDef& op_def = OpRegistry::Global()->LookUp(op->Name())->op_def;
  for (const string& host_memory_arg : kernel_def->host_memory_arg()) {
    for (const auto& output_arg : op_def.output_arg()) {
      if (output_arg.name() == host_memory_arg) {
        return false;
      }
    }
  }

  return true;
}

using BoolTensorInputs = std::vector<std::pair<std::string, bool>>;

// Identifies boolean tensor inputs from the EagerOperation and returns them. If
// delete_inputs is set to true then it will also delete them from the
// function's input signature. Currently this is only useful to invoke when
// small_constants_optimizer is enabled because the runtime will have equivalent
// FunctionDefs of the original tf.function without the boolean tensor input.
absl::StatusOr<BoolTensorInputs> GetBoolInputs(EagerOperation* op,
                                               bool delete_inputs) {
  BoolTensorInputs result;
  if (!op->is_function()) return result;
  // Extract tensor inputs.
  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  if (!op->TensorHandleInputs(&inputs).ok()) return result;
  // Extract the FunctionDef.
  const FunctionDef* fdef = op->EagerContext().GetFunctionDef(op->Name());
  if (fdef == nullptr) return result;
  // Ensure the number of inputs matches the specification in the FunctionDef.
  if (fdef->signature().input_arg_size() != inputs->size()) return result;

  // Remove all boolean inputs.
  absl::InlinedVector<TensorHandle*, 4> stripped_inputs;
  for (int32_t i = 0; i < fdef->signature().input_arg_size(); ++i) {
    const auto& input_arg = fdef->signature().input_arg(i);
    // Identify non-boolean inputs to this EagerOperation.
    if (input_arg.type() != DT_BOOL) {
      stripped_inputs.push_back(inputs->at(i));
      continue;
    }
    // Identify boolean inputs to this EagerOperation that are on host.
    const TensorHandle* handle = inputs->at(i);
    Status s;
    const char* input_device = handle->DeviceType(&s);
    if (!s.ok() || !absl::StrContains(input_device, "CPU")) {
      return errors::InvalidArgument(
          "Expecting boolean tensor to be on host when "
          "small_constants_optimizer is enabled.");
    }
    const Tensor* tensor;
    TF_RETURN_IF_ERROR(handle->Tensor(&tensor));
    // small_constant_optimizer does not handle non-scalar boolean inputs.
    if (tensor->NumElements() != 1) {
      stripped_inputs.push_back(inputs->at(i));
      continue;
    }
    const bool input_value = tensor->scalar<bool>()();
    result.emplace_back(input_arg.name(), input_value);
  }

  if (!delete_inputs) return result;
  // If we were able to identify all boolean inputs, update the op's inputs.
  op->Clear();
  for (auto* input : stripped_inputs) {
    TF_RETURN_IF_ERROR(op->AddInput(input));
  }
  return result;
}

// Returns the value of the `op`'s input `arg_name`.
// Returns `std::nullopt` by default in the following cases:
// - `arg_name` is not a boolean tensor.
// - `op` does nat have an input `arg_name`.
// - `arg_name` is not on HOST.
// - any issues with the `FunctionDef` in the `EagerContext`.
std::optional<bool> GetBoolArgumentValue(const EagerOperation& op,
                                         const absl::string_view arg_name) {
  if (!op.is_function()) return std::nullopt;
  // Extract tensor inputs.
  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  if (!op.TensorHandleInputs(&inputs).ok()) return std::nullopt;
  // Extract the FunctionDef.
  const FunctionDef* fdef = op.EagerContext().GetFunctionDef(op.Name());
  if (fdef == nullptr) return std::nullopt;
  // Ensure the number of inputs matches the specification in the FunctionDef.
  if (fdef->signature().input_arg_size() != inputs->size()) return std::nullopt;

  // Identify the value of the boolean input.
  for (int32_t i = 0; i < fdef->signature().input_arg_size(); ++i) {
    const auto& input_arg = fdef->signature().input_arg(i);
    if (input_arg.name() != arg_name) continue;
    if (input_arg.type() != DT_BOOL) return std::nullopt;

    // If the input is not on host returns std::nullopt.
    const TensorHandle* handle = inputs->at(i);
    Status s;
    const char* input_device = handle->DeviceType(&s);
    if (!s.ok() || !absl::StrContains(input_device, "CPU")) return std::nullopt;

    // If there was an error reading the input value returns std::nullopt.
    const Tensor* tensor;
    auto read_tensor_status = handle->Tensor(&tensor);
    if (!read_tensor_status.ok()) return std::nullopt;

    // Return the input value `arg_name` passed to the `op`.
    const bool input_value = tensor->scalar<bool>()();
    return input_value;
  }

  // Could not find `arg_name` return std::nullopt by default.
  return std::nullopt;
}

bool IsSmallConstantOptimizationEnabled(const EagerOperation& op) {
  if (!op.is_function()) return false;
  const FunctionDef* fdef = op.EagerContext().GetFunctionDef(op.Name());
  if (fdef == nullptr) return false;
  return small_constants_optimizer::IsSmallConstantOptimizationEnabled(*fdef);
}

bool IsSummaryOptimizerEnabled(const EagerOperation* op) {
  if (!op->is_function()) return false;
  const FunctionDef* fdef = op->EagerContext().GetFunctionDef(op->Name());
  if (fdef == nullptr) return false;
  const auto include_summary_arg =
      summary_optimizer::GetDisableSummariesInputArg(*fdef);
  if (include_summary_arg.first.empty()) return false;
  const auto arg_value = GetBoolArgumentValue(*op, include_summary_arg.first);
  if (!arg_value.has_value()) return false;
  return arg_value.value() == include_summary_arg.second;
}

absl::StatusOr<Fprint128> GetKernelCacheKey(
    const EagerOperation& op, const Fprint128& op_cache_key,
    const std::vector<Device*>& input_device_ptrs,
    const std::unordered_map<int, DtypeAndPartialTensorShape>&
        input_resource_variable_dtypes_and_shapes,
    bool reuse_rendezvous_for_functions) {
  EagerContext& ctx = op.EagerContext();

  Fprint128 cache_key = op_cache_key;
  /// Include soft placement policy in cache key since the placement strategy
  // can change and thus affect which kernel is picked.
  cache_key = tsl::FingerprintCat128(cache_key, ctx.AllowSoftPlacement());

  // Include run_eager_op_as_function policy in cache key since the execution
  // strategy can change and affect which kernel is picked.
  VLOG(3) << "ctx.RunEagerOpAsFunction(): " << ctx.RunEagerOpAsFunction();
  cache_key = tsl::FingerprintCat128(cache_key, ctx.RunEagerOpAsFunction());

  // The launch-time rendezvous reuse setting is bundled with the kernel, so we
  // need to include it in the cache key.
  cache_key = tsl::FingerprintCat128(cache_key, reuse_rendezvous_for_functions);

  for (int i = 0, end = input_device_ptrs.size(); i < end; ++i) {
    cache_key = tsl::FingerprintCat128(
        cache_key, Fingerprint128(input_device_ptrs[i]->name()));

    auto input_resource = input_resource_variable_dtypes_and_shapes.find(i);
    if (input_resource != input_resource_variable_dtypes_and_shapes.end()) {
      // const DtypeAndPartialTensorShape& dtype_and_shape
      const DtypeAndPartialTensorShape& dtype_and_shape =
          input_resource->second;
      // Add _Arg index, dtype and shape to "cache_key".
      cache_key = tsl::FingerprintCat128(cache_key, i);
      cache_key = tsl::FingerprintCat128(cache_key, dtype_and_shape.dtype);
      AppendTensorShapeToFingerprint(dtype_and_shape.shape, &cache_key);
    }
  }

  return cache_key;
}

// Extracts function input info for `op` with `kernel_def`.
// The following are extracted:
//   `input_device_ptrs` - The input devices of `op`.
//   `composite_devices` - Maps from a CompositeDevice name to a list of
//     physical device names.
//   `input_resource_variable_dtypes_shape` - A map from input index
//     to dtype and shapes for resource inputs.
Status ExtractFunctionInputInfo(
    EagerOperation* op, const KernelDef* kernel_def,
    std::vector<Device*>& input_device_ptrs,
    absl::flat_hash_map<string, const std::vector<string>*>& composite_devices,
    std::unordered_map<int, DtypeAndPartialTensorShape>&
        input_resource_variable_dtypes_and_shapes) {
  tsl::profiler::TraceMe activity("EagerCopyToDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  EagerContext& ctx = op->EagerContext();
  input_device_ptrs.reserve(op->Inputs().size());
  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
  Device* op_device = nullptr;
  const NodeDef* node_def = nullptr;
  if (!op->is_function()) {
    op_device = std::get<Device*>(op->Device());
    node_def = &op->MutableAttrs()->BuildNodeDef();
  }
  for (int i = 0, end = inputs->size(); i < end; ++i) {
    TensorHandle* input = (*inputs)[i];

    Device* input_device;
    bool is_host_memory_arg =
        IsHostMemoryArg(*op, node_def, op_device, kernel_def, i);
    TF_RETURN_IF_ERROR(
        GetDeviceForInput(*op, ctx, is_host_memory_arg, input, &input_device));
    VLOG(1) << op->Name() << ":input:" << i << " " << input_device->name();
    input_device_ptrs.push_back(input_device);
    CompositeDevice* composite_device = nullptr;
    if (ctx.FindCompositeDeviceFromName(input_device->name(), &composite_device)
            .ok()) {
      composite_devices[input_device->name()] =
          composite_device->underlying_devices();
    }
    if (input->dtype == DT_RESOURCE) {
      // We only care about data type and shape for resource variable inputs.
      // But we have no way to tell if input is resource variable (other than
      // looking it up in ResourceMgr, which is slow). So we just get
      // resource_dtypes_and_shapes for all DT_RESOURCE inputs. If
      // resource_dtypes_and_shapes is not empty, take the first element.
      std::vector<DtypeAndPartialTensorShape> resource_dtypes_and_shapes;
      TF_RETURN_IF_ERROR(
          input->GetResourceHandleDtypesAndShapes(&resource_dtypes_and_shapes));
      if (!resource_dtypes_and_shapes.empty()) {
        const DtypeAndPartialTensorShape& dtype_and_shape =
            resource_dtypes_and_shapes.at(0);
        input_resource_variable_dtypes_and_shapes[i] = dtype_and_shape;
      }
    }
  }
  return absl::OkStatus();
}

Status SetOpDevice(EagerContext& ctx, EagerOperation* op, Device** device) {
  // Here in local execute, set preferred device to be on the local task to
  // avoid placing op on a remote device with higher priority.
  const DeviceNameUtils::ParsedName& preferred_device =
      DeviceNameUtils::HasSomeDetails(op->GetDeviceParsedName())
          ? op->GetDeviceParsedName()
          : DeviceNameUtils::AddressSpace(ctx.HostCPUParsedName());
  // Note: We use the unwrapped op for inferring the device.
  // Without this, when wrapping CPU-only ops like RangeDataset we would
  // place the wrapped op on a GPU (if one is available) which leads to
  // errors because placer pins the function output nodes to GPU thereby
  // forcing a H2D copy of the dataset variant which is not supported.
  auto ndef = op->MutableAttrs()->BuildNodeDef();
#ifdef INTEL_MKL
  if (IsMKLEnabled() &&
      absl::StartsWith(op->Name(), mkl_op_registry::kMklOpPrefix)) {
    GetMKLNodeDef(&ndef);
  }
#endif  // INTEL_MKL

  TF_RETURN_IF_ERROR(ctx.SelectDevice(preferred_device, ndef, device));

  VLOG(1) << "PreferredDevice " << op->Name() << ": " << preferred_device;
  VLOG(1) << "Placer place op [" << op->Name()
          << "] on device: " << (*device)->name();
  VLOG(4) << "Available kernels for " << op->Name() << " are"
          << KernelsRegisteredForOp(op->Name());
  op->SetDevice(*device);
  return absl::OkStatus();
}

Fprint128 GetDeviceCacheKey(EagerOperation* op, const EagerContext& ctx) {
  Fprint128 device_cache_key = op->MutableAttrs()->CacheKey(op->DeviceName());
  device_cache_key =
      tsl::FingerprintCat128(device_cache_key, ctx.AllowSoftPlacement());
  return device_cache_key;
}

Status GetOrCreateKernelAndDevice(
    EagerOperation* op, TensorHandle** retvals, int* num_retvals,
    core::RefCountPtr<KernelAndDevice>* out_kernel) {
  EagerContext& ctx = op->EagerContext();
  Device* device = std::get<Device*>(op->Device());

  // Update the EagerOperation with information about the boolean input tensors
  // when small constant optimization is enabled.
  if (IsSmallConstantOptimizationEnabled(*op)) {
    TF_ASSIGN_OR_RETURN(BoolTensorInputs bool_inputs,
                        GetBoolInputs(op, /*delete_inputs=*/false));
    string folded_name = op->Name();
    for (const auto& [input_name, input_value] : bool_inputs) {
      folded_name = small_constants_optimizer::FoldedFunctionName(
          folded_name, input_name, input_value);
    }
    op->UpdateName(folded_name);
  }

  // Update the EagerOperation with information about the boolean input tensors
  // when the summary_optimizer is enabled.
  if (IsSummaryOptimizerEnabled(op)) {
    op->UpdateName(summary_optimizer::StrippedFunctionName(op->Name()));
  }

  // Set the EagerOperation's device prior to extracting the input_device_ptrs
  // to avoid any redundant H2D/D2H copies.
  if (device == nullptr && !op->is_function()) {
    Fprint128 device_cache_key = GetDeviceCacheKey(op, ctx);
    device = ctx.GetCachedDevice(device_cache_key);
    if (device == nullptr) {
      TF_RETURN_IF_ERROR(SetOpDevice(ctx, op, &device));
      ctx.AddDeviceToCache(device_cache_key, device);
    } else {
      op->SetDevice(device);
    }
  }

  // When running in eager_op_as_function mode Send/Recv ops need to be
  // placed on the same rendezvous to match the behaviour of eager mode.
  bool reuse_rendezvous_for_functions =
      (ctx.RunEagerOpAsFunction() && !op->is_function());

  std::vector<Device*> input_device_ptrs;
  absl::flat_hash_map<string, const std::vector<string>*> composite_devices;
  std::unordered_map<int, DtypeAndPartialTensorShape>
      input_resource_variable_dtypes_and_shapes;
  const KernelDef* kernel_def = nullptr;
  if (!op->is_function()) {
    const NodeDef* node_def = &op->MutableAttrs()->BuildNodeDef();
    kernel_def = GetKernelDef(*op, node_def, device);
  }
  if (op->is_function() || ctx.RunEagerOpAsFunction()) {
    TF_RETURN_IF_ERROR(ExtractFunctionInputInfo(
        op, kernel_def, input_device_ptrs, composite_devices,
        input_resource_variable_dtypes_and_shapes));
  }

  TF_ASSIGN_OR_RETURN(
      Fprint128 cache_key,
      GetKernelCacheKey(*op, op->MutableAttrs()->CacheKey(op->DeviceName()),
                        input_device_ptrs,
                        input_resource_variable_dtypes_and_shapes,
                        reuse_rendezvous_for_functions));
  core::RefCountPtr<KernelAndDevice> kernel = ctx.GetCachedKernel(cache_key);
  AbstractOperationPtr wrapped_op_releaser;
  // We can eliminate some overhead by running simple functions using regular
  // CallOp kernel. However, it is tricky to figure out which functions should
  // be run using CallOp. Also, currently CallOp runs neither optimization
  // passes (needed for TPU/XLA) nor grappler.
  // Here are some cases where a function should be run in multi-device mode:
  //  - Function takes at least two resources on different devices.
  //  - Function takes a resource on deviceA and a body op explicitly placed
  //  on deviceB.
  //  - Function has a colocation constraint.
  //  - Function has an explicit device annotation (which might not be using
  //    full canonical device name) different from op_device. Note that false
  //    positives are ok.
  //  - Function has a node or a (node) attribute that can potentially make
  //    the function multi-device after a rewrite pass (e.g. various XLA/TPU
  //    special nodes and attributes)
  if (kernel == nullptr) {
    VLOG(2) << "Creating new kernel for " << op->Name() << " on device "
            << DeviceNameOrUnspecified(std::get<Device*>(op->Device()));

    if (device == nullptr) {
      TF_RETURN_IF_ERROR(SetOpDevice(ctx, op, &device));
    } else {
      VLOG(1) << "Device for [" << op->Name()
              << "] already set to: " << device->name();
    }

    bool run_function_with_flr = false;
    std::optional<string> xla_compile_device_type;
    if (op->is_function()) {
      bool compile_with_xla;
      // By default we should run functions with FunctionLibraryRuntime.
      run_function_with_flr = true;
      // TODO(b/222338429): We can remove checking this once all accelerator
      // jit_compile runs through flr.
      bool has_tpu_replication = false;
      TF_RETURN_IF_ERROR(MustCompileWithXLA(op, ctx, &compile_with_xla));
      TF_RETURN_IF_ERROR(HasTPUReplication(*op, ctx, &has_tpu_replication));
      TF_RETURN_IF_ERROR(
          UpdateCompileCounter(op, ctx, compile_with_xla, has_tpu_replication));
      if (compile_with_xla && !has_tpu_replication) {
        if (ctx.JitCompileRewrite()) {
          xla_compile_device_type = op->GetDeviceParsedName().type;
        } else {
          // Note that it is not ideal, but currently correct, to set this
          // attribute after computing the kernel cache key above.
          // Note: If the attribute is already set to true, this is a noop.
          run_function_with_flr = false;
          op->MutableAttrs()->Set(kXlaMustCompileAttr, true);
        }
      }
    }

    bool function_outputs_on_op_device = false;
    if (op->is_function()) {
      GetFuncAttr(op, ctx, kOutputsOnOpDevice, &function_outputs_on_op_device)
          .IgnoreError();
    }
    VLOG(2) << op->Name() << " function_outputs_on_op_device: "
            << function_outputs_on_op_device;

    // Note: We wrap the eager op AFTER the device has been inferred to ensure
    // that placement of the NodeDef in the function is exactly the same as in
    // eager mode. This is specially important for cases where the
    // preferred device is not the actual device on which the op is run.
    // E.g. the preferred device for a `RangeDataset` op could be set to `GPU`
    // but `ctx->SelectDevice` would still place it on CPU. Placer on the other
    // hand would throw an error.
    //
    // Note: The wrapped function is never jit compiled but rather run via the
    // FLR. This is needed because certain ops e.g. `VarHandleOp` can not be
    // jit compiled. Ideally we would run this via the jit compiled path and
    // expect unsupported ops to be outside compiled but that is not supported
    // on GPUs right now.
    bool allow_small_function_optimizations = false;
    bool int_args_and_retvals_on_device = false;
    bool allow_control_flow_sync_execution = false;
    // TODO(b/176491312): Remove this if shape inference on import flag is
    // removed.
    bool shape_inference_on_tfe_dialect_import = true;
    if (ctx.RunEagerOpAsFunction() && !op->is_function()) {
      EagerOperation* wrapped_op = nullptr;
      TF_RETURN_IF_ERROR(ValidateOp(op));
      TF_RETURN_IF_ERROR(WrapInCallOp(op, &wrapped_op));
      DCHECK(wrapped_op);
      DCHECK(wrapped_op->is_function());
      wrapped_op_releaser.reset(wrapped_op);
      run_function_with_flr = true;
      allow_small_function_optimizations = true;
      allow_control_flow_sync_execution = true;
      shape_inference_on_tfe_dialect_import = false;
      int_args_and_retvals_on_device =
          IntArgsAndRetvalsOnDevice(op, kernel_def);
      op = wrapped_op;
      if (int_args_and_retvals_on_device) {
        op->MutableAttrs()->Set(FunctionLibraryDefinition::kIntsOnDeviceAttr,
                                true);
      }
    }
    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();

    FunctionLibraryRuntime* flr =
        device == nullptr ? nullptr : ctx.func_lib(device);
    if (device != nullptr && flr == nullptr) {
      return errors::NotFound(
          "Unable to find a FunctionLibraryRuntime corresponding to device ",
          device->name());
    }
    auto runner = (flr != nullptr && flr->runner() != nullptr) ? flr->runner()
                                                               : ctx.runner();
    GraphCollector* graph_collector = nullptr;
    if (ctx.ShouldStoreGraphs()) {
      graph_collector = ctx.GetGraphCollector();
    }
    // Treat the function as multi_device only when we are not compiling
    // it wholly with XLA. When compiling wholly with XLA, flr->CreateKernel
    // will create an XlaLaunchOp kernel to compile and run the function.
    if (run_function_with_flr) {
      // Multi-device functions don't use the rendezvous from eager context.
      // If we use that rendezvous, multiple concurrent calls to the same
      // function will likely result in collisions. However, this also means
      // that we don't support legitimate sending/receiving across function
      // boundary.
      VLOG(2) << "Running " << ndef.op() << " using multi-device function. "
              << "Full node_def=" << ndef.DebugString();
      std::function<int64_t()> get_op_id = nullptr;
#if !defined(IS_MOBILE_PLATFORM)
      get_op_id = [&ctx]() { return ctx.RemoteMgr()->NextOpId(); };
#endif  // IS_MOBILE_PLATFORM

      auto rendezvous_creator =
          ctx.RendezvousFactory(reuse_rendezvous_for_functions);
      kernel.reset(new KernelAndDeviceFunc(
          flr, ctx.pflr(), std::move(input_device_ptrs),
          std::move(composite_devices),
          std::move(input_resource_variable_dtypes_and_shapes), runner,
          ctx.GetCollectiveExecutorHandle(), ctx.HostCPU(), op->Name(),
          function_outputs_on_op_device, allow_small_function_optimizations,
          allow_control_flow_sync_execution,
          shape_inference_on_tfe_dialect_import, int_args_and_retvals_on_device,
          xla_compile_device_type, ctx.AllowSoftPlacement(),
          std::move(rendezvous_creator), get_op_id));
    } else {
      VLOG(2) << "Running " << ndef.op() << " using op kernel. "
              << ". Full node_def=" << ndef.DebugString();
      kernel.reset(new KernelAndDeviceOp(
          ctx.GetRendezvous(), ctx.LogMemory(), flr, runner,
          ctx.GetCollectiveExecutorHandle(), ctx.HostCPU()));
    }

    TF_RETURN_IF_ERROR(kernel->Init(ctx.LogDevicePlacement(), ndef,
                                    graph_collector, op->eager_func_params()));

    // Exclude tf.data op kernels from being cached. The reason for this is
    // that tf.data op kernels that accept a user-defined function will have a
    // unique cache key every time they are executed (because the user-defined
    // function is traced every time). Caching such kernels provides no
    // benefit and in some cases results in linear memory growth of use
    // programs that build input pipeline graphs in a loop.
    const OpDef* op_def;
    if (op->is_function()) {
      const FunctionDef* function_def = op->GetFunctionDef();
      if (function_def != nullptr) {
        op_def = &(function_def->signature());
      } else {
        TF_RETURN_IF_ERROR(OpDefForOp(op->Name().c_str(), &op_def));
      }
    } else {
      TF_RETURN_IF_ERROR(OpDefForOp(op->Name().data(), &op_def));
    }
    if (op_def != nullptr && KernelCacheEnabled(*op_def)) {
      // TODO(intel-tf): Implement an eviction policy to prevent potential
      // memory growth (https://github.com/tensorflow/tensorflow/issues/58676)
      VLOG(2) << "Caching op " << op->Name();
      // If the kernel is already in the cache, this discards the passed-in
      // kernel and returns the cached kernel.
      kernel = ctx.AddKernelToCache(cache_key, std::move(kernel));
    }
  }

  int num_outputs = kernel->num_outputs();
  if (num_outputs > *num_retvals) {
    return errors::InvalidArgument("Expecting ", num_outputs,
                                   " outputs, but *num_retvals is ",
                                   *num_retvals);
  }
  *num_retvals = num_outputs;

  kernel->Ref();  // Ownership of reference is passed to out_kernel.
  out_kernel->reset(kernel.get());
  return absl::OkStatus();
}

Status CreateUnshapedOutput(
    const KernelAndDevice& kernel, const int output_num, Device* output_device,
    const DataType& output_dtype,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    EagerContext* ctx, TensorHandle** output) {
#if defined(IS_MOBILE_PLATFORM)
  return errors::Unimplemented(
      "Remote outputs are not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
  int64_t op_id;
  if (eager_func_params.has_value()) {
    op_id = eager_func_params.value().op_id;
  } else {
    return errors::InvalidArgument(
        "Unable to find a remote op id for a remote output of ", kernel.name());
  }
  string remote_task;
  if (!DeviceNameUtils::GetTaskName(output_device->parsed_name(),
                                    &remote_task)) {
    return errors::InvalidArgument(
        "Unable to find remote task corresponding to device ",
        output_device->name());
  }
  if (ctx->RemoteMgr()->IsMaster()) {
    *output = TensorHandle::CreateUnshapedRemoteHandle(
        op_id, output_num, remote_task, output_dtype, output_device, ctx);
  } else {
    *output = TensorHandle::CreateLazyRemoteHandle(op_id, output_num,
                                                   output_dtype, output_device,
                                                   /*is_ready=*/false, ctx);
  }
  return absl::OkStatus();
#endif  // !IS_MOBILE_PLATFORM
}

Status AddOrExecuteNode(core::RefCountPtr<KernelAndDevice> kernel,
                        EagerOperation* op, TensorHandle** retvals) {
  EagerExecutor& executor = op->Executor();
  EagerContext& ctx = op->EagerContext();
  GraphCollector* graph_collector = nullptr;
  if (ctx.ShouldStoreGraphs()) {
    graph_collector = ctx.GetGraphCollector();
  }
  const int num_outputs = kernel->num_outputs();
  std::optional<EagerFunctionParams> eager_func_params =
      op->eager_func_params();
  if (kernel->IsCrossProcess() && !eager_func_params.has_value()) {
    // Create an eager op id for a cross-process function if not exist.
#if defined(IS_MOBILE_PLATFORM)
    return errors::Unimplemented(
        "Cross-process functions are not supported on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
    const int64_t op_id = ctx.RemoteMgr()->NextOpId();
    eager_func_params = EagerFunctionParams{
        op_id, /* is_component_function= */ false, /* step_id= */ std::nullopt};
#endif  // !IS_MOBILE_PLATFORM
  }
  if (executor.Async()) {
    const DataTypeVector& output_dtypes = kernel->output_dtypes();
    for (int i = 0, end = num_outputs; i < end; ++i) {
      Device* output_device = ctx.CanonicalDevice(kernel->OutputDevice(i));
      if (output_device == nullptr || output_device->IsLocal()) {
        retvals[i] = TensorHandle::CreateEmptyLocalHandle(
            /* d= */ output_device, /* op_device= */ kernel->device(),
            /* resource_device= */ kernel->OutputResourceDevice(i),
            output_dtypes[i], &ctx);
      } else {
        TF_RETURN_IF_ERROR(
            CreateUnshapedOutput(*kernel, i, output_device, output_dtypes[i],
                                 eager_func_params, &ctx, &retvals[i]));
      }
    }
    const absl::InlinedVector<TensorHandle*, 4>* inputs;
    TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
    auto node = std::make_unique<AsyncExecuteNode>(
        &ctx, *inputs, eager_func_params, std::move(kernel), graph_collector,
        op->GetCancellationManager(),
        absl::Span<TensorHandle*>(retvals, num_outputs), op->GetStackTrace());
    // Release the inputs from the eager operation since the AsyncExecuteNode
    // would have taken ownership. This allows the inputs to be forwarded if
    // possible.
    op->Clear();
    // For async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(b/137118203): Consider executing "cheap" kernels inline for
    // performance.
    return executor.AddOrExecute(std::move(node));
  } else {
    for (int i = 0, end = num_outputs; i < end; ++i) {
      retvals[i] = nullptr;
    }
    const absl::InlinedVector<TensorHandle*, 4>* inputs;
    TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
    ExecuteNode node(&ctx, *inputs, eager_func_params, kernel, graph_collector,
                     op->GetCancellationManager(),
                     {retvals, static_cast<size_t>(num_outputs)},
                     op->GetStackTrace());
    Status s = executor.SyncExecute(&node);
    // We release the inputs AFTER executing the operation in sync mode since
    // ExecuteNode does not increment the reference count and thus does not have
    // ownership of the inputs while executing.
    op->Clear();
    return s;
  }
}

// There are a lot of references to devices in this function and around.
// Here is what they mean:
//  EagerOperation::Device(): The device on which the user requested the op
//    be executed, except if we had to change the device due to resource inputs
//    or CPU pinning. If the user did not request a device, the op does not
//    take resources, and we did not pin it to CPU, the device can be nullptr.
//  KernelAndDevice::Device(): The first time we see an op (combined with
//    its attributes), we need to create a KernelAndDevice object for it.
//    If op->Device() is a nullptr, we select a device for the op when
//    creating the KernelAndDevice. A concrete device will always be selected
//    here except when `op` is a function to be executed using function library
//    runtime. In this case, we don't select a device because running
//    a function with explicitly requested device has different behavior than
//    running without an explicitly requested device.
Status EagerLocalExecute(EagerOperation* op, TensorHandle** retvals,
                         int* num_retvals) {
  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
      op->op_name(), op->eager_func_params().has_value()
                         ? op->eager_func_params().value().step_id.value_or(0)
                         : 0);
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerLocalExecute: ", op->Name()); },
      tsl::profiler::TraceMeLevel::kInfo);
  EagerContext& ctx = op->EagerContext();
  auto& executor = op->Executor();
  TF_RETURN_IF_ERROR(executor.status());

  core::RefCountPtr<KernelAndDevice> kernel;
  auto status = GetOrCreateKernelAndDevice(op, retvals, num_retvals, &kernel);

#ifdef INTEL_MKL
  if (IsMKLEnabled() && kernel != nullptr &&
      op->Device() == kVariantDeviceNull) {
    // oneDNN optimization pass relies on the op's assigned device to determine
    // whether it can be rewritten.
    op->SetDevice(kernel->device());
  }
#endif  // INTEL_MKL

  // Run all the registered rewrite pass after the placement, regardless whether
  // the placement is successful or not. The passes can either create new ops
  // (without placement) or update some fields of the input op.
  std::unique_ptr<tensorflow::EagerOperation> out_op;
  TF_RETURN_IF_ERROR(EagerOpRewriteRegistry::Global()->RunRewrite(
      EagerOpRewriteRegistry::POST_PLACEMENT, op, &out_op));
  if (out_op) {
    op = out_op.get();
    // If the out op doesn't have device, either because it is a new op or
    // the op wasn't placed successfully, then we do the placement again.
    if (op->Device() == kVariantDeviceNull) {
      status = GetOrCreateKernelAndDevice(op, retvals, num_retvals, &kernel);
    }
  }
  if (!status.ok()) return status;

  int num_outputs = kernel->num_outputs();
  TF_RETURN_IF_ERROR(ValidateInputTypeAndPlacement(&ctx, op, kernel));

  if (ctx.LogDevicePlacement() || VLOG_IS_ON(1)) {
    string msg = strings::StrCat("Executing op ", op->Name(), " in device ",
                                 kernel->device()->name());
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }

  Status s = AddOrExecuteNode(std::move(kernel), op, retvals);
  // Since the operation failed, we need to Unref any outputs if they were
  // allocated.
  if (!s.ok()) {
    for (int i = 0, end = num_outputs; i < end; ++i) {
      if (retvals[i] != nullptr) {
        retvals[i]->Unref();
        retvals[i] = nullptr;
      }
    }
  }

  return s;
}

// Run a Pack op to pack the tensors pointed by a packed input TensorHandle if
// the op is a primitive op.
Status MaybePackInputTensor(EagerOperation* op) {
  if (op->is_function() || op->EagerContext().RunEagerOpAsFunction()) {
    // Functions could take packed TensorHandles as inputs.
    return absl::OkStatus();
  }
  EagerContext& ctx = op->EagerContext();
  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
  for (int i = 0; i < inputs->size(); ++i) {
    TensorHandle* handle = (*inputs)[i];
    if (handle->Type() == TensorHandle::PACKED) {
      EagerOperation pack_op(&ctx);
      TF_RETURN_IF_ERROR(pack_op.Reset("Pack", /*device_name=*/nullptr,
                                       /*remote=*/false, /*executor=*/nullptr));
      pack_op.MutableAttrs()->Set("N", handle->NumPackedHandles());
      pack_op.MutableAttrs()->Set("T", handle->dtype);
      for (int i = 0; i < handle->NumPackedHandles(); ++i) {
        tensorflow::TensorHandle* h = nullptr;
        TF_RETURN_IF_ERROR(handle->ExtractPackedHandle(i, &h));
        TF_RETURN_IF_ERROR(pack_op.AddInput(h));
      }
      int num_retvals = 1;
      absl::FixedArray<tensorflow::TensorHandle*> retvals(num_retvals);
      TF_RETURN_IF_ERROR(
          EagerLocalExecute(&pack_op, retvals.data(), &num_retvals));
      tensorflow::TensorHandle* ret = retvals.at(0);
      op->UpdateInput(i, ret);
      ret->Unref();
    }
  }
  return absl::OkStatus();
}

#if !defined(IS_MOBILE_PLATFORM)
void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
  EagerContext& ctx = op->EagerContext();

  remote_op->set_id(ctx.RemoteMgr()->NextOpId());
  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMapWithoutDefaults(remote_op->mutable_attrs());
  remote_op->set_device(std::get<Device*>(op->Device())->name());
  remote_op->set_is_function(op->is_function());
}

Status StoreResourceDtypesAndShapes(const eager::Operation& remote_op,
                                    const DataTypeVector& output_dtypes,
                                    TensorHandle** retvals) {
  if (remote_op.name() == "VarHandleOp") {
    if (output_dtypes.size() != 1) {
      return errors::Internal("VarHandleOp should only have one output.");
    }
    if (output_dtypes[0] != DT_RESOURCE) {
      return errors::Internal(
          "The output of VarHandleOp should be a DT_RESOURCE.");
    }
    AttrSlice attr_slice = AttrSlice(&remote_op.attrs());
    const AttrValue* dtype;
    TF_RETURN_IF_ERROR(attr_slice.Find("dtype", &dtype));
    const AttrValue* shape;
    TF_RETURN_IF_ERROR(attr_slice.Find("shape", &shape));
    retvals[0]->SetResourceHandleDtypeAndShape(
        {DtypeAndPartialTensorShape{dtype->type(), shape->shape()}});
  }
  return absl::OkStatus();
}

Status EagerRemoteExecute(EagerOperation* op, TensorHandle** retvals,
                          int* num_retvals) {
  EagerContext& ctx = op->EagerContext();

  // TODO(fishx): Remove following code when lazy tensor copy is ready.
  if (op->Device() == kVariantDeviceNull) {
    tensorflow::Device* device = nullptr;
    string device_name = op->DeviceName();
    TF_RETURN_IF_ERROR(ctx.FindDeviceFromName(device_name.c_str(), &device));
    op->SetDevice(device);
  }

  core::RefCountPtr<eager::EagerClient> eager_client;
  uint64 context_id = ctx.GetContextId();
  TF_RETURN_IF_ERROR(ctx.GetClient(op->GetDeviceParsedName(), &eager_client));
  string remote_task;
  if (!DeviceNameUtils::GetTaskName(op->GetDeviceParsedName(), &remote_task)) {
    return errors::InvalidArgument(
        "Unable to find remote task corresponding to device ",
        op->DeviceName());
  }

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(context_id);

  eager::Operation* remote_op = request->add_queue()->mutable_operation();

  tensorflow::Device* op_device = std::get<Device*>(op->Device());
  {
    tsl::profiler::TraceMe activity("CopyInputToExpectedDevice",
                                    tsl::profiler::TraceMeLevel::kInfo);
    const bool is_function = op->is_function();
    const absl::InlinedVector<TensorHandle*, 4>* inputs;
    TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
    for (int i = 0, end = inputs->size(); i < end; i++) {
      tensorflow::TensorHandle* input = (*inputs)[i];
      tensorflow::Device* input_device = input->device();
      tensorflow::Device* input_device_or_cpu = input->DeviceOrHostCPU(ctx);
      const string* input_device_name = &input_device_or_cpu->name();
      bool serialize_resource_dtype_and_shape = false;
      if (op_device != input_device &&
          // If the expected and actual devices are on the same task, don't
          // explicitly copy, and instead depend on the copy to happen locally
          // when the op is executed on the device.
          !ctx.OnSameTask(op_device, input_device)) {
        if (!is_function || input_device_or_cpu->IsLocal()) {
          tensorflow::Device* remote_cpu_device;
          TF_RETURN_IF_ERROR(
              ctx.CPUDeviceOnTask(op_device, &remote_cpu_device));
          // Always copy to the remote CPU so that the actual device can be
          // correctly determined after the kernel is selected/instantiated,
          // since the op might have its inputs on host memory.
          TensorHandle* handle = input;
          Device* handle_device = handle->DeviceOrHostCPU(ctx);
          // If the input is already on the right device, then nothing to do.
          if (remote_cpu_device != handle_device) {
            VLOG(6) << "remote_cpu_device != handle_device";
            TF_RETURN_IF_ERROR(CopyInputToExpectedDevice(
                &ctx, op, op_device, handle, i, handle_device,
                remote_cpu_device, &handle));
            op->UpdateInput(i, handle);
            input = handle;
            input_device = remote_cpu_device;
            input_device_name = &remote_cpu_device->name();
            // Unref handle since it has a ref as an input now
            handle->Unref();
          }
        } else {
          serialize_resource_dtype_and_shape =
              (input->dtype == DT_RESOURCE) &&
              (!input->HasResourceShapeMirror(op_device,
                                              ctx.GetContextViewId()));
        }
      }
      int64_t num_elements;
      TF_RETURN_IF_ERROR(input->NumElements(&num_elements));
      if ((input->Type() == TensorHandle::HandleType::LOCAL) &&
          (num_elements == 1) && (input->DataType() != DT_VARIANT) &&
          SendAsProtosWhenPossible()) {
        auto* input_tensor_proto = remote_op->add_op_inputs()->mutable_tensor();
        const tensorflow::Tensor* input_tensor = nullptr;
        TensorHandle* local_cpu_input_handle = nullptr;
        TF_RETURN_IF_ERROR(EagerCopyToDevice(input, &ctx, &ctx.Executor(),
                                             ctx.HostCPU(), false,
                                             &local_cpu_input_handle));
        TF_RETURN_IF_ERROR(local_cpu_input_handle->Tensor(&input_tensor));
        input_tensor->AsProtoTensorContent(input_tensor_proto);
        // `TensorHandle::AddResourceShapeMirror` can change `input` but only if
        // `TensorHandle::handle_dtypes_and_shapes_` is not empty. And that
        // requires `TensorHandle::dtype` to be equal to `DT_RESOURCE` which
        // cannot be the case when we are here. So nothing else to do.
      } else {
        auto* input_handle =
            remote_op->add_op_inputs()->mutable_remote_handle();
        // For a remote component function, a function execution request and an
        // input generation request may come from different workers. We need to
        // guarantee that the input generation request is processed before the
        // function execution request, so wait until the remote input is ready
        // before sending it to the multi-device function device.
        bool wait_until_ready =
            SkipRemoteHandleWaitReady() ? false : op->is_function();
        TF_RETURN_IF_ERROR(ctx.RemoteMgr()->SerializeRemoteTensorHandle(
            input, wait_until_ready, input_handle, input_device,
            *input_device_name, serialize_resource_dtype_and_shape));
        if (!input_handle->resource_dtypes_and_shapes().empty()) {
          TF_RETURN_IF_ERROR(
              input->AddResourceShapeMirror(op_device, input_handle->op_id(),
                                            input_handle->output_num(), &ctx));
        }
      }
    }
  }

  PrepareRemoteOp(remote_op, op);

  DataTypeVector output_dtypes;
  TF_RETURN_IF_ERROR(GetOutputDTypes(op, &output_dtypes));

  const size_t num_outputs = output_dtypes.size();
  if (num_outputs != *num_retvals) {
    return errors::InvalidArgument(
        "num_retvals does not match expected output dtypes");
  }
  *num_retvals = num_outputs;

  const tensorflow::uint64 id = remote_op->id();
  for (size_t i = 0; i < num_outputs; ++i) {
    // TODO(nareshmodi): Change the callback to instead add the decref to a
    // list of pending decrefs that we can send as a batch with the next
    // execute.

    // The device_ and resource_device_ of this TensorHandle might be
    // incorrect. For multi-device functions, we don't know the output device
    // until the function is instantiated on a remote worker. Luckily, we don't
    // need to know the correct remote device here. We just need to know that it
    // is remote. If we need copy this tensor to this process or run any ops
    // which take this tensor as an input, block until the correct device is
    // set.
    const bool unknown_device = op->is_function();
    retvals[i] = TensorHandle::CreateUnshapedRemoteHandle(
        id, i, remote_task, output_dtypes[i], op_device, &ctx, unknown_device);
  }

  // Store the data type and shape of a remote resource variable on the
  // corresponding remote TensorHandle (output of 'VarHandleOp').
  // If the variable is an input of a remote function, the function may need
  // the type and shape during function instantiation. Store the type and
  // shape on eager master and sent them to the default function device along
  // with the EnqueueRequest.
  TF_RETURN_IF_ERROR(
      StoreResourceDtypesAndShapes(*remote_op, output_dtypes, retvals));

  auto& executor = op->Executor();
  VLOG(4) << "Execute remote eager op: " << op->Name()
          << " (is async?: " << executor.Async() << ").";

  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));

  std::unique_ptr<EagerNode> node(new eager::RemoteExecuteNode(
      &op->EagerContext(), std::move(request), op_device,
      ctx.GetContextViewId(), eager_client.get(), op->GetCancellationManager(),
      op->MutableAttrs()->BuildNodeDef(), op->FuncLibDef(), *inputs,
      {retvals, num_outputs}));

  if (op->EagerContext().LogDevicePlacement() || VLOG_IS_ON(1)) {
    string msg = strings::StrCat(
        "Executing op ", op->Name(), " on task ",
        DeviceNameUtils::ParsedNameToString(op->GetDeviceParsedName()));
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }

  Status s = executor.AddOrExecute(std::move(node));
  // Since the operation failed, we need to Unref any outputs that were
  // allocated.
  if (!s.ok()) {
    for (size_t i = 0; i < num_outputs; ++i) {
      retvals[i]->Unref();
      // Ensure that any smart pointers created to wrap results become noops
      // rather than operating on invalid memory.
      retvals[i] = nullptr;
    }
  }

  return s;
}
#endif  // IS_MOBILE_PLATFORM

Status GetKernelOutputs(
    std::vector<EagerKernelRet>* outputs, int num_outputs,
    TensorHandle** retvals, EagerContext* ctx, KernelAndDevice* kernel,
    const absl::optional<EagerFunctionParams>& eager_func_params) {
  for (int i = 0, end = num_outputs; i < end; ++i) {
    if (retvals[i] == nullptr) {
      EagerKernelRet& ret = (*outputs)[i];
      Device* output_device = ctx->CanonicalDevice(kernel->OutputDevice(i));
      if (ret.index() == 0) {
        retvals[i] = TensorHandle::CreateLocalHandle(
            std::move(std::get<Tensor>(ret)),
            /* d= */ output_device,
            /* op_device= */ kernel->device(),
            /* resource_device= */ kernel->OutputResourceDevice(i), ctx);
      } else {
        const DataTypeVector& output_dtypes = kernel->output_dtypes();
        TF_RETURN_IF_ERROR(
            CreateUnshapedOutput(*kernel, i, output_device, output_dtypes[i],
                                 eager_func_params, ctx, &retvals[i]));
#if !defined(IS_MOBILE_PLATFORM)
        TF_RETURN_IF_ERROR(
            retvals[i]->SetRemoteShape(std::get<TensorShape>(ret),
                                       output_device, ctx->GetContextViewId()));
#endif  // IS_MOBILE_PLATFORM
      }
    } else {
      if (!kernel->IsFunction() &&
          TF_PREDICT_FALSE(kernel->device() != retvals[i]->op_device())) {
        return errors::Internal(
            "Kernel output tensor handle has a different op device than the "
            "kernel. This should never happen.");
      }
      if (TF_PREDICT_FALSE(ctx->CanonicalDevice(kernel->OutputDevice(i)) !=
                           retvals[i]->device())) {
        return errors::Internal(
            "Kernel output tensor handle locates on a different device than "
            "the specified kernel output device. This should never happen.");
      }

      EagerKernelRet& ret = (*outputs)[i];
      if (ret.index() == 0) {
        TF_RETURN_IF_ERROR(retvals[i]->SetTensor(
            std::move(std::get<Tensor>(ret)),
            ctx->CanonicalDevice(kernel->OutputDevice(i))));
      } else {
#if defined(IS_MOBILE_PLATFORM)
        return errors::Unimplemented(
            "Remote outputs are not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
        TF_RETURN_IF_ERROR(retvals[i]->SetRemoteShape(
            std::get<TensorShape>(ret), retvals[i]->device(),
            ctx->GetContextViewId()));
#endif  // !IS_MOBILE_PLATFORM
      }
    }
  }
  return absl::OkStatus();
}

void CollectGraphs(EagerContext* ctx) {
  mutex_lock ml(*ctx->MetadataMu());

  GraphCollector* collector = ctx->GetGraphCollector();
  mutex_lock mll(collector->mu);

  // Adding to partition graphs for backward compatibility.
  for (const auto& graph : collector->partitioned_graphs) {
    *ctx->RunMetadataProto()->add_partition_graphs() = graph;
  }

  if (collector->dirty) {
    auto* function_graphs = ctx->RunMetadataProto()->add_function_graphs();
    *function_graphs->mutable_post_optimization_graph() =
        collector->optimized_graph;
    *function_graphs->mutable_pre_optimization_graph() = collector->raw_graph;
    for (const auto& graph : collector->partitioned_graphs) {
      *function_graphs->add_partition_graphs() = graph;
    }
  }

  collector->ClearGraphs();
}
}  // namespace

Status DoEagerExecute(EagerOperation* op, TensorHandle** retvals,
                      int* num_retvals) {
  tsl::profiler::TraceMe activity([&] {
    return tsl::profiler::TraceMeEncode(
        "EagerExecute",
        {{"eager_op", op->Name()}, {"is_func", op->is_function()}});
  });

  if (!op->Executor().Async()) {
    VLOG(6) << "op: " << op->Name() << " is not Async.";
    if (!op->EagerContext()
             .GetGlobalRendezvousForFunctionLocalRendezvousStatus()
             .ok()) {
      VLOG(6) << "global_rendezvous_for_functions_ is in bad state. Resetting.";
      op->EagerContext().ResetGlobalRendezvousForFunction();
    }
    // In sync mode, always clear error to maintain the same behavior as before.
    // TODO(b/141004939): Remove this.
    op->Executor().ClearError();
  }

  std::unique_ptr<tensorflow::EagerOperation> out_op;
  TF_RETURN_IF_ERROR(EagerOpRewriteRegistry::Global()->RunRewrite(
      EagerOpRewriteRegistry::PRE_EXECUTION, op, &out_op));

  if (op->IsLocal()) {
    if (out_op) {
      op = out_op.get();
    }
    TF_RETURN_IF_ERROR(MaybePackInputTensor(op));
    return EagerLocalExecute(op, retvals, num_retvals);
  }

#if defined(IS_MOBILE_PLATFORM)
  return errors::Unimplemented(
      "Eager's remote execution is not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
  if (out_op) {
    op = out_op.get();
  }
  return EagerRemoteExecute(op, retvals, num_retvals);
#endif  // !IS_MOBILE_PLATFORM
}

// TODO(gjn): Consider moving into ExecuteNode class
Status EagerKernelExecute(
    EagerContext* ctx, const absl::InlinedVector<TensorHandle*, 4>& op_inputs,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const core::RefCountPtr<KernelAndDevice>& kernel,
    GraphCollector* graph_collector, CancellationManager* cancellation_manager,
    absl::Span<TensorHandle*> retvals,
    const absl::optional<ManagedStackTrace>& stack_trace) {
  tsl::profiler::TraceMe activity("EagerKernelExecute",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<EagerKernelRet> outputs(1);

  ExecuteNodeArgs inputs(op_inputs.size());
  TF_RETURN_IF_ERROR(inputs.Init(ctx, op_inputs, kernel));
  // TODO(apassos) figure out how to record stats for ops which are a part of
  // functions.
  // TODO(b/111859745): When we support recovering from kernel/device errors, we
  // would need to call XlaDevice::EnsureDeviceContextOk() before using an XLA
  // device. We don't call it now because it is an unneeded overhead (it
  // acquires a lock) and we can't recover from errors anyway.
  ScopedStepContainer* container = ctx->StepContainer();
  tsl::CoordinationServiceAgent* coord_agent = nullptr;
#if !defined(IS_MOBILE_PLATFORM)
  if (ctx->GetDistributedManager() != nullptr)
    coord_agent = ctx->GetDistributedManager()->GetCoordinationServiceAgent();
#endif  // !IS_MOBILE_PLATFORM
  TF_RETURN_IF_ERROR(kernel->Run(container, inputs, &outputs,
                                 cancellation_manager, eager_func_params,
                                 stack_trace, coord_agent));
  if (graph_collector != nullptr) {
    CollectGraphs(ctx);
  }

  if (TF_PREDICT_FALSE(retvals.size() != outputs.size())) {
    return errors::Internal(
        "EagerKernelExecute returns a list of ", outputs.size(),
        " tensors but ", retvals.size(),
        " is expected. This should never "
        "happen. Please file a bug with the TensorFlow team.");
  }
  return GetKernelOutputs(&outputs, retvals.size(), retvals.data(), ctx,
                          kernel.get(), eager_func_params);
}

Status EagerExecute(EagerOperation* op, TensorHandle** retvals,
                    int* num_retvals) {
  if (VLOG_IS_ON(1) && op->is_function()) {
    const std::string& op_name = op->Name();
    const std::string& exec_mode = op->IsLocal() ? "local" : "remote";
    const std::string& device_name = op->DeviceName();

    auto msg = absl::StrCat("eager executing ", exec_mode, " operation '",
                            op_name, "'");

    if (!device_name.empty()) {
      absl::StrAppend(&msg, " on device '", device_name, "'");
    }

    VLOG(1) << "Entering " << msg;

    Status status = DoEagerExecute(op, retvals, num_retvals);

    VLOG(1) << "Exiting " << msg << ", status code is " << status;

    return status;
  }
  return DoEagerExecute(op, retvals, num_retvals);
}

namespace {

Status LocalEagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                              EagerExecutor* executor, Device* dstd,
                              bool mirror, TensorHandle** result) {
  TF_RETURN_IF_ERROR(executor->status());
  Device* d = ctx->CanonicalDevice(dstd);
  if (mirror && h->HasLocalMirror(d)) {
    h->Ref();
    *result = h;
    return absl::OkStatus();
  }

  bool async = executor->Async();
  if (mirror) {
    h->Ref();
    *result = h;

    if (h->HasLocalMirror(d)) {
      return absl::OkStatus();
    }

    // We don't bother adding an empty local mirror in sync mode since we'll be
    // executing the operation directly and be calling AddLocalMirror. A
    // reference count is still needed which will be removed if the operation
    // fails.
    if (async) {
      Status s = h->AddEmptyLocalMirror(d);
      if (!s.ok()) {
        // If a mirror was added since we called HasLocalMirror then just return
        // since another thread has already added the mirror.
        if (s.code() == error::Code::ALREADY_EXISTS) {
          return absl::OkStatus();
        }

        // Remove the previously added reference count since adding the mirror
        // failed.
        h->Unref();
        *result = nullptr;
        return s;
      }
    }
  } else {
    *result = TensorHandle::CreateEmptyLocalHandle(
        d, dstd, h->resource_device(), h->dtype, ctx);
  }

  Status s;
  if (async) {
    // Note that `h` may not be currently ready. However execution order will
    // make sure that `h` is ready before the copy is actually done.
    std::unique_ptr<EagerNode> node(
        new CopyToDeviceNode(h, *result, d, *ctx, async, mirror));
    s = executor->AddOrExecute(std::move(node));
  } else {
    CopyToDeviceNode node(h, *result, d, *ctx, async, mirror);
    s = executor->SyncExecute(&node);
  }

  // Since the operation failed, we need to Unref any outputs that were
  // allocated.
  if (!s.ok()) {
    (*result)->Unref();
    *result = nullptr;
  }

  return s;
}

}  // namespace

Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                         EagerExecutor* executor, Device* device, bool mirror,
                         TensorHandle** result) {
  TF_RETURN_IF_ERROR(h->WaitUnknownDevice());
  auto send_device = h->DeviceOrHostCPU(*ctx);
  bool sender_is_local = send_device->IsLocal();

  bool receiver_is_local = device->IsLocal();

  if (!executor->Async()) {
    // In sync mode, always clear error to maintain the same behavior as before.
    // TODO(b/141004939): Remove this.
    executor->ClearError();
  }

  if (sender_is_local && receiver_is_local) {
    return LocalEagerCopyToDevice(h, ctx, executor, device, mirror, result);
  } else {
#if defined(IS_MOBILE_PLATFORM)
    return errors::Unimplemented(
        "Eager's remote execution is not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
    uint64 recv_op_id = 0;
    if (receiver_is_local) {
      Device* d = ctx->CanonicalDevice(device);
      // TODO(gjn): Need to add support for async execution. Note if receiver
      // is local, we need to first add support in TensorHandle to wait on local
      // mirrors.
      if (mirror) {
        h->Ref();
        *result = h;

        if (h->HasLocalMirror(d)) {
          return absl::OkStatus();
        }

        Status s = h->AddEmptyLocalMirror(d);
        if (!s.ok()) {
          // If a mirror was added since we called HasLocalMirror then just
          // return since another thread has already added the mirror.
          if (s.code() == error::Code::ALREADY_EXISTS) {
            return absl::OkStatus();
          }

          // Remove the previously added reference count since adding the mirror
          // failed.
          h->Unref();
          *result = nullptr;
          return s;
        }
      } else {
        *result = TensorHandle::CreateEmptyLocalHandle(
            /* d= */ d, /* op_device= */ device,
            /*resource_device=*/nullptr, h->dtype, ctx);
      }
    } else {
      if (mirror) {
        if (h->HasRemoteMirror(device, ctx->GetContextViewId())) {
          h->Ref();
          *result = h;
          return absl::OkStatus();
        }
      }
      string remote_task;
      if (!DeviceNameUtils::GetTaskName(device->parsed_name(), &remote_task)) {
        return errors::InvalidArgument(
            "Unable to find remote task corresponding to device ",
            device->name());
      }
      recv_op_id = ctx->RemoteMgr()->NextOpId();
      if (mirror) {
        TF_RETURN_IF_ERROR(h->AddUnshapedRemoteMirror(device, recv_op_id, 0,
                                                      remote_task, ctx));
        h->Ref();
        *result = h;
      } else {
        *result = TensorHandle::CreateUnshapedRemoteHandle(
            recv_op_id, 0, remote_task, h->dtype, device, ctx);
      }
    }

    auto node = std::make_unique<eager::RemoteCopyNode>(
        ctx, executor, h, result[0], device, recv_op_id);
    Status s = executor->AddOrExecute(std::move(node));
    if (!s.ok()) {
      result[0]->Unref();
      result[0] = nullptr;
    }
    return s;
#endif  // !IS_MOBILE_PLATFORM
  }
}

namespace {
// Low-level utility function to execute the kernel specified by `kernel` on
// `kernel->device()`, with the provided inputs as `op_inputs` in the 'ctx'.
// Different from `EagerKernelExecute` that ties up the thread until the
// underlying function finishes execute, this function does not block the thread
// and could return before the function execution finishes. The provided
// `StatusCallback` will be triggered after function execution with its status.
void EagerKernelExecuteAsync(
    EagerContext* ctx, const absl::InlinedVector<TensorHandle*, 4>& op_inputs,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const core::RefCountPtr<KernelAndDevice> kernel,
    GraphCollector* graph_collector, CancellationManager* cancellation_manager,
    TensorHandle** retvals, int num_outputs, StatusCallback done) {
  auto inputs = std::make_shared<ExecuteNodeArgs>(op_inputs.size());
  auto outputs = std::make_shared<std::vector<EagerKernelRet>>(1);

  Status s = inputs->Init(ctx, op_inputs, kernel);
  if (!s.ok()) {
    done(s);
    return;
  }
  tsl::CoordinationServiceAgent* coord_agent = nullptr;
#if !defined(IS_MOBILE_PLATFORM)
  if (ctx->GetDistributedManager() != nullptr)
    coord_agent = ctx->GetDistributedManager()->GetCoordinationServiceAgent();
#endif  // !IS_MOBILE_PLATFORM

  kernel->Ref();  // Ownership of reference is transferred to the callback
  kernel->RunAsync(
      ctx->StepContainer(), *inputs, outputs.get(), cancellation_manager,
      eager_func_params, coord_agent,
      [retvals, inputs, outputs, num_outputs, ctx, graph_collector,
       eager_func_params, kernel_raw = kernel.get(),
       done = std::move(done)](const Status& s) {
        auto wrapped_done = [&](const Status& s) {
          kernel_raw->Unref();
          done(s);
        };
        if (!s.ok()) {
          wrapped_done(s);
          return;
        }
        if (graph_collector != nullptr) {
          CollectGraphs(ctx);
        }
        DCHECK_EQ(num_outputs, outputs->size());
        wrapped_done(GetKernelOutputs(outputs.get(), num_outputs, retvals, ctx,
                                      kernel_raw, eager_func_params));
      });
}
}  // namespace

// Low-level utility to run the eager operation on local devices. Different from
// `EagerLocalExecute` which blocks and waits for the finishing the op
// execution, this method does not block the thread and could return before the
// eager operation execution finishes. The provided `StatusCallback` will be
// triggered after execution with its status.
void EagerLocalExecuteAsync(EagerOperation* op, TensorHandle** retvals,
                            int* num_retvals, StatusCallback done) {
  if (!op->IsLocal()) {
    done(errors::InvalidArgument(
        "Remote execution is not supported in async EagerLocalExecuteAsync"));
    return;
  }

  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
      op->op_name(), op->eager_func_params().has_value()
                         ? op->eager_func_params().value().step_id.value_or(0)
                         : 0);
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerLocalExecuteAsync: ", op->Name()); },
      tsl::profiler::TraceMeLevel::kInfo);
  EagerContext& ctx = op->EagerContext();

  core::RefCountPtr<KernelAndDevice> kernel;
  Status s = GetOrCreateKernelAndDevice(op, retvals, num_retvals, &kernel);
  if (!s.ok()) {
    done(s);
    return;
  }

  int num_outputs = kernel->num_outputs();
  s = ValidateInputTypeAndPlacement(&ctx, op, kernel);
  if (!s.ok()) {
    done(s);
    return;
  }

  if (ctx.LogDevicePlacement() || VLOG_IS_ON(1)) {
    string msg = strings::StrCat("Executing op ", op->Name(), " in device ",
                                 kernel->device()->name());
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }

  GraphCollector* graph_collector = nullptr;
  if (ctx.ShouldStoreGraphs()) {
    graph_collector = ctx.GetGraphCollector();
  }

  for (int i = 0, end = num_outputs; i < end; ++i) {
    const DataTypeVector& output_dtypes = kernel->output_dtypes();
    retvals[i] = TensorHandle::CreateEmptyLocalHandle(
        /* d= */ ctx.CanonicalDevice(kernel->OutputDevice(i)),
        /* op_device= */ kernel->device(),
        /* resource_device= */ kernel->OutputResourceDevice(i),
        output_dtypes[i], &ctx);
  }

  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  s = op->TensorHandleInputs(&inputs);
  if (!s.ok()) {
    done(s);
    return;
  }
  EagerKernelExecuteAsync(
      &ctx, *inputs, op->eager_func_params(), std::move(kernel),
      graph_collector, op->GetCancellationManager(), retvals, num_outputs,
      [op, num_outputs, retvals, done = std::move(done)](const Status& s) {
        op->Clear();
        // Since the operation failed, we need to Unref any outputs if they were
        // allocated.
        if (!s.ok()) {
          for (int i = 0, end = num_outputs; i < end; ++i) {
            if (retvals[i] != nullptr) {
              retvals[i]->Unref();
              retvals[i] = nullptr;
            }
          }
        }
        done(s);
      });
}
}  // namespace tensorflow
