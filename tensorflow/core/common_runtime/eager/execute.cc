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

#include <cstddef>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.pb.h"
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
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_copy_node.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"

#ifdef INTEL_MKL
#include "tensorflow/core/graph/mkl_graph_util.h"
#endif

namespace tensorflow {

namespace {

const string& DeviceNameOrUnspecified(Device* device) {
  static string* unspecified_string = new string("<unspecified>");
  return (device == nullptr) ? *unspecified_string : device->name();
}

// Returns whether a kernel should be cached.
bool KernelCacheEnabled(const OpDef& op_def) {
  if (data::DatasetOpKernel::IsDatasetOp(&op_def)) {
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
  profiler::TraceMe activity(
      [&] {
        return absl::StrCat("_Send input ", i, " from ", handle_device->name(),
                            " to ", expected_input_device->name());
      },
      profiler::TraceMeLevel::kInfo);
  Status status =
      EagerCopyToDevice(handle, ctx, &op->Executor(), expected_input_device,
                        /* mirror= */ true, &result_handle);
  activity.Stop();
  if (!status.ok()) {
    return Status(
        status.code(),
        absl::StrCat("Failed copying input tensor from ", handle_device->name(),
                     " to ", expected_input_device->name(), " in order to run ",
                     op->Name(), ": ", status.error_message()));
  }

  *result = result_handle;

  return Status::OK();
}

// `op_device_name` the name of the device on which the op will run, if any.
// For functions running using function library runtime, the device can be
// unspecified.
Status ValidateInputTypeAndPlacement(
    EagerContext* ctx, EagerOperation* op,
    const core::RefCountPtr<KernelAndDevice>& kernel) {
  profiler::TraceMe activity("ValidateInputTypeAndPlacement",
                             profiler::TraceMeLevel::kInfo);
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
  return Status::OK();
}

Status GetOutputDTypes(EagerOperation* op, DataTypeVector* output_dtypes) {
  const auto& node_def = op->MutableAttrs()->BuildNodeDef();
  const OpDef* op_def = nullptr;

  const FunctionDef* function_def =
      op->EagerContext().FuncLibDef()->Find(op->Name());
  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(op->Name().c_str(), &op_def));
  }

  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, *op_def, output_dtypes));

  return Status::OK();
}

inline tensorflow::Fprint128 FingerprintCat128(const tensorflow::Fprint128& a,
                                               const tensorflow::Fprint128& b) {
  return {tensorflow::FingerprintCat64(a.low64, b.low64),
          tensorflow::FingerprintCat64(a.high64, b.high64)};
}

inline tensorflow::Fprint128 FingerprintCat128(const tensorflow::Fprint128& a,
                                               const int64 b) {
  auto x = tensorflow::FingerprintCat64(a.low64, b);
  return {x, tensorflow::FingerprintCat64(a.high64, x)};
}

Status GetDeviceForInput(const EagerContext& ctx, TensorHandle* tensor_handle,
                         Device** result) {
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
    const bool use_host_memory =
        is_tpu ? MTypeFromDTypeIntsOnDevice(tensor_handle->dtype)
               : MTypeFromDType(tensor_handle->dtype);
    if (use_host_memory) {
      *result = cpu_device;
    } else {
      device_name = device != nullptr ? device->name() : cpu_device->name();
      *result = (device == nullptr ? cpu_device : device);
    }
  }
  return Status::OK();
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
    *fingerprint = FingerprintCat128(*fingerprint, c);
  } else {
    for (int i = 0; i < shape.dims(); i++) {
      int64 dim = shape.dim_size(i);
      *fingerprint = FingerprintCat128(*fingerprint, dim);
    }
  }
}

Status GetFuncAttr(const EagerOperation* op, const EagerContext& ctx,
                   const char* attr_name, bool* value) {
  Status status = op->Attrs().Get(attr_name, value);
  if (status.ok()) {
    DVLOG(2) << "Caller explicitly specifies "
             << (attr_name ? "=true " : "=false, ") << op->DebugString();
    return Status::OK();
  }

  const FunctionDef* function_def =
      ctx.pflr()->GetFunctionLibraryDefinition()->Find(op->Name());
  if (function_def == nullptr) {
    return errors::NotFound("Failed to find function '", op->Name(), "'");
  }

  status = GetNodeAttr(AttrSlice(&function_def->attr()), attr_name, value);
  if (status.ok()) {
    DVLOG(2) << "Function definition explicitly specifies "
             << (attr_name ? "=true" : "=false");
    return Status::OK();
  }
  return status;
}

Status MustCompileWithXLA(const EagerOperation* op, const EagerContext& ctx,
                          bool* compile_with_xla) {
  if (!op->is_function()) {
    *compile_with_xla = false;
    return Status::OK();
  }

  if (op->remote_func_params().has_value() &&
      op->remote_func_params().value().step_id.has_value()) {
    // If the op is a component of a multi-device function, don't compile it
    // with XLA.
    *compile_with_xla = false;
    return Status::OK();
  }

  Status status = GetFuncAttr(op, ctx, kXlaMustCompileAttr, compile_with_xla);
  if (status.ok()) {
    return Status::OK();
  }

  // No explicit requests. Compile for XLA devices by default.
  if (op->GetDeviceParsedName().type == "TPU" ||
      op->GetDeviceParsedName().type == "XLA_GPU" ||
      op->GetDeviceParsedName().type == "XLA_CPU") {
    DVLOG(2) << "Compiling " << op->Name()
             << " with XLA because it is running on an XLA device "
             << op->GetDeviceParsedName().type;
    *compile_with_xla = true;
  } else {
    *compile_with_xla = false;
  }

  return Status::OK();
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
  return Status::OK();
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
    return Status::OK();
  };
  absl::btree_map<string, int> attr_to_len;
  TF_RETURN_IF_ERROR(FillAttrToLen(opdef.input_arg(), &attr_to_len));
  TF_RETURN_IF_ERROR(FillAttrToLen(opdef.output_arg(), &attr_to_len));
  for (auto& name_len : attr_to_len) {
    absl::StrAppend(&fname, "_", name_len.first, "_", name_len.second);
  }
  *name = fname;
  return Status::OK();
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
        for (size_t i = 0; i < number_attr; i++) {
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
          arg_def->set_type_attr(EscapeOrigName(arg.type_attr()));
        }
      }
    }
    return Status::OK();
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
  return Status::OK();
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
        return Status::OK();
      };
  absl::flat_hash_map<string, DataType> attrs_to_add;
  TF_RETURN_IF_ERROR(FillAttrsToAdd(opdef.input_arg(), &attrs_to_add));
  TF_RETURN_IF_ERROR(FillAttrsToAdd(opdef.output_arg(), &attrs_to_add));
  for (auto& name_type : attrs_to_add) {
    TF_RETURN_IF_ERROR(
        wrapped_op->SetAttrType(name_type.first.data(), name_type.second));
  }
  // TODO(srbs): Rename all original attributes using EscapeOrigName.
  return Status::OK();
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
  return Status::OK();
}

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

#ifdef INTEL_MKL
    if (IsMklEnabled() &&
        absl::StartsWith(op->Name(), mkl_op_registry::kMklOpPrefix)) {
      // All MKL eager ops have `_kernel` private attribute that needs to be set
      // to a fixed label.
      AttrValue attr_kernel;
      attr_kernel.set_s(mkl_op_registry::kMklNameChangeOpLabel);
      (*ndef->mutable_attr()).insert({"_kernel", attr_kernel});
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
  TF_RETURN_IF_ERROR(call_op->SetDeviceName(op->DeviceName().c_str()));
  *wrapped_op = down_cast<EagerOperation*>(call_op.release());
  // Attributes on the elementary eager operation are applied to the call op and
  // to the NodeDef inside the FunctionDef. This allows us to have a single
  // FunctionDef for different attribute values. When the function is
  // instantiated, these attributes get forwarded to the NodeDef. This is done
  // by setting the AttrValue.placeholder field for the NodeDef attrs.
  (*wrapped_op)->AddAttrs(op_attrs);
  return AddMixedTypeListAttrs(*wrapped_op, op_attrs, opdef);
}

Status GetOrCreateKernelAndDevice(
    EagerOperation* op, TensorHandle** retvals, int* num_retvals,
    core::RefCountPtr<KernelAndDevice>* out_kernel) {
  EagerContext& ctx = op->EagerContext();
  Device* device = absl::get<Device*>(op->Device());

  Fprint128 cache_key = op->MutableAttrs()->CacheKey(op->DeviceName());
  /// Include soft placement policy in cache key since the placement strategy
  // can change and thus affect which kernel is picked.
  cache_key = FingerprintCat128(cache_key, ctx.AllowSoftPlacement());
  // The launch-time rendezvous reuse setting is bundled with the kernel, so we
  // need to include it in the cache key.
  cache_key =
      FingerprintCat128(cache_key, ctx.GetReuseRendezvousForFunctions());

  std::vector<Device*> input_dev_ptrs;
  absl::flat_hash_map<string, const std::vector<string>*> composite_devices;
  std::unordered_map<int, DtypeAndPartialTensorShape>
      input_resource_variable_dtypes_and_shapes;
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
  if (op->is_function() || ctx.RunEagerOpAsFunction()) {
    profiler::TraceMe activity("EagerCopyToDeviceAndAddCacheKey",
                               profiler::TraceMeLevel::kInfo);
    input_dev_ptrs.reserve(op->Inputs().size());
    const absl::InlinedVector<TensorHandle*, 4>* inputs;
    TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
    for (int i = 0, end = inputs->size(); i < end; i++) {
      TensorHandle* input = (*inputs)[i];

      // Get device for this input, and add it to 'cache_key'.
      Device* input_device;
      TF_RETURN_IF_ERROR(GetDeviceForInput(ctx, input, &input_device));
      input_dev_ptrs.push_back(input_device);
      CompositeDevice* composite_device = nullptr;
      if (ctx.FindCompositeDeviceFromName(input_device->name(),
                                          &composite_device)
              .ok()) {
        composite_devices[input_device->name()] =
            composite_device->underlying_devices();
      }
      cache_key =
          FingerprintCat128(cache_key, Fingerprint128(input_device->name()));

      // If input is a ResourceHandle, get its resource handle dtypes and shapes
      // and add them to 'cache_key'.
      if (input->dtype == DT_RESOURCE) {
        // We only care about data type and shape for resource variable inputs.
        // But we have no way to tell if input is resource variable (other than
        // looking it up in ResourceMgr, which is slow). So we just get
        // resource_dtypes_and_shapes for all DT_RESOURCE inputs. If
        // resource_dtypes_and_shapes is not empty, take the first element.
        std::vector<DtypeAndPartialTensorShape> resource_dtypes_and_shapes;
        TF_RETURN_IF_ERROR(input->GetResourceHandleDtypesAndShapes(
            &resource_dtypes_and_shapes));
        if (!resource_dtypes_and_shapes.empty()) {
          const DtypeAndPartialTensorShape& dtype_and_shape =
              resource_dtypes_and_shapes.at(0);
          input_resource_variable_dtypes_and_shapes[i] = dtype_and_shape;

          // Add _Arg index, dtype and shape to "cache_key".
          cache_key = FingerprintCat128(cache_key, i);
          DataType dtype = dtype_and_shape.dtype;
          cache_key = FingerprintCat128(cache_key, dtype);
          AppendTensorShapeToFingerprint(dtype_and_shape.shape, &cache_key);
        }
      }
    }
  }

  core::RefCountPtr<KernelAndDevice> kernel = ctx.GetCachedKernel(cache_key);
  AbstractOperationPtr wrapped_op_releaser;
  if (kernel == nullptr) {
    if (ctx.RunEagerOpAsFunction() && !op->is_function()) {
      EagerOperation* wrapped_op = nullptr;
      TF_RETURN_IF_ERROR(WrapInCallOp(op, &wrapped_op));
      DCHECK(wrapped_op);
      DCHECK(wrapped_op->is_function());
      wrapped_op_releaser.reset(wrapped_op);
      op = wrapped_op;
    }
    DVLOG(2) << "Creating new kernel for " << op->Name() << " on device "
             << DeviceNameOrUnspecified(absl::get<Device*>(op->Device()));
    bool run_function_with_flr = false;
    bool function_outputs_on_op_device = false;
    if (op->is_function()) {
      bool compile_with_xla;
      TF_RETURN_IF_ERROR(MustCompileWithXLA(op, ctx, &compile_with_xla));
      if (compile_with_xla) {
        // Note that it is not ideal, but currently correct, to set this
        // attribute after computing the kernel cache key above.
        // Note: If the attribute is already set to true, this is a noop.
        op->MutableAttrs()->Set(kXlaMustCompileAttr, true);
      } else {
        run_function_with_flr = true;
      }
      GetFuncAttr(op, ctx, kOutputsOnOpDevice, &function_outputs_on_op_device)
          .IgnoreError();
    }

    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
    if (device == nullptr) {
      // Here in local execute, set preferred device to be on the local task to
      // avoid placing op on a remote device with higher priority.
      const DeviceNameUtils::ParsedName& preferred_device =
          DeviceNameUtils::HasSomeDetails(op->GetDeviceParsedName())
              ? op->GetDeviceParsedName()
              : DeviceNameUtils::AddressSpace(ctx.HostCPUParsedName());
      TF_RETURN_IF_ERROR(ctx.SelectDevice(preferred_device, ndef, &device));

      DVLOG(1) << "Placer place op [" << op->Name()
               << "] on device: " << device->name();
      DVLOG(4) << "Available kernels for " << op->Name() << " are"
               << KernelsRegisteredForOp(op->Name());
      op->SetDevice(device);
    }

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
      DVLOG(2) << "Running " << ndef.op() << " using multi-device function. "
               << "Full node_def=" << ndef.DebugString();
      std::function<int64()> get_op_id = nullptr;
#if !defined(IS_MOBILE_PLATFORM)
      get_op_id = [&ctx]() { return ctx.RemoteMgr()->NextOpId(); };
#endif  // IS_MOBILE_PLATFORM
      kernel.reset(new KernelAndDeviceFunc(
          flr, ctx.pflr(), std::move(input_dev_ptrs),
          std::move(composite_devices),
          std::move(input_resource_variable_dtypes_and_shapes), runner,
          ctx.GetCollectiveExecutorHandle(), ctx.HostCPU(), op->Name(),
          function_outputs_on_op_device, ctx.RendezvousCreator(), get_op_id));
    } else {
      DVLOG(2) << "Running " << ndef.op() << " using op kernel. "
               << ". Full node_def=" << ndef.DebugString();
      kernel.reset(new KernelAndDeviceOp(
          ctx.GetRendezvous(), ctx.LogMemory(), flr, runner,
          ctx.GetCollectiveExecutorHandle(), ctx.HostCPU()));
    }

    TF_RETURN_IF_ERROR(
        kernel->Init(ctx.LogDevicePlacement(), ndef, graph_collector));

    if (op->is_function()) {
      ctx.AddKernelToCache(cache_key, kernel.get());
    } else {
      // Exclude tf.data op kernels from being cached. The reason for this is
      // that tf.data op kernels that accept a user-defined function will have a
      // unique cache key every time they are executed (because the user-defined
      // function is traced every time). Caching such kernels provides no
      // benefit and in some cases results in linear memory growth of use
      // programs that build input pipeline graphs in a loop.
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(OpDefForOp(op->Name().data(), &op_def));
      if (KernelCacheEnabled(*op_def)) {
        ctx.AddKernelToCache(cache_key, kernel.get());
      }
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
  return Status::OK();
}

Status CreateUnshapedOutput(
    const KernelAndDevice& kernel, const int output_num, Device* output_device,
    const DataType& output_dtype,
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
    EagerContext* ctx, TensorHandle** output) {
#if defined(IS_MOBILE_PLATFORM)
  return errors::Unimplemented(
      "Remote outputs are not available on mobile devices.");
#else  // !IS_MOBILE_PLATFORM
  int64 op_id;
  if (remote_func_params.has_value()) {
    op_id = remote_func_params.value().op_id;
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
  return Status::OK();
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
  absl::optional<EagerRemoteFunctionParams> remote_func_params =
      op->remote_func_params();
  if (kernel->IsCrossProcess() && !remote_func_params.has_value()) {
    // Create an eager op id for a cross-process function if not exist.
#if defined(IS_MOBILE_PLATFORM)
    return errors::Unimplemented(
        "Cross-process functions are not supported on mobile devices.");
#else  // !IS_MOBILE_PLATFORM
    const int64 op_id = ctx.RemoteMgr()->NextOpId();
    remote_func_params =
        EagerRemoteFunctionParams{op_id, /*step_id=*/absl::nullopt};
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
                                 remote_func_params, &ctx, &retvals[i]));
      }
    }
    const absl::InlinedVector<TensorHandle*, 4>* inputs;
    TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));
    auto node = absl::make_unique<AsyncExecuteNode>(
        &ctx, *inputs, remote_func_params, std::move(kernel), graph_collector,
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
    ExecuteNode node(&ctx, *inputs, remote_func_params, kernel, graph_collector,
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
  ScopedMemoryDebugAnnotation op_annotation(
      op->op_name(), op->remote_func_params().has_value()
                         ? op->remote_func_params().value().step_id.value_or(0)
                         : 0);
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerLocalExecute: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);
  EagerContext& ctx = op->EagerContext();
  auto& executor = op->Executor();
  TF_RETURN_IF_ERROR(executor.status());

  core::RefCountPtr<KernelAndDevice> kernel;
  auto status = GetOrCreateKernelAndDevice(op, retvals, num_retvals, &kernel);

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
      }
    }
  }

  return s;
}

// Run a Pack op to pack the tensors pointed by a packed input TensorHandle if
// the op is a primitive op.
Status MaybePackInputTensor(EagerOperation* op) {
  if (op->is_function()) {
    // Functions could take packed TensorHandles as inputs.
    return Status::OK();
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
  return Status::OK();
}

#if !defined(IS_MOBILE_PLATFORM)
void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
  EagerContext& ctx = op->EagerContext();

  remote_op->set_id(ctx.RemoteMgr()->NextOpId());
  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMapWithoutDefaults(remote_op->mutable_attrs());
  remote_op->set_device(absl::get<Device*>(op->Device())->name());
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
  return Status::OK();
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

  tensorflow::Device* op_device = absl::get<Device*>(op->Device());
  {
    profiler::TraceMe activity("CopyInputToExpectedDevice",
                               profiler::TraceMeLevel::kInfo);
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
      auto* input_handle = remote_op->add_op_inputs()->mutable_remote_handle();
      // For a remote component function, a function execution request and an
      // input generation request may come from different workers. We need to
      // guarantee that the input generation request is processed before the
      // function execution request, so wait until the remote input is ready
      // before sending it to the multi-device function device.
      const bool wait_until_ready = op->is_function();
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
  DVLOG(4) << "Execute remote eager op: " << op->Name()
           << " (is async?: " << executor.Async() << ").";

  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  TF_RETURN_IF_ERROR(op->TensorHandleInputs(&inputs));

  std::unique_ptr<EagerNode> node(new eager::RemoteExecuteNode(
      &op->EagerContext(), std::move(request), op_device,
      ctx.GetContextViewId(), eager_client.get(), op->GetCancellationManager(),
      op->MutableAttrs()->BuildNodeDef(), op->EagerContext().FuncLibDef(),
      *inputs, {retvals, num_outputs}));

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
    }
  }

  return s;
}
#endif  // IS_MOBILE_PLATFORM

Status GetKernelOutputs(
    std::vector<EagerKernelRet>* outputs, int num_outputs,
    TensorHandle** retvals, EagerContext* ctx, KernelAndDevice* kernel,
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params) {
  for (int i = 0, end = num_outputs; i < end; ++i) {
    if (retvals[i] == nullptr) {
      EagerKernelRet& ret = (*outputs)[i];
      Device* output_device = ctx->CanonicalDevice(kernel->OutputDevice(i));
      if (ret.index() == 0) {
        retvals[i] = TensorHandle::CreateLocalHandle(
            std::move(absl::get<Tensor>(ret)),
            /* d= */ output_device,
            /* op_device= */ kernel->device(),
            /* resource_device= */ kernel->OutputResourceDevice(i), ctx);
      } else {
        const DataTypeVector& output_dtypes = kernel->output_dtypes();
        TF_RETURN_IF_ERROR(
            CreateUnshapedOutput(*kernel, i, output_device, output_dtypes[i],
                                 remote_func_params, ctx, &retvals[i]));
#if !defined(IS_MOBILE_PLATFORM)
        TF_RETURN_IF_ERROR(
            retvals[i]->SetRemoteShape(absl::get<TensorShape>(ret),
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
            std::move(absl::get<Tensor>(ret)),
            ctx->CanonicalDevice(kernel->OutputDevice(i))));
      } else {
#if defined(IS_MOBILE_PLATFORM)
        return errors::Unimplemented(
            "Remote outputs are not available on mobile devices.");
#else  // !IS_MOBILE_PLATFORM
        TF_RETURN_IF_ERROR(retvals[i]->SetRemoteShape(
            absl::get<TensorShape>(ret), retvals[i]->device(),
            ctx->GetContextViewId()));
#endif  // !IS_MOBILE_PLATFORM
      }
    }
  }
  return Status::OK();
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

Status EagerExecute(EagerOperation* op, TensorHandle** retvals,
                    int* num_retvals) {
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerExecute: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);

  if (!op->Executor().Async()) {
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
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
    const core::RefCountPtr<KernelAndDevice>& kernel,
    GraphCollector* graph_collector, CancellationManager* cancellation_manager,
    absl::Span<TensorHandle*> retvals,
    const absl::optional<ManagedStackTrace>& stack_trace) {
  profiler::TraceMe activity("EagerKernelExecute",
                             profiler::TraceMeLevel::kInfo);
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
  TF_RETURN_IF_ERROR(kernel->Run(container, inputs, &outputs,
                                 cancellation_manager, remote_func_params,
                                 stack_trace));
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
                          kernel.get(), remote_func_params);
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
    return Status::OK();
  }

  bool async = executor->Async();
  if (mirror) {
    h->Ref();
    *result = h;

    if (h->HasLocalMirror(d)) {
      return Status::OK();
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
          return Status::OK();
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
          return Status::OK();
        }

        Status s = h->AddEmptyLocalMirror(d);
        if (!s.ok()) {
          // If a mirror was added since we called HasLocalMirror then just
          // return since another thread has already added the mirror.
          if (s.code() == error::Code::ALREADY_EXISTS) {
            return Status::OK();
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
          return Status::OK();
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
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
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

  kernel->Ref();  // Ownership of reference is transferred to the callback
  kernel->RunAsync(
      ctx->StepContainer(), *inputs, outputs.get(), cancellation_manager,
      remote_func_params,
      [retvals, inputs, outputs, num_outputs, ctx, graph_collector,
       remote_func_params, kernel_raw = kernel.get(),
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
                                      kernel_raw, remote_func_params));
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

  ScopedMemoryDebugAnnotation op_annotation(
      op->op_name(), op->remote_func_params().has_value()
                         ? op->remote_func_params().value().step_id.value_or(0)
                         : 0);
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerLocalExecuteAsync: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);
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
      &ctx, *inputs, op->remote_func_params(), std::move(kernel),
      graph_collector, op->GetCancellationManager(), retvals, num_outputs,
      [op, num_outputs, retvals, done = std::move(done)](const Status& s) {
        op->Clear();
        // Since the operation failed, we need to Unref any outputs if they were
        // allocated.
        if (!s.ok()) {
          for (int i = 0, end = num_outputs; i < end; ++i) {
            if (retvals[i] != nullptr) {
              retvals[i]->Unref();
            }
          }
        }
        done(s);
      });
}
}  // namespace tensorflow
