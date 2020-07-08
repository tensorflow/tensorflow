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
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
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

namespace tensorflow {

namespace {

const string& DeviceNameOrUnspecified(Device* device) {
  static string* unspecified_string = new string("<unspecified>");
  return (device == nullptr) ? *unspecified_string : device->name();
}

const string& DeviceNameOrUnspecified(VariantDevice device) {
  if (VariantDeviceIsCustom(device)) {
    return absl::get<CustomDevice*>(device)->name();
  } else {
    return DeviceNameOrUnspecified(absl::get<Device*>(device));
  }
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
      if (op->Name() == "Identity" || op->Name() == "IdentityN") {
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
    return errors::Internal("Failed copying input tensor from ",
                            handle_device->name(), " to ",
                            expected_input_device->name(), " in order to run ",
                            op->Name(), ": ", status.error_message());
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
  const bool skip_remote_copy =
      ctx->LazyCopyFunctionRemoteInputs() && kernel->IsFunction();
  if (n_inputs > 0) {
    const DataType* input_types = &kernel->input_dtypes()[0];
    TensorHandle* const* handles = &op->Inputs()[0];
    for (int i = 0; i < n_inputs; ++i) {
      TensorHandle* handle = handles[i];
      Device* expected_device = kernel->InputDevice(i);
      auto handle_device_variant = handle->DeviceOrHostCPU(*ctx);
      if (VariantDeviceIsCustom(handle_device_variant)) {
        return errors::Unimplemented(
            "Custom devices and remote execution are not yet supported "
            "together.");
      }
      Device* handle_device = absl::get<Device*>(handle_device_variant);
      const bool maybe_copy =
          !skip_remote_copy || handle->Type() != TensorHandle::REMOTE;
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
  if (TF_PREDICT_FALSE(VariantDeviceIsCustom(tensor_handle->device()))) {
    return errors::Unimplemented(
        "The kernel cache does not work with custom devices.");
  }
  Device* cpu_device = ctx.HostCPU();
  string device_name;
  if (tensor_handle->Type() != TensorHandle::LOCAL) {
    Device* device = absl::get<Device*>(tensor_handle->device());
    device_name = device != nullptr ? device->name() : cpu_device->name();
    *result = (device == nullptr ? cpu_device : device);
  } else if (tensor_handle->dtype == DT_RESOURCE) {
    // Use the resource's actual device because it is the device that will
    // influence partitioning the multi-device function.
    const Tensor* tensor;
    // TODO(fishx): Avoid blocking here.
    TF_RETURN_IF_ERROR(tensor_handle->Tensor(&tensor));
    const ResourceHandle& handle = tensor->flat<ResourceHandle>()(0);
    device_name = handle.device();

    Device* input_device;
    TF_RETURN_IF_ERROR(
        ctx.FindDeviceFromName(device_name.c_str(), &input_device));
    *result = input_device;
  } else if (MTypeFromDType(tensor_handle->dtype) == HOST_MEMORY) {
    *result = cpu_device;
  } else {
    Device* device = absl::get<Device*>(tensor_handle->device());
    device_name = device != nullptr ? device->name() : cpu_device->name();
    *result = (device == nullptr ? cpu_device : device);
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

  // Does node have an explicit request to compile or not?
  Status status = op->Attrs().Get(kXlaMustCompileAttr, compile_with_xla);
  if (status.ok()) {
    DVLOG(2) << "Caller explicitly requested "
             << (*compile_with_xla ? "" : "not ")
             << "to compile with XLA: " << op->DebugString();
    return Status::OK();
  }

  // Does FunctionDef have an explicit request to compile or not?
  const FunctionDef* function_def =
      ctx.pflr()->GetFunctionLibraryDefinition()->Find(op->Name());
  if (function_def == nullptr) {
    return errors::NotFound("Failed to find function '", op->Name(), "'");
  }

  status = GetNodeAttr(AttrSlice(&function_def->attr()), kXlaMustCompileAttr,
                       compile_with_xla);
  if (status.ok()) {
    DVLOG(2) << "Function definition explicitly specifies "
             << (*compile_with_xla ? "" : "not ") << "to compile with XLA";
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

Status GetOrCreateKernelAndDevice(
    EagerOperation* op, TensorHandle** retvals, int* num_retvals,
    core::RefCountPtr<KernelAndDevice>* out_kernel) {
  EagerContext& ctx = op->EagerContext();
  Device* device = absl::get<Device*>(op->Device());

  Fprint128 cache_key = op->MutableAttrs()->CacheKey(op->DeviceName());
  /// Include soft placement policy in cache key since the placement strategy
  // can change and thus affect which kernel is picked.
  cache_key = FingerprintCat128(cache_key, ctx.AllowSoftPlacement());

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
  if (op->is_function()) {
    profiler::TraceMe activity("EagerCopyToDeviceAndAddCacheKey",
                               profiler::TraceMeLevel::kInfo);
    input_dev_ptrs.reserve(op->Inputs().size());
    // When LazyCopyFunctionRemoteInputs is disabled, all inputs need to be on
    // local devices, since we execute a remote function through worker service,
    // which doesn't accept remote inputs.
    for (int i = 0; i < op->Inputs().size(); i++) {
      TensorHandle* input = op->Inputs()[i];
      if (!ctx.LazyCopyFunctionRemoteInputs() &&
          input->Type() == TensorHandle::REMOTE) {
        TensorHandle* handle = nullptr;
        TF_RETURN_IF_ERROR(
            EagerCopyToDevice(input, &ctx, &op->Executor(),
                              device == nullptr ? ctx.HostCPU() : device,
                              /*mirror=*/true, &handle));
        op->UpdateInput(i, handle);
        // Unref handle since it has a ref as an input now
        handle->Unref();
        input = handle;
      }

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
  if (kernel == nullptr) {
    DVLOG(2) << "Creating new kernel for " << op->Name() << " on device "
             << DeviceNameOrUnspecified(op->Device());
    bool run_function_with_flr = false;
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
    }

    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
    if (device == nullptr) {
      PrioritizedDeviceTypeVector supported_devs;
      auto device_type_list = ctx.prioritized_device_type_list();
      TF_RETURN_IF_ERROR(
          SupportedDeviceTypesForNode(*device_type_list, ndef, &supported_devs,
                                      &ctx.HostCPU()->parsed_name()));
      if (supported_devs.empty()) {
        return errors::NotFound(
            "Could not find device for node: ",
            errors::FormatNodeNameForError(ndef.name()), " = ", ndef.op(), "[",
            SummarizeAttrs(ndef), "]", "\nAll kernels registered for op ",
            ndef.op(), ":\n", KernelsRegisteredForOp(ndef.op()));
      }
      TF_RETURN_IF_ERROR(ctx.SelectDevice(op->GetDeviceParsedName(),
                                          supported_devs, DT_INVALID, &device));

      DVLOG(1) << "Placer place op [" << op->Name()
               << "] on device: " << device->name();
      DVLOG(4) << "Available kernels for " << op->Name() << "are "
               << KernelsRegisteredForOp(op->Name());
      op->SetDevice(device);
    }

    FunctionLibraryRuntime* flr =
        device == nullptr ? nullptr : ctx.func_lib(device);
    if (device != nullptr && flr == nullptr) {
      return errors::Unavailable(
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
      if (ctx.LazyCopyFunctionRemoteInputs()) {
        get_op_id = [&ctx]() { return ctx.RemoteMgr()->NextOpId(); };
      }
#endif  // IS_MOBILE_PLATFORM
      kernel.reset(new KernelAndDeviceFunc(
          flr, ctx.pflr(), std::move(input_dev_ptrs),
          std::move(composite_devices),
          std::move(input_resource_variable_dtypes_and_shapes), runner,
          ctx.GetCollectiveExecutorHandle(), ctx.HostCPU(), op->Name(),
          [&ctx](const int64 step_id) { return ctx.CreateRendezvous(step_id); },
          get_op_id));
    } else {
      DVLOG(2) << "Running " << ndef.op() << " using op kernel. "
               << ". Full node_def=" << ndef.DebugString();
      kernel.reset(new KernelAndDeviceOp(
          ctx.GetRendezvous(), ctx.LogMemory(), flr, runner,
          ctx.GetCollectiveExecutorHandle(), ctx.HostCPU()));
    }

    TF_RETURN_IF_ERROR(
        kernel->Init({ctx.LogDevicePlacement()}, ndef, graph_collector));

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
      if (!data::DatasetOpKernel::IsDatasetOp(op_def)) {
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
  TF_RETURN_IF_ERROR(
      GetOrCreateKernelAndDevice(op, retvals, num_retvals, &kernel));

  int num_outputs = kernel->num_outputs();
  TF_RETURN_IF_ERROR(ValidateInputTypeAndPlacement(&ctx, op, kernel));

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

  Status s;
  if (executor.Async()) {
    const DataTypeVector& output_dtypes = kernel->output_dtypes();
    for (int i = 0; i < num_outputs; ++i) {
      retvals[i] = TensorHandle::CreateEmptyLocalHandle(
          /* d= */ ctx.CanonicalDevice(kernel->OutputDevice(i)),
          /* op_device= */ kernel->device(),
          /* resource_device= */ kernel->OutputResourceDevice(i),
          output_dtypes[i], &ctx);
    }
    auto node = absl::make_unique<AsyncExecuteNode>(
        &ctx, op->Inputs(), op->remote_func_params(), std::move(kernel),
        graph_collector, op->GetCancellationManager(),
        absl::Span<TensorHandle*>(retvals, num_outputs));
    // Release the inputs from the eager operation since the AsyncExecuteNode
    // would have taken ownership. This allows the inputs to be forwarded if
    // possible.
    op->Clear();
    // For async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(b/137118203): Consider executing "cheap" kernels inline for
    // performance.
    s = executor.AddOrExecute(std::move(node));
  } else {
    for (int i = 0; i < num_outputs; ++i) {
      retvals[i] = nullptr;
    }
    ExecuteNode node(&ctx, op->Inputs(), op->remote_func_params(), kernel,
                     graph_collector, op->GetCancellationManager(),
                     {retvals, static_cast<size_t>(num_outputs)});
    s = executor.SyncExecute(&node);
    // We release the inputs AFTER executing the operation in sync mode since
    // ExecuteNode does not increment the reference count and thus does not have
    // ownership of the inputs while executing.
    op->Clear();
  }
  // Since the operation failed, we need to Unref any outputs if they were
  // allocated.
  if (!s.ok()) {
    for (int i = 0; i < num_outputs; ++i) {
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
  for (int i = 0; i < op->Inputs().size(); ++i) {
    TensorHandle* handle = op->Inputs()[i];
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
        VariantDeviceName(op->Device()));
  }

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(context_id);

  eager::Operation* remote_op = request->add_queue()->mutable_operation();

  tensorflow::Device* op_device = absl::get<Device*>(op->Device());
  {
    profiler::TraceMe activity("CopyInputToExpectedDevice",
                               profiler::TraceMeLevel::kInfo);
    const bool eagerly_copy_function_remote_inputs =
        !ctx.LazyCopyFunctionRemoteInputs() || !op->is_function();
    for (int i = 0; i < op->Inputs().size(); i++) {
      tensorflow::TensorHandle* input = op->Inputs()[i];
      tensorflow::Device* input_device = absl::get<Device*>(input->device());
      tensorflow::Device* input_device_or_cpu =
          absl::get<Device*>(input->DeviceOrHostCPU(ctx));
      const string* input_device_name = &input_device_or_cpu->name();
      bool serialize_resource_dtype_and_shape = false;
      if (op_device != input_device &&
          // If the expected and actual devices are on the same task, don't
          // explicitly copy, and instead depend on the copy to happen locally
          // when the op is executed on the device.
          !ctx.OnSameTask(op_device, input_device)) {
        if (eagerly_copy_function_remote_inputs ||
            input_device_or_cpu->IsLocal()) {
          tensorflow::Device* remote_cpu_device;
          TF_RETURN_IF_ERROR(
              ctx.CPUDeviceOnTask(op_device, &remote_cpu_device));
          // TODO(b/110044833): It's possible the same tensor gets copied to the
          // remote device repeatedly.
          // Always copy to the remote CPU so that the actual device can be
          // correctly determined after the kernel is selected/instantiated,
          // since the op might have its inputs on host memory.
          TensorHandle* handle = op->Inputs()[i];
          Device* handle_device =
              absl::get<Device*>(handle->DeviceOrHostCPU(ctx));
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
      // For a multi-device function, a remote RunComponentFunction request is
      // not sent through StreamingEnqueueAsync. It could arrive at a remote
      // worker before a remote execution request which produces an input of the
      // component function. So we wait until the remote input is ready before
      // serializing it.
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

  const size_t num_outputs = static_cast<int>(output_dtypes.size());
  if (num_outputs != *num_retvals) {
    return errors::InvalidArgument(
        "num_retvals does not match expected output dtypes");
  }
  *num_retvals = num_outputs;

  const tensorflow::uint64 id = remote_op->id();
  for (int i = 0; i < num_outputs; ++i) {
    // TODO(nareshmodi): Change the callback to instead add the decref to a
    // list of pending decrefs that we can send as a batch with the next
    // execute.

    // The device_ and resource_device_ of this TensorHandle might be
    // incorrect. It is pretty hard to make it correct because for
    // multi-device functions, we don't know the output device until the
    // function is instantiated. Luckily, we don't need to know the correct
    // remote device here. We just need to know that it is remote. If we need
    // to copy this tensor to this process, the remote end will know the
    // correct device of this handle.
    retvals[i] = TensorHandle::CreateUnshapedRemoteHandle(
        id, i, remote_task, output_dtypes[i], op_device, &ctx);
  }

  if (ctx.LazyCopyFunctionRemoteInputs()) {
    // Store the data type and shape of a remote resource variable on the
    // corresponding remote TensorHandle (output of 'VarHandleOp').
    // If the variable is an input of a remote function, the function may need
    // the type and shape during function instantiation. When
    // LazyCopyFunctionRemoteInputs is enabled, we no longer copy the resource
    // handle (contains the type and shape) of the variable to the default
    // function device. Instead, we store the type and shape on eager master
    // and sent them to the default function device along with the
    // EnqueueRequest.
    TF_RETURN_IF_ERROR(
        StoreResourceDtypesAndShapes(*remote_op, output_dtypes, retvals));
  }

  auto& executor = op->Executor();
  DVLOG(4) << "Execute remote eager op: " << op->Name()
           << " (is async?: " << executor.Async() << ").";

  std::unique_ptr<EagerNode> node(new eager::RemoteExecuteNode(
      &op->EagerContext(), std::move(request), op_device,
      ctx.GetContextViewId(), eager_client.get(),
      op->MutableAttrs()->BuildNodeDef(), op->EagerContext().FuncLibDef(),
      op->Inputs(), {retvals, num_outputs}));

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
    for (int i = 0; i < num_outputs; ++i) {
      retvals[i]->Unref();
    }
  }

  return s;
}
#endif  // IS_MOBILE_PLATFORM

Status GetKernelOutputs(std::vector<Tensor>* outputs, int num_outputs,
                        TensorHandle** retvals, EagerContext* ctx,
                        KernelAndDevice* kernel) {
  for (int i = 0; i < num_outputs; ++i) {
    if (retvals[i] == nullptr) {
      retvals[i] = TensorHandle::CreateLocalHandle(
          std::move((*outputs)[i]),
          /* d= */ ctx->CanonicalDevice(kernel->OutputDevice(i)),
          /* op_device= */ kernel->device(),
          /* resource_device= */ kernel->OutputResourceDevice(i), ctx);
    } else {
      if (TF_PREDICT_FALSE(kernel->device() != retvals[i]->op_device())) {
        return errors::Internal(
            "Kernel output tensor handle has a different op device than the "
            "kernel. This should never happen.");
      }
      if (TF_PREDICT_FALSE(ctx->CanonicalDevice(kernel->OutputDevice(i)) !=
                           absl::get<Device*>(retvals[i]->device()))) {
        return errors::Internal(
            "Kernel output tensor handle locates on a different device than "
            "the specified kernel output device. This should never happen.");
      }

      TF_RETURN_IF_ERROR(
          retvals[i]->SetTensor(std::move((*outputs)[i]),
                                ctx->CanonicalDevice(kernel->OutputDevice(i))));
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

  if (VariantDeviceIsCustom(op->Device())) {
    return absl::get<CustomDevice*>(op->Device())
        ->Execute(op, retvals, num_retvals);
  }

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
    absl::Span<TensorHandle*> retvals) {
  profiler::TraceMe activity("EagerKernelExecute",
                             profiler::TraceMeLevel::kInfo);
  std::vector<Tensor> outputs(1);

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
                                 cancellation_manager, remote_func_params));
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
                          kernel.get());
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
  auto send_device = h->DeviceOrHostCPU(*ctx);
  if (VariantDeviceIsCustom(send_device)) {
    return errors::Unimplemented(
        "Copying a TensorHandle from a custom device is not supported.");
  }
  bool sender_is_local = absl::get<Device*>(send_device)->IsLocal();

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
  auto outputs = std::make_shared<std::vector<Tensor>>(1);

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
       kernel_raw = kernel.get(), done = std::move(done)](const Status& s) {
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
                                      kernel_raw));
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
  if (VariantDeviceIsCustom(op->Device())) {
    done(errors::Unimplemented(
        "Custom device is not supported in EagerLocalExecuteAsync."));
    return;
  }
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

  for (int i = 0; i < num_outputs; ++i) {
    retvals[i] = nullptr;
  }

  EagerKernelExecuteAsync(
      &ctx, op->Inputs(), op->remote_func_params(), std::move(kernel),
      graph_collector, op->GetCancellationManager(), retvals, num_outputs,
      [op, num_outputs, retvals, done = std::move(done)](const Status& s) {
        op->Clear();
        // Since the operation failed, we need to Unref any outputs if they were
        // allocated.
        if (!s.ok()) {
          for (int i = 0; i < num_outputs; ++i) {
            if (retvals[i] != nullptr) {
              retvals[i]->Unref();
            }
          }
        }
        done(s);
      });
}
}  // namespace tensorflow
