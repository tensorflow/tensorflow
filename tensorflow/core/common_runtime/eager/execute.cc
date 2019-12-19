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

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
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
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"

namespace tensorflow {

namespace {

// Using absl::StrJoin with lambda does not work in tf-lite builds.
std::vector<string> DevicesToString(const std::vector<Device*> devices) {
  std::vector<string> v;
  v.reserve(devices.size());
  for (Device* d : devices) {
    v.push_back(d->name());
  }
  return v;
}

const string& DeviceNameOrUnspecified(Device* device) {
  static string* unspecified_string = new string("<unspecified>");
  return (device == nullptr) ? *unspecified_string : device->name();
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
                        ctx->MirrorTensors(), &result_handle);
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
  for (int i = 0; i < n_inputs; ++i) {
    TensorHandle* handle = op->Inputs()[i];
    Device* expected_device = kernel->InputDevice(i);
    Device* handle_device = handle->DeviceOrHostCPU(ctx);
    const bool maybe_copy = !skip_remote_copy || !handle->IsRemote();
    // If the input is already on the right device, then nothing to do.
    if (expected_device != handle_device && maybe_copy) {
      TF_RETURN_IF_ERROR(CopyInputToExpectedDevice(ctx, op, kernel->device(),
                                                   handle, i, handle_device,
                                                   expected_device, &handle));
      op->UpdateInput(i, handle);
      // Unref handle since it has a ref as an input now
      handle->Unref();
    }
    if (handle->dtype != kernel->input_type(i)) {
      return errors::InvalidArgument(
          "cannot compute ", op->Name(), " as input #", i, "(zero-based)",
          " was expected to be a ", DataTypeString(kernel->input_type(i)),
          " tensor but is a ", DataTypeString(handle->dtype), " tensor");
    }
  }
  return Status::OK();
}

Status SelectDevice(EagerOperation* op, const NodeDef& ndef, EagerContext* ctx,
                    Device** device) {
  std::vector<Device*> final_devices;
  PrioritizedDeviceTypeVector supported_devs;
  TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
      ctx->prioritized_device_type_list(), ndef, &supported_devs,
      &ctx->HostCPU()->parsed_name()));
  if (supported_devs.empty()) {
    return errors::NotFound("Could not find valid device for node.\nNode:",
                            FormatNodeDefForError(ndef),
                            "\nAll kernels registered for op ", ndef.op(),
                            " :\n", KernelsRegisteredForOp(ndef.op()));
  }

  if (DeviceNameUtils::HasSomeDetails(op->GetDeviceParsedName())) {
    ctx->pflr()->device_set()->FindMatchingDevices(op->GetDeviceParsedName(),
                                                   &final_devices);

    if (!final_devices.empty()) {
      final_devices = ColocationGraph::FilterSupportedDevices(
          final_devices, supported_devs, /*default_device=*/nullptr);
    }

    if (final_devices.empty() && ctx->AllowSoftPlacement()) {
      DeviceNameUtils::ParsedName soft_device_name = op->GetDeviceParsedName();
      soft_device_name.type.clear();
      soft_device_name.has_type = false;
      soft_device_name.has_id = false;
      // TODO(fishx): Soft placement logic picks up another task if the
      // requested does not exist.
      ctx->pflr()->device_set()->FindMatchingDevices(soft_device_name,
                                                     &final_devices);
      if (!final_devices.empty()) {
        final_devices = ColocationGraph::FilterSupportedDevices(
            final_devices, supported_devs, /*default_device=*/nullptr);
      }
    }
    if (final_devices.empty()) {
      return errors::InvalidArgument(
          "Could not satisfy device specification '", op->GetDeviceParsedName(),
          "'. All available devices [",
          absl::StrJoin(DevicesToString(ctx->pflr()->device_set()->devices()),
                        ", "),
          "]. Eager operation: ", op->DebugString());
    }
  } else {
    // TODO(fishx): Allow setting default device in eager context.
    final_devices = ColocationGraph::FilterSupportedDevices(
        ctx->pflr()->device_set()->devices(), supported_devs,
        /*default_device=*/nullptr);
    if (final_devices.empty()) {
      return errors::InvalidArgument(
          "No OpKernel registered to suppport this eager operation:",
          op->DebugString());
    }
  }

  DVLOG(1) << "Placer place op [" << op->Name()
           << "] on device: " << final_devices[0]->name();
  DVLOG(4) << "Available kernels for " << op->Name() << "are "
           << KernelsRegisteredForOp(op->Name());
  op->SetDevice(final_devices[0]);
  *device = final_devices[0];
  return Status::OK();
}

Status GetOutputDTypes(EagerOperation* op, DataTypeVector* output_dtypes) {
  const auto& node_def = op->MutableAttrs()->BuildNodeDef();
  const OpDef* op_def = nullptr;

  const FunctionDef* function_def =
      op->EagerContext()->FuncLibDef()->Find(op->Name());
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

Status GetDeviceForInput(const EagerContext* ctx, TensorHandle* tensor_handle,
                         Device** result) {
  Device* cpu_device = ctx->HostCPU();
  string device_name;
  if (tensor_handle->IsRemote()) {
    Device* device = tensor_handle->device();
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
        ctx->FindDeviceFromName(device_name.c_str(), &input_device));
    *result = input_device;
  } else if (MTypeFromDType(tensor_handle->dtype) == HOST_MEMORY) {
    *result = cpu_device;
  } else {
    Device* device = tensor_handle->device();
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

Status ShouldCompileWithXLA(const EagerOperation* op, const EagerContext* ctx,
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
  Status status = op->Attrs().Get(kXlaCompileAttr, compile_with_xla);
  if (status.ok()) {
    DVLOG(2) << "Caller explicitly requested "
             << (*compile_with_xla ? "" : "not ")
             << "to compile with XLA: " << op->DebugString();
    return Status::OK();
  }

  // Does FunctionDef have an explicit request to compile or not?
  const FunctionDef* function_def =
      ctx->pflr()->GetFunctionLibraryDefinition()->Find(op->Name());
  if (function_def == nullptr) {
    return errors::NotFound("Failed to find function '", op->Name(), "'");
  }

  status = GetNodeAttr(AttrSlice(&function_def->attr()), kXlaCompileAttr,
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
  MEMDEBUG_CACHE_OP(op->op_name());
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerLocalExecute: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);
  EagerContext* ctx = op->EagerContext();
  auto& executor = op->Executor();
  TF_RETURN_IF_ERROR(executor.status());
  Device* device = op->Device();

  Fprint128 cache_key = op->MutableAttrs()->CacheKey(op->GetDeviceName());

  std::vector<Device*> input_dev_ptrs;
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
      if (!ctx->LazyCopyFunctionRemoteInputs() && input->IsRemote()) {
        TensorHandle* handle = nullptr;
        TF_RETURN_IF_ERROR(EagerCopyToDevice(
            input, ctx, &executor, device == nullptr ? ctx->HostCPU() : device,
            ctx->MirrorTensors(), &handle));
        op->UpdateInput(i, handle);
        // Unref handle since it has a ref as an input now
        handle->Unref();
        input = handle;
      }

      // Get device for this input, and add it to 'cache_key'.
      Device* input_device;
      TF_RETURN_IF_ERROR(GetDeviceForInput(ctx, input, &input_device));
      input_dev_ptrs.push_back(input_device);
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

  core::RefCountPtr<KernelAndDevice> kernel = ctx->GetCachedKernel(cache_key);
  if (kernel == nullptr) {
    DVLOG(2) << "Creating new kernel for " << op->Name() << " on device "
             << DeviceNameOrUnspecified(op->Device());
    bool run_function_with_flr = false;
    if (op->is_function()) {
      bool compile_with_xla;
      TF_RETURN_IF_ERROR(ShouldCompileWithXLA(op, ctx, &compile_with_xla));
      if (compile_with_xla) {
        // Note that it is not ideal, but currently correct, to set this
        // attribute after computing the kernel cache key above.
        // Note: If the attribute is already set to true, this is a noop.
        op->MutableAttrs()->Set(kXlaCompileAttr, true);
      } else {
        run_function_with_flr = true;
      }
    }

    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
    if (device == nullptr) {
      TF_RETURN_IF_ERROR(SelectDevice(op, ndef, ctx, &device));
    }
    if (ctx->LogDevicePlacement() || VLOG_IS_ON(1)) {
      string msg = strings::StrCat("Executing op ", ndef.op(), " in device ",
                                   DeviceNameOrUnspecified(device));
      if (!logging::LogToListeners(msg)) {
        LOG(INFO) << msg;
      }
    }

    FunctionLibraryRuntime* flr =
        device == nullptr ? nullptr : ctx->func_lib(device);
    if (device != nullptr && flr == nullptr) {
      return errors::Unavailable(
          "Unable to find a FunctionLibraryRuntime corresponding to device ",
          device->name());
    }
    auto runner = (flr != nullptr && flr->runner() != nullptr) ? flr->runner()
                                                               : ctx->runner();
    GraphCollector* graph_collector = nullptr;
    if (ctx->ShouldStoreGraphs()) {
      graph_collector = ctx->GetGraphCollector();
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
      if (ctx->LazyCopyFunctionRemoteInputs()) {
        get_op_id = [ctx]() { return ctx->RemoteMgr()->NextOpId(); };
      }
#endif  // IS_MOBILE_PLATFORM
      kernel.reset(new KernelAndDeviceFunc(
          flr, ctx->pflr(), std::move(input_dev_ptrs),
          std::move(input_resource_variable_dtypes_and_shapes), runner,
          ctx->GetCollectiveExecutorHandle(), ctx->HostCPU(), op->Name(),
          [ctx](const int64 step_id) { return ctx->CreateRendezvous(step_id); },
          get_op_id));
    } else {
      DVLOG(2) << "Running " << ndef.op() << " using op kernel. "
               << ". Full node_def=" << ndef.DebugString();
      kernel.reset(new KernelAndDeviceOp(
          ctx->GetRendezvous(), ctx->LogMemory(), flr, runner,
          ctx->GetCollectiveExecutorHandle(), ctx->HostCPU()));
    }

    TF_RETURN_IF_ERROR(kernel->Init(ndef, graph_collector));

    if (op->is_function()) {
      ctx->AddKernelToCache(cache_key, kernel.get());
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
        ctx->AddKernelToCache(cache_key, kernel.get());
      }
    }
  }
  const DataTypeVector& output_dtypes = kernel->output_dtypes();
  const size_t num_outputs = static_cast<int>(output_dtypes.size());
  if (num_outputs > *num_retvals) {
    return errors::InvalidArgument("Expecting ", num_outputs,
                                   " outputs, but *num_retvals is ",
                                   *num_retvals);
  }
  *num_retvals = num_outputs;
  TF_RETURN_IF_ERROR(ValidateInputTypeAndPlacement(ctx, op, kernel));

  GraphCollector* graph_collector = nullptr;
  if (ctx->ShouldStoreGraphs()) {
    graph_collector = ctx->GetGraphCollector();
  }

  const bool async = executor.Async();
  for (int i = 0; i < num_outputs; ++i) {
    TF_RETURN_IF_ERROR(TensorHandle::CreateEmptyLocalHandle(
        async,
        /* d= */ ctx->CanonicalDevice(kernel->OutputDevice(i)),
        /* op_device= */ kernel->device(),
        /* resource_device= */ kernel->OutputResourceDevice(i),
        output_dtypes[i], ctx, &retvals[i]));
  }

  Status s;
  if (async) {
    auto node = absl::make_unique<ExecuteNode>(
        ctx, op->Inputs(), op->remote_func_params(), std::move(kernel),
        graph_collector, output_dtypes, op->GetCancellationManager(),
        executor.Async(), absl::Span<TensorHandle*>(retvals, num_outputs));
    // For async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(b/137118203): Consider executing "cheap" kernels inline for
    // performance.
    s = executor.AddOrExecute(std::move(node));
  } else {
    ExecuteNode node(ctx, op->Inputs(), op->remote_func_params(),
                     std::move(kernel), graph_collector, output_dtypes,
                     op->GetCancellationManager(), executor.Async(),
                     {retvals, num_outputs});
    s = executor.SyncExecute(&node);
  }
  // Since the operation failed, we need to Unref any outputs that were
  // allocated.
  if (!s.ok()) {
    for (int i = 0; i < num_outputs; ++i) {
      retvals[i]->Unref();
    }
  }

  return s;
}

#if !defined(IS_MOBILE_PLATFORM)
void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
  EagerContext* ctx = op->EagerContext();

  remote_op->set_id(ctx->RemoteMgr()->NextOpId());
  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMapWithoutDefaults(remote_op->mutable_attrs());
  remote_op->set_device(op->Device()->name());
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
  EagerContext* ctx = op->EagerContext();

  // TODO(fishx): Remove following code when lazy tensor copy is ready.
  if (op->Device() == nullptr) {
    tensorflow::Device* device = nullptr;
    string device_name = op->GetDeviceName();
    TF_RETURN_IF_ERROR(ctx->FindDeviceFromName(device_name.c_str(), &device));
    op->SetDevice(device);
  }

  core::RefCountPtr<eager::EagerClient> eager_client;
  uint64 context_id = ctx->GetContextId();
  TF_RETURN_IF_ERROR(ctx->GetClient(op->GetDeviceParsedName(), &eager_client));
  string remote_task;
  if (!DeviceNameUtils::GetTaskName(op->GetDeviceParsedName(), &remote_task)) {
    return errors::InvalidArgument(
        "Unable to find remote task corresponding to device ",
        op->Device()->name());
  }

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(context_id);

  eager::Operation* remote_op = request->add_queue()->mutable_operation();

  {
    profiler::TraceMe activity("CopyInputToExpectedDevice",
                               profiler::TraceMeLevel::kInfo);
    const bool eagerly_copy_function_remote_inputs =
        !ctx->LazyCopyFunctionRemoteInputs() || !op->is_function();
    for (int i = 0; i < op->Inputs().size(); i++) {
      tensorflow::TensorHandle* input = op->Inputs()[i];
      tensorflow::Device* input_device = input->device();
      const string* input_device_name = &input->DeviceOrHostCPU(ctx)->name();
      bool serialize_resource_dtype_and_shape = false;
      if (op->Device() != input_device &&
          // If the expected and actual devices are on the same task, don't
          // explicitly copy, and instead depend on the copy to happen locally
          // when the op is executed on the device.
          !ctx->OnSameTask(op->Device(), input_device)) {
        if (eagerly_copy_function_remote_inputs ||
            input->DeviceOrHostCPU(ctx)->IsLocal()) {
          tensorflow::Device* remote_cpu_device;
          TF_RETURN_IF_ERROR(
              ctx->CPUDeviceOnTask(op->Device(), &remote_cpu_device));
          // TODO(b/110044833): It's possible the same tensor gets copied to the
          // remote device repeatedly.
          // Always copy to the remote CPU so that the actual device can be
          // correctly determined after the kernel is selected/instantiated,
          // since the op might have its inputs on host memory.
          TensorHandle* handle = op->Inputs()[i];
          Device* handle_device = handle->DeviceOrHostCPU(ctx);
          // If the input is already on the right device, then nothing to do.
          if (remote_cpu_device != handle_device) {
            TF_RETURN_IF_ERROR(CopyInputToExpectedDevice(
                ctx, op, op->Device(), handle, i, handle_device,
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
              (!input->HasResourceShapeMirror(op->Device()));
        }
      }
      auto* input_handle = remote_op->add_inputs();
      TF_RETURN_IF_ERROR(ctx->RemoteMgr()->SerializeRemoteTensorHandle(
          input, input_handle, input_device, *input_device_name,
          serialize_resource_dtype_and_shape));
      if (!input_handle->resource_dtypes_and_shapes().empty()) {
        auto tensor_handle_data =
            absl::make_unique<UnshapedRemoteTensorHandleData>(
                input_handle->op_id(), input_handle->output_num(), remote_task,
                context_id, ctx);
        TF_RETURN_IF_ERROR(input->AddResourceShapeMirror(
            std::move(tensor_handle_data), op->Device()));
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

  tensorflow::Device* op_device = op->Device();
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
    Status status = TensorHandle::CreateUnshapedRemoteHandle(
        id, i, remote_task, context_id, output_dtypes[i], op_device, ctx,
        &retvals[i]);
    if (!status.ok()) {
      for (int j = 0; j < i; ++j) {
        retvals[j]->Poison(errors::Internal(
            "Failed to construct unshaped remote tensor handle at index ", i,
            " for op ", op->Name()));
      }
      return status;
    }
  }

  if (ctx->LazyCopyFunctionRemoteInputs()) {
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
      std::move(request), op_device, eager_client.get(),
      op->MutableAttrs()->BuildNodeDef(), op->EagerContext()->FuncLibDef(),
      op->Inputs(), {retvals, num_outputs}));
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

// These ops are not pinnable since they generate data. It can be slower to
// generate and then copy the data instead of just generating the data on the
// device directly.
bool IsPinnableOp(const string& op_type) {
  static const gtl::FlatSet<string>* unpinnable_ops = new gtl::FlatSet<string>({
      "RandomUniform",
      "RandomUniformInt",
      "RandomStandardNormal",
      "StatelessRandomUniform",
      "StatelessRandomUniformInt",
      "StatelessRandomNormal",
  });

  // XRT ops refer to per-device handles that are not safe to move between
  // devices.
  return unpinnable_ops->find(op_type) == unpinnable_ops->end() &&
         !absl::StartsWith(op_type, "XRT");
}

// The Op device may be updated if:
// - A resource touching input is specified: all resource-touching ops run in
// the device the resource is, regardless of anything else that has been
// specified. This is identical to the graph mode behavior.
//
// - All op inputs are on the CPU, small (<64 elements) and integers
// (int32/int64). This can be disabled by setting the environment variable
// "TF_EAGER_ENABLE_SMALL_TENSOR_CPU_PINNING" to "0" or "false".
Status MaybeUpdateOpDevice(EagerOperation* op) {
  const auto& exempt_ops = InputColocationExemptionRegistry::Global()->Get();
  if (op->is_function() || exempt_ops.find(op->Name()) != exempt_ops.end()) {
    // Don't update the device of direct function calls.
    // Particularly, if the user did not explicitly request any device for this
    // function, picking a device would result in this device being the default
    // for nodes inside the function. This is undesirable for multi-device
    // functions since the not-explicitly-placed nodes inside the body will all
    // end up on this default device.
    return Status::OK();
  }
  EagerContext* ctx = op->EagerContext();
  bool all_inputs_eligible_for_cpu_pinning =
      ctx->PinSmallOpsToCPU() && !op->is_function() && IsPinnableOp(op->Name());
  Device* op_device = op->Device() == nullptr ? ctx->HostCPU() : op->Device();
  for (int i = 0; i < op->Inputs().size(); ++i) {
    TensorHandle* tensor_handle = op->Inputs()[i];
    if (tensor_handle->dtype == DT_RESOURCE) {
      Device* resource_device = tensor_handle->resource_device();
      DVLOG(2) << "for op " << op->Name() << " input " << i << " "
               << DataTypeString(tensor_handle->dtype)
               << " input device = " << resource_device->name()
               << ", op device = " << op_device->name();
      // We check for `op->Device() == nullptr` because it can be later
      // interpreted as unspecified device and a different device can
      // be selected based on device priority. If any input to an op
      // is a resource we must pin it to prevent different device selection.
      // TODO(iga): null device can mean "unspecified" or "CPU". Clean this up.
      if (resource_device != op_device || op->Device() == nullptr) {
        DVLOG(1) << (resource_device != op_device ? "Changing " : "Setting ")
                 << "device of operation " << op->Name() << " to "
                 << resource_device->name() << " because input #" << i
                 << " is a resource in this device.";
        op->SetDevice(resource_device);
      }
      all_inputs_eligible_for_cpu_pinning = false;
      // No point in looking at other inputs. If there are other resources,
      // they must have the same device and we already declared the op to be
      // ineligible for CPU pinning.
      break;
    } else if (all_inputs_eligible_for_cpu_pinning) {
      Device* input_device = tensor_handle->DeviceOrHostCPU(ctx);
      DVLOG(2) << "for op " << op->Name() << " input " << i << " "
               << DataTypeString(tensor_handle->dtype)
               << " input device = " << input_device->name()
               << ", op device = " << op_device->name();

      // Input is on CPU.
      if (input_device != ctx->HostCPU()) {
        all_inputs_eligible_for_cpu_pinning = false;
        continue;
      }

      if (tensor_handle->dtype != DataType::DT_INT32 &&
          tensor_handle->dtype != DataType::DT_INT64) {
        all_inputs_eligible_for_cpu_pinning = false;
        continue;
      }

      int64 num_elements;
      TF_RETURN_IF_ERROR(tensor_handle->NumElements(&num_elements));
      if (num_elements > 64) {
        all_inputs_eligible_for_cpu_pinning = false;
      }
    }
  }

  // Ops without inputs are usually ops that generate a tensor in some way and
  // usually require being present on whatever device they are scheduled on
  // - for e.g. VarHandleOp or _Recv).
  // TODO(nareshmodi): Is it possible there is no int32/int64 CPU kernel for
  // an op, but there is a GPU kernel?
  if (!op->Inputs().empty() && all_inputs_eligible_for_cpu_pinning) {
    DVLOG(1) << "Forcing op " << op->Name()
             << " to be on the CPU since all input tensors have an "
                "int32/int64 dtype, and are small (less than 64 elements).";
    op->SetDevice(ctx->HostCPU());
  }

  return Status::OK();
}
}  // namespace

Status EagerExecute(EagerOperation* op, TensorHandle** retvals,
                    int* num_retvals) {
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerExecute: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(MaybeUpdateOpDevice(op));

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
    return EagerLocalExecute(op, retvals, num_retvals);
  }

  if (op->EagerContext()->LogDevicePlacement() || VLOG_IS_ON(1)) {
    string msg = strings::StrCat(
        "Executing op ", op->Name(), " on task ",
        DeviceNameUtils::ParsedNameToString(op->GetDeviceParsedName()));
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
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
    EagerContext* ctx, const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
    const core::RefCountPtr<KernelAndDevice>& kernel,
    GraphCollector* graph_collector, CancellationManager* cancellation_manager,
    absl::Span<TensorHandle*> retvals) {
  profiler::TraceMe activity("EagerKernelExecute",
                             profiler::TraceMeLevel::kInfo);
  std::vector<Tensor> outputs(1);

  ExecuteNodeArgs inputs(op_inputs.size());
  TF_RETURN_IF_ERROR(inputs.Init(ctx, op_inputs));
  // TODO(apassos) figure out how to record stats for ops which are a part of
  // functions.
  // TODO(b/111859745): When we support recovering from kernel/device errors, we
  // would need to call XlaDevice::EnsureDeviceContextOk() before using an XLA
  // device. We don't call it now because it is an unneeded overhead (it
  // acquires a lock) and we can't recover from errors anyway.
  ScopedStepContainer* container = ctx->StepContainer();
  if (container == nullptr) {
    TF_RETURN_IF_ERROR(kernel->Run(inputs, &outputs, cancellation_manager,
                                   remote_func_params));
  } else {
    TF_RETURN_IF_ERROR(kernel->Run(container, inputs, &outputs,
                                   cancellation_manager, remote_func_params));
  }
  if (graph_collector != nullptr) {
    mutex_lock ml(*ctx->MetadataMu());
    {
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
        *function_graphs->mutable_pre_optimization_graph() =
            collector->raw_graph;
        for (const auto& graph : collector->partitioned_graphs) {
          *function_graphs->add_partition_graphs() = graph;
        }
      }

      collector->ClearGraphs();
    }
  }
  DCHECK_EQ(retvals.size(), outputs.size());
  for (int i = 0; i < retvals.size(); ++i) {
    DCHECK_EQ(kernel->device(), retvals[i]->op_device());
    DCHECK_EQ(ctx->CanonicalDevice(kernel->OutputDevice(i)),
              retvals[i]->device());

    TF_RETURN_IF_ERROR(retvals[i]->SetTensor(std::move(outputs[i])));
  }
  return Status::OK();
}

namespace {

Status LocalEagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                              EagerExecutor* executor, Device* dstd,
                              TensorHandle** result) {
  TF_RETURN_IF_ERROR(executor->status());
  TF_RETURN_IF_ERROR(TensorHandle::CreateEmptyLocalHandle(
      true, ctx->CanonicalDevice(dstd), dstd, h->resource_device(), h->dtype,
      ctx, result));

  // Note that `h` may not be currently ready. However execution order will
  // make sure that `h` is ready before the copy is actually done.
  std::unique_ptr<EagerNode> node(new CopyToDeviceNode(h, *result, dstd, ctx));
  Status s = executor->AddOrExecute(std::move(node));
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
  Device* send_device = h->DeviceOrHostCPU(ctx);

  bool sender_is_local = send_device->IsLocal();

  bool recver_is_local = device->IsLocal();

  if (!executor->Async()) {
    // In sync mode, always clear error to maintain the same behavior as before.
    // TODO(b/141004939): Remove this.
    executor->ClearError();
  }

  if (sender_is_local && recver_is_local) {
    return LocalEagerCopyToDevice(h, ctx, executor, device, result);
  } else {
#if defined(IS_MOBILE_PLATFORM)
    return errors::Unimplemented(
        "Eager's remote execution is not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
    if (mirror) {
      if (h->HasRemoteMirror(device)) {
        h->Ref();
        *result = h;
        return Status::OK();
      }
    }
    uint64 recv_op_id = 0;
    if (recver_is_local) {
      TF_RETURN_IF_ERROR(TensorHandle::CreateEmptyLocalHandle(
          true, /* d= */ device, /* op_device= */ device,
          /*resource_device=*/nullptr, h->dtype, ctx, result));
    } else {
      uint64 context_id = ctx->GetContextId();
      string remote_task;
      if (!DeviceNameUtils::GetTaskName(device->parsed_name(), &remote_task)) {
        return errors::InvalidArgument(
            "Unable to find remote task corresponding to device ",
            device->name());
      }
      recv_op_id = ctx->RemoteMgr()->NextOpId();
      auto tensor_handle_data =
          absl::make_unique<UnshapedRemoteTensorHandleData>(
              recv_op_id, 0, remote_task, context_id, ctx);
      if (mirror) {
        TF_RETURN_IF_ERROR(
            h->AddUnshapedRemoteMirror(std::move(tensor_handle_data), device));
        h->Ref();
        *result = h;
      } else {
        TF_RETURN_IF_ERROR(TensorHandle::CreateUnshapedRemoteHandle(
            std::move(tensor_handle_data), h->dtype, device, ctx, result));
      }
    }
    auto node = absl::make_unique<eager::RemoteCopyNode>(
        ctx, executor, h, result[0], device, recv_op_id);
    Status s = executor->AddOrExecute(std::move(node));
    if (!s.ok()) {
      result[0]->Unref();
    }
    return s;
#endif  // !IS_MOBILE_PLATFORM
  }
}
}  // namespace tensorflow
