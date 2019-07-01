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

#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
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

namespace tensorflow {

namespace {

// Copy of the definition in third_party/tensorflow/compiler/jit/defs.h
// Copied here because we don't currently compile XLA on windows. So, can't
// depend on it directly.
const char* const kXlaCompileAttr = "_XlaCompile";

// Initializes the step stats if needed.
void MaybeInitializeStepStats(StepStats* step_stats, EagerContext* ctx) {
  // Lazily initialize the RunMetadata with information about all devices if
  // this is the first call.
  while (step_stats->dev_stats_size() < ctx->devices()->size()) {
    int device_idx = step_stats->dev_stats_size();
    auto* dev_stats = step_stats->add_dev_stats();
    dev_stats->set_device(ctx->devices()->at(device_idx)->name());
  }
}

int StepStatsDeviceIndex(StepStats* step_stats, EagerContext* ctx,
                         Device* device) {
  // Find the current device's index.
  if (device == nullptr) {
    device = ctx->HostCPU();
  }
  for (int i = 0; i < ctx->devices()->size(); ++i) {
    if (ctx->devices()->at(i) == device ||
        ctx->devices()->at(i)->name() == device->name()) {
      return i;
    }
  }
  // TODO(apassos) do not fall back to host CPU if device is unknown.
  return 0;
}

// This function expects *handle to point to an existing tensor handle. The
// function will update the *handle to be pointed to the existing input tensor
// handle or else the newly copied tensor handle. The existing handle will have
// a Ref added, vs the new handle has a Ref due to being newly constructed.
//
// `op_device_name` is passed in explicitly because `op->device()` might be
// unset and we might have selected some specific device to run this op on.
Status MaybeCopyInputToExpectedDevice(EagerOperation* op,
                                      const string& op_device_name, int i,
                                      const Device* expected_input_device,
                                      RunMetadata* run_metadata,
                                      TensorHandle** result) {
  tensorflow::TensorHandle* handle = op->Inputs()[i];
  EagerContext* ctx = op->EagerContext();
  Device* handle_device = handle->device();
  const Device* actual_device =
      handle_device == nullptr ? ctx->HostCPU() : handle_device;

  if (expected_input_device == actual_device) {
    // No copy was done, so the result is just the original handle with a Ref
    handle->Ref();
    *result = handle;
    return Status::OK();
  }

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
          actual_device->name(), " (operation running on ", op_device_name, ")",
          " Tensors can be copied explicitly using:"
          " `with tf.device(device_name): x = tf.identity(x)`"
          " or transparently copied by using"
          " tf.config.experimental.set_device_policy('silent')."
          " Copying tensors between devices may slow down your model");
    case DEVICE_PLACEMENT_WARN:
      LOG(WARNING) << "before computing " << op->Name() << " input #" << i
                   << " was expected to be on " << expected_input_device->name()
                   << " but is actually on " << actual_device->name()
                   << " (operation running on " << op_device_name
                   << "). This triggers a copy which can be a performance "
                      "bottleneck.";
      break;
    case DEVICE_PLACEMENT_SILENT:  // Do nothing.
      break;
  }
  // We are only here if the policy is warn or silent copies, so we should
  // trigger a copy.
  auto pre_time_nanos = Env::Default()->NowNanos();
  TensorHandle* result_handle = nullptr;
  Status status =
      EagerCopyToDevice(handle, ctx, expected_input_device->name().c_str(),
                        ctx->MirrorTensors(), &result_handle);
  if (run_metadata != nullptr) {
    auto* step_stats = run_metadata->mutable_step_stats();
    MaybeInitializeStepStats(step_stats, ctx);
    // Record the sending on the source device for now.
    int device_idx = StepStatsDeviceIndex(step_stats, ctx, handle_device);
    auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
    auto* node_stats = dev_stats->add_node_stats();
    node_stats->set_node_name("_Send");
    node_stats->set_all_start_micros(pre_time_nanos / EnvTime::kMicrosToNanos);
    node_stats->set_all_start_nanos(pre_time_nanos);
    int64 now_nanos = Env::Default()->NowNanos();
    node_stats->set_op_end_rel_micros((now_nanos - pre_time_nanos) /
                                      EnvTime::kMicrosToNanos);
    node_stats->set_op_end_rel_nanos(now_nanos - pre_time_nanos);
    node_stats->set_all_end_rel_micros((now_nanos - pre_time_nanos) /
                                       EnvTime::kMicrosToNanos);
    node_stats->set_all_end_rel_nanos(now_nanos - pre_time_nanos);
  }
  if (!status.ok()) {
    if (result_handle != nullptr) result_handle->Unref();
    return errors::Internal("Failed copying input tensor from ",
                            actual_device->name(), " to ",
                            expected_input_device->name(), " in order to run ",
                            op->Name(), ": ", status.error_message());
  }

  *result = result_handle;

  return Status::OK();
}

// `op_device_name` the name of the device on which the op will run, if any.
// For functions running using function library runtime, the device can be
// unspecified.
Status ValidateInputTypeAndPlacement(EagerContext* ctx,
                                     const string& op_device_name,
                                     EagerOperation* op,
                                     const KernelAndDevice* kernel,
                                     RunMetadata* run_metadata) {
  if (kernel->num_inputs() != op->Inputs().size()) {
    return errors::InvalidArgument("expected ", kernel->num_inputs(),
                                   " inputs, got ", op->Inputs().size());
  }
  for (int i = 0; i < op->Inputs().size(); ++i) {
    const Device* expected_device = kernel->InputDevice(i);
    TensorHandle* handle = nullptr;
    TF_RETURN_IF_ERROR(MaybeCopyInputToExpectedDevice(
        op, op_device_name, i, expected_device, run_metadata, &handle));
    op->UpdateInput(i, handle);
    // Unref handle since it has a ref as an input now
    handle->Unref();
    if (handle->dtype != kernel->input_type(i)) {
      return errors::InvalidArgument(
          "cannot compute ", op->Name(), " as input #", i, "(zero-based)",
          " was expected to be a ", DataTypeString(kernel->input_type(i)),
          " tensor but is a ", DataTypeString(handle->dtype), " tensor");
    }
  }
  return Status::OK();
}

Status SelectDevice(const NodeDef& ndef, EagerContext* ctx, Device** device) {
  PrioritizedDeviceTypeVector final_devices;
  TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
      ctx->prioritized_device_type_list(), ndef, &final_devices));
  if (final_devices.empty()) {
    return errors::Internal("Could not find valid device for node.\nNode: ",
                            FormatNodeDefForError(ndef),
                            "\nAll kernels registered for op ", ndef.op(),
                            " :\n", KernelsRegisteredForOp(ndef.op()));
  }
  for (Device* d : *ctx->devices()) {
    if (d->device_type() == final_devices[0].first.type_string()) {
      *device = d;
      return Status::OK();
    }
  }
  return errors::Unknown("Could not find a device for node ",
                         FormatNodeDefForError(ndef));
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

bool IsMultiDevice(const FunctionDef* fdef, const string& op_device) {
  if (fdef == nullptr) {
    // Primitive op.
    return false;
  }

  // Run all functions as multi-device.
  return true;

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

Status ShouldCompileWithXLA(const EagerOperation* op, const Device* device,
                            const EagerContext* ctx, bool* compile_with_xla) {
  if (!op->is_function() || device == nullptr) {
    *compile_with_xla = false;
    return Status::OK();
  }

  // Does node have an explicit request to compile or not?
  Status status = op->Attrs().Get(kXlaCompileAttr, compile_with_xla);
  if (status.ok()) {
    VLOG(2) << "Caller explicitly requested "
            << (*compile_with_xla ? "" : "not ")
            << "to compile with XLA: " << op->DebugString();
    return Status::OK();
  }

  // Does FunctionDef have an explicit request to compile or not?
  const FunctionDef* function_def =
      ctx->func_lib(device)->GetFunctionLibraryDefinition()->Find(op->Name());
  if (function_def == nullptr) {
    return errors::NotFound("Failed to find function '", op->Name(), "'");
  }

  status = GetNodeAttr(AttrSlice(&function_def->attr()), kXlaCompileAttr,
                       compile_with_xla);
  if (status.ok()) {
    VLOG(2) << "Function definition explicitly specifies "
            << (*compile_with_xla ? "" : "not ") << "to compile with XLA";
    return Status::OK();
  }

  // No explicit requests. Compile for XLA devices by default.
  if (device->device_type() == "TPU" || device->device_type() == "XLA_GPU" ||
      device->device_type() == "XLA_CPU") {
    VLOG(2) << "Compiling " << op->Name()
            << " with XLA because it is running on an XLA device "
            << device->device_type();
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
Status EagerLocalExecute(EagerOperation* op,
                         gtl::InlinedVector<TensorHandle*, 2>* retvals,
                         int* num_retvals) {
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerLocalExecute: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);
  const string unspecified_device_name("<unspecified>");
  EagerContext* ctx = op->EagerContext();
  TF_RETURN_IF_ERROR(ctx->GetStatus());
  Device* device = op->Device();

  const string& maybe_unspecified_device_name =
      device == nullptr ? unspecified_device_name : device->name();
  Fprint128 cache_key =
      op->MutableAttrs()->CacheKey(maybe_unspecified_device_name);

  bool is_multi_device_function = IsMultiDevice(
      ctx->FindFunctionDef(op->Name()), maybe_unspecified_device_name);

  std::vector<Device*> input_dev_ptrs;
  // `input_tensor_shapes` contains (potentially a subset of) non DT_RESOURCE
  // arguments, and `input_resource_variable_dtypes_and_shapes` contains shapes
  // and underlying types for (potentially a subset) of DT_RESOURCE arguments.
  std::unordered_map<int, TensorShape> input_tensor_shapes;
  std::unordered_map<int, DtypeAndPartialTensorShape>
      input_resource_variable_dtypes_and_shapes;
  if (is_multi_device_function) {
    input_dev_ptrs.reserve(op->Inputs().size());
    // All inputs need to be on local devices.
    // TODO(b/122851476): This is a limitation of the current code base (but
    // should be possible to get around).
    // Code changes will need to be made to pass input objects to the
    // function library runtime instead of just "Tensor"s.
    // Once that is the case, we will be able to write a thin wrapper layer over
    // the EagerService that behaves similar to the current
    // ClusterFunctionLibraryRuntime/DistributedFunctionLibraryRuntime.
    for (int i = 0; i < op->Inputs().size(); i++) {
      TensorHandle* input = op->Inputs()[i];
      if (input->IsRemote()) {
        TensorHandle* handle = nullptr;
        TF_RETURN_IF_ERROR(EagerCopyToDevice(
            input, ctx, device == nullptr ? "" : device->name().c_str(),
            ctx->MirrorTensors(), &handle));
        op->UpdateInput(i, handle);
        // Unref handle since it has a ref as an input now
        handle->Unref();
        input = handle;
      }

      // Get device for this input, and add it to 'cache_key'.
      Device* device;
      TF_RETURN_IF_ERROR(GetDeviceForInput(ctx, input, &device));
      input_dev_ptrs.push_back(device);
      cache_key = FingerprintCat128(cache_key, Fingerprint128(device->name()));

      // If input is normal tensor, get its shape and add it to 'cache_key';
      // If input is a ResourceHandle, get its resource handle dtypes and shapes
      // and add them to 'cache_key'.
      if (input->dtype != DT_RESOURCE) {
        TensorShape shape;
        TF_RETURN_IF_ERROR(input->Shape(&shape));

        input_tensor_shapes[i] = shape;

        // Add both _Arg index and shape to "cache_key".
        cache_key = FingerprintCat128(cache_key, i);
        AppendTensorShapeToFingerprint(shape, &cache_key);
      } else {
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
    VLOG(2) << "Creating new kernel for " << op->Name() << " on device "
            << maybe_unspecified_device_name;
    bool compile_with_xla;
    TF_RETURN_IF_ERROR(
        ShouldCompileWithXLA(op, device, ctx, &compile_with_xla));
    if (compile_with_xla) {
      // Note that it is not ideal, but currently correct, to set this
      // attribute after computing the kernel cache key above.
      // TODO(iga): Creating XlaLaunchOp kernel directly here would be much
      // better than setting this attribute and relying on
      // custom_kernel_creator.
      // Note: If the attribute is already set to true, this is a noop.
      op->MutableAttrs()->Set(kXlaCompileAttr, true);
    }
    bool run_function_with_flr = is_multi_device_function && !compile_with_xla;

    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
    if (!run_function_with_flr && device == nullptr) {
      TF_RETURN_IF_ERROR(SelectDevice(ndef, ctx, &device));
    }
    const string& device_name =
        device == nullptr ? unspecified_device_name : device->name();
    if (ctx->LogDevicePlacement() || VLOG_IS_ON(1)) {
      string msg = strings::StrCat("Executing op ", ndef.op(), " in device ",
                                   device_name);
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
      VLOG(2) << "Running " << ndef.op() << " using multi-device function. "
              << "compile_with_xla=" << compile_with_xla
              << ". Full node_def=" << ndef.DebugString();
      kernel.reset(new KernelAndDeviceFunc(
          flr, ctx->pflr(), std::move(input_dev_ptrs),
          std::move(input_tensor_shapes),
          std::move(input_resource_variable_dtypes_and_shapes), runner,
          ctx->GetCollectiveExecutorHandle(), ctx->HostCPU(), op->Name(),
          [ctx](const int64 step_id) {
            return ctx->CreateRendezvous(step_id);
          }));
    } else {
      VLOG(2) << "Running " << ndef.op() << " using op kernel. "
              << "compile_with_xla=" << compile_with_xla
              << ". Full node_def=" << ndef.DebugString();
      kernel.reset(new KernelAndDeviceOp(
          ctx->GetRendezvous(), ctx->LogMemory(), flr, runner,
          ctx->GetCollectiveExecutorHandle(), ctx->HostCPU()));
    }

    TF_RETURN_IF_ERROR(kernel->Init(ndef, graph_collector));

    ctx->AddKernelToCache(cache_key, kernel.get());
  }
  const DataTypeVector& output_dtypes = kernel->output_dtypes();
  const int output_dtypes_size = static_cast<int>(output_dtypes.size());
  if (output_dtypes_size > *num_retvals) {
    return errors::InvalidArgument("Expecting ", output_dtypes.size(),
                                   " outputs, but *num_retvals is ",
                                   *num_retvals);
  }
  *num_retvals = output_dtypes_size;
  const string& device_name = kernel->device() == nullptr
                                  ? unspecified_device_name
                                  : kernel->device()->name();
  TF_RETURN_IF_ERROR(ValidateInputTypeAndPlacement(
      ctx, device_name, op, kernel.get(),
      ctx->ShouldStoreStepStats() ? ctx->RunMetadataProto() : nullptr));

  std::unique_ptr<NodeExecStats> maybe_stats;
  StepStats* maybe_step_stats = nullptr;
  GraphCollector* graph_collector = nullptr;
  if (ctx->ShouldStoreGraphs()) {
    graph_collector = ctx->GetGraphCollector();
  }
  if (ctx->ShouldStoreStepStats()) {
    maybe_step_stats = ctx->RunMetadataProto()->mutable_step_stats();
    int64 now_nanos = Env::Default()->NowNanos();
    maybe_stats.reset(new NodeExecStats);
    maybe_stats->set_node_name(op->Name());
    maybe_stats->set_all_start_micros(now_nanos / EnvTime::kMicrosToNanos);
    maybe_stats->set_all_start_nanos(now_nanos);
    maybe_stats->set_op_start_rel_micros(0);
    maybe_stats->set_op_start_rel_nanos(0);
    maybe_stats->set_scheduled_micros(now_nanos / EnvTime::kMicrosToNanos);
    maybe_stats->set_scheduled_nanos(now_nanos);
    // TODO(apassos) track referenced tensors
  }
  retvals->resize(*num_retvals);
  if (ctx->Async()) {
    // Note that for async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(agarwal): Consider executing "cheap" kernels inline for
    // performance.
    tensorflow::uint64 id = ctx->NextId();
    for (int i = 0; i < *num_retvals; ++i) {
      TF_RETURN_IF_ERROR(TensorHandle::CreateAsyncLocalHandle(
          /* d= */ kernel->OutputDevice(i),
          /* op_device= */ kernel->device(),
          /* resource_device= */ kernel->OutputResourceDevice(i),
          output_dtypes[i], ctx, &(*retvals)[i]));
    }
    std::unique_ptr<EagerNode> node(new ExecuteNode(
        id, ctx, op->Inputs(), std::move(kernel), maybe_stats.release(),
        maybe_step_stats, graph_collector, output_dtypes, *retvals));
    ctx->ExecutorAdd(std::move(node));

    return Status::OK();
  } else {
    // Execute checks if retvals[i] is nullptr or not to figure if it needs to
    // allocate it.
    return EagerKernelExecute(ctx, op->Inputs(), kernel.get(),
                              maybe_stats.get(), maybe_step_stats,
                              graph_collector, retvals->data(), *num_retvals);
  }
}

#if !defined(IS_MOBILE_PLATFORM)
// When !ctx->UseSendTensorRPC(), then tensors are shipped between remote
// devices by the receiver invoking the WorkerService.RecvTensor RPC *on the
// sender* (Rendezvous::RecvAsync() invoked by the _Recv kernel).
//
// However, in some configurations the node that has the tensor to be copied
// isn't running a server (WorkerService RPC interface). For such cases,
// this function enables sending tensors using the EagerService.SendTensor RPC
// *on the receiver*.
Status EagerRemoteSendTensor(EagerContext* ctx, TensorHandle* h,
                             Device* recv_device, bool mirror,
                             TensorHandle** result) {
  eager::EagerClient* eager_client;
  uint64 context_id = ctx->GetContextId();
  TF_RETURN_IF_ERROR(ctx->GetClient(recv_device, &eager_client));

  eager::SendTensorRequest request;
  eager::SendTensorResponse response;

  request.set_context_id(context_id);
  request.set_op_id(ctx->NextId());
  request.set_device_name(recv_device->name());

  Device* tensor_handle_device = h->device();

  // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence
  // copy it to the CPU before copying it out.
  // TODO(b/110044833): this is currently slow, but can be fixed by making
  // tensor handles aware of more than one device.
  TensorHandle* actual_handle;
  if (tensor_handle_device != nullptr &&
      tensor_handle_device->device_type() != "CPU") {
    Tensor tensor;
    TF_RETURN_IF_ERROR(h->CopyToDevice(ctx, ctx->HostCPU(), &tensor));
    TF_RETURN_IF_ERROR(
        TensorHandle::CreateLocalHandle(tensor, nullptr, ctx, &actual_handle));
  } else {
    actual_handle = h;
    actual_handle->Ref();
  }

  const Tensor* tensor;
  TF_RETURN_IF_ERROR(actual_handle->Tensor(&tensor));
  tensor->AsProtoTensorContent(request.add_tensors());

  const tensorflow::uint64 id = request.op_id();

  // TODO(nareshmodi): support making this call async.
  Notification n;
  Status status;
  eager_client->SendTensorAsync(&request, &response,
                                [&n, &status](const Status& s) {
                                  status = s;
                                  n.Notify();
                                });
  n.WaitForNotification();
  if (!status.ok()) return status;

  auto tensor_handle_data = absl::make_unique<RemoteTensorHandleData>(
      id, 0, tensor->shape(), eager_client, context_id, ctx);
  if (mirror) {
    status = h->AddRemoteMirror(std::move(tensor_handle_data), recv_device);
    h->Ref();
    *result = h;
  } else {
    status = TensorHandle::CreateRemoteHandle(std::move(tensor_handle_data),
                                              tensor->dtype(), recv_device,
                                              nullptr, ctx, result);
  }

  actual_handle->Unref();

  return status;
}

Status AddRemoteInput(eager::Operation* remote_op, TensorHandle* input,
                      Device* input_device) {
  tensorflow::int64 op_id;
  int32 output_num;
  TF_RETURN_IF_ERROR(input->RemoteAddress(input_device, &op_id, &output_num));

  auto* remote_op_input = remote_op->add_inputs();
  remote_op_input->set_op_id(op_id);
  remote_op_input->set_output_num(output_num);

  return Status::OK();
}

Status EnqueueAndWait(eager::EagerClient* eager_client,
                      const std::unique_ptr<eager::EnqueueRequest>& request,
                      eager::EnqueueResponse* response) {
  Notification n;
  Status status;
  eager_client->EnqueueAsync(request.get(), response,
                             [&n, &status](const Status& s) {
                               status = s;
                               n.Notify();
                             });
  n.WaitForNotification();

  return status;
}

void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
  EagerContext* ctx = op->EagerContext();

  remote_op->set_id(ctx->NextId());
  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(op->Device()->name());
}

Status EagerRemoteExecute(EagerOperation* op, TensorHandle** retvals,
                          int* num_retvals) {
  EagerContext* ctx = op->EagerContext();

  eager::EagerClient* eager_client;
  uint64 context_id = ctx->GetContextId();
  TF_RETURN_IF_ERROR(ctx->GetClient(op->Device(), &eager_client));

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  eager::EnqueueResponse response;

  request->set_context_id(context_id);

  eager::Operation* remote_op = request->add_queue()->mutable_operation();

  for (int i = 0; i < op->Inputs().size(); i++) {
    tensorflow::TensorHandle* input = op->Inputs()[i];
    tensorflow::Device* input_device = input->device();
    if (op->Device() != input_device &&
        // If the expected and actual devices are on the same task, don't
        // explicitly copy, and instead depend on the copy to happen locally
        // when the op is executed on the device.
        !ctx->OnSameTask(op->Device(), input_device)) {
      tensorflow::Device* remote_cpu_device;
      TF_RETURN_IF_ERROR(
          ctx->CPUDeviceOnTask(op->Device(), &remote_cpu_device));
      // TODO(b/110044833): It's possible the same tensor gets copied to the
      // remote device repeatedly.
      // Always copy to the remote CPU so that the actual device can be
      // correctly determined after the kernel is selected/instantiated, since
      // the op might have its inputs on host memory.
      TensorHandle* handle = nullptr;
      TF_RETURN_IF_ERROR(MaybeCopyInputToExpectedDevice(
          op, op->Device()->name(), i, remote_cpu_device,
          /* run_metadata= */ nullptr, &handle));
      op->UpdateInput(i, handle);
      input = handle;
      input_device = remote_cpu_device;
      // Unref handle since it has a ref as an input now
      handle->Unref();
    }

    TF_RETURN_IF_ERROR(AddRemoteInput(remote_op, input, input_device));
  }

  PrepareRemoteOp(remote_op, op);

  DataTypeVector output_dtypes;
  TF_RETURN_IF_ERROR(GetOutputDTypes(op, &output_dtypes));

  if (*num_retvals != output_dtypes.size()) {
    return errors::InvalidArgument(
        "num_retvals does not match expected output dtypes");
  }

  tensorflow::Device* op_device = op->Device();

  bool is_async = ctx->Async();
  uint64 remote_node_id = 0;

  VLOG(4) << "Execute remote eager op: " << op->Name()
          << " (is async?: " << is_async << ").";

  const tensorflow::uint64 id = remote_op->id();
  if (is_async) {
    remote_node_id = ctx->NextId();
    for (int i = 0; i < *num_retvals; i++) {
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
      TF_RETURN_IF_ERROR(TensorHandle::CreateUnshapedRemoteHandle(
          id, i, eager_client, context_id, output_dtypes[i], op_device,
          output_dtypes[i] == DT_RESOURCE ? op_device : nullptr, ctx,
          &retvals[i]));
    }

    // TODO(gjn): If the retval TensorHandle is simply going to be used as a
    // mirror then there should be no need to call SetRemoteShape
    std::unique_ptr<EagerNode> node(new eager::RemoteExecuteNode(
        remote_node_id, std::move(request), eager_client, op->Inputs(), retvals,
        *num_retvals));
    ctx->ExecutorAdd(std::move(node));
  } else {
    TF_RETURN_IF_ERROR(EnqueueAndWait(eager_client, request, &response));

    for (int i = 0; i < *num_retvals; i++) {
      TF_RETURN_IF_ERROR(TensorHandle::CreateRemoteHandle(
          id, i, response.queue_response(0).shape(i), eager_client, context_id,
          output_dtypes[i], op_device,
          output_dtypes[i] == DT_RESOURCE ? op_device : nullptr, ctx,
          &retvals[i]));
    }
  }

  return Status::OK();
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
  auto exempt_ops = InputColocationExemptionRegistry::Global()->Get();
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
      VLOG(2) << "for op " << op->Name() << " input " << i << " "
              << DataTypeString(tensor_handle->dtype)
              << " input device = " << resource_device->name()
              << ", op device = " << op_device->name();
      // We check for `op->Device() == nullptr` because it can be later
      // interpreted as unspecified device and a different device can
      // be selected based on device priority. If any input to an op
      // is a resource we must pin it to prevent different device selection.
      // TODO(iga): null device can mean "unspecified" or "CPU". Clean this up.
      if (resource_device != op_device || op->Device() == nullptr) {
        VLOG(1) << (resource_device != op_device ? "Changing " : "Setting ")
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
      Device* input_device = tensor_handle->device();
      input_device = input_device == nullptr ? ctx->HostCPU() : input_device;
      VLOG(2) << "for op " << op->Name() << " input " << i << " "
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
    VLOG(1) << "Forcing op " << op->Name()
            << " to be on the CPU since all input tensors have an "
               "int32/int64 dtype, and are small (less than 64 elements).";
    op->SetDevice(ctx->HostCPU());
  }

  return Status::OK();
}
}  // namespace

Status EagerExecute(EagerOperation* op,
                    gtl::InlinedVector<TensorHandle*, 2>* retvals,
                    int* num_retvals) {
  profiler::TraceMe activity(
      [&] { return absl::StrCat("EagerExecute: ", op->Name()); },
      profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(MaybeUpdateOpDevice(op));

  bool op_is_local = op->EagerContext()->IsLocal(op->Device());

  if (op_is_local) {
    return EagerLocalExecute(op, retvals, num_retvals);
  }

  if (op->EagerContext()->LogDevicePlacement() || VLOG_IS_ON(1)) {
    string msg = strings::StrCat("Executing op ", op->Name(), " in device ",
                                 op->Device()->name());
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }

#if defined(IS_MOBILE_PLATFORM)
  return errors::Unimplemented(
      "Eager's remote execution is not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
  return EagerRemoteExecute(op, retvals->data(), num_retvals);
#endif  // !IS_MOBILE_PLATFORM
}

Status EagerKernelExecute(EagerContext* ctx,
                          const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
                          KernelAndDevice* kernel, NodeExecStats* maybe_stats,
                          StepStats* maybe_step_stats,
                          GraphCollector* graph_collector,
                          TensorHandle** retvals, int num_retvals) {
  profiler::TraceMe activity("EagerKernelExecute",
                             profiler::TraceMeLevel::kInfo);
  std::vector<Tensor> outputs(1);

  // If there are multiple references to a TensorHandle in 'op_inputs' we must
  // increment the reference count of the corresponding Tensor or risk it being
  // overwritten during kernel execution. The reference count is incremented
  // below when we insert a copy of the Tensor into protected_tensors, and will
  // be decremented once execution is complete.
  std::vector<tensorflow::Tensor> protected_tensors;
  for (int i = 0; i < op_inputs.size(); ++i) {
    if (!op_inputs[i]->RefCountIsOne()) {
      const Tensor* input_tensor = nullptr;
      TF_RETURN_IF_ERROR(op_inputs[i]->Tensor(&input_tensor));
      protected_tensors.push_back(*input_tensor);
    }
  }

  gtl::InlinedVector<TensorValue, 4> input_vector(op_inputs.size());
  for (int i = 0; i < op_inputs.size(); ++i) {
    TF_RETURN_IF_ERROR(op_inputs[i]->TensorValue(&input_vector[i]));
  }

  // TODO(apassos) figure out how to record stats for ops which are a part of
  // functions.
  // TODO(agarwal): change Run to take vector of handles ?
  // TODO(b/111859745): When we support recovering from kernel/device errors, we
  // would need to call XlaDevice::EnsureDeviceContextOk() before using an XLA
  // device. We don't call it now because it is an unneeded overhead (it
  // acquires a lock) and we can't recover from errors anyway.
  ScopedStepContainer* container = ctx->StepContainer();
  if (container == nullptr) {
    TF_RETURN_IF_ERROR(kernel->Run(input_vector, &outputs, maybe_stats,
                                   maybe_step_stats, graph_collector));
  } else {
    TF_RETURN_IF_ERROR(kernel->Run(container, input_vector, &outputs,
                                   maybe_stats, maybe_step_stats,
                                   graph_collector));
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
  if (maybe_stats != nullptr) {
    int64 nanos = Env::Default()->NowNanos();
    maybe_stats->set_op_end_rel_micros(nanos / EnvTime::kMicrosToNanos -
                                       maybe_stats->all_start_micros());
    maybe_stats->set_op_end_rel_nanos(nanos - maybe_stats->all_start_nanos());
    maybe_stats->set_all_end_rel_micros(nanos / EnvTime::kMicrosToNanos -
                                        maybe_stats->all_start_micros());
    maybe_stats->set_all_end_rel_nanos(nanos - maybe_stats->all_start_nanos());
    if (ctx->ShouldStoreStepStats()) {
      mutex_lock ml(*ctx->MetadataMu());
      {
        auto* step_stats = ctx->RunMetadataProto()->mutable_step_stats();
        // Lazily initialize the RunMetadata with information about all devices
        // if this is the first call.
        while (step_stats->dev_stats_size() < ctx->devices()->size()) {
          step_stats->add_dev_stats();
        }
        // Find the current device's index.
        // If device is a nullptr (we are running a function without explicitly
        // requested device), attribute the function runtime to CPU.
        Device* attribution_device = kernel->device();
        if (attribution_device == nullptr) {
          attribution_device = ctx->HostCPU();
        }
        int device_idx = 0;
        for (int i = 0; i < ctx->devices()->size(); ++i) {
          if (ctx->devices()->at(i) == attribution_device) {
            device_idx = i;
            break;
          }
        }
        // Populate the device stats for this device.
        auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
        dev_stats->set_device(attribution_device->name());
        *dev_stats->add_node_stats() = *maybe_stats;
      }
    }
  }
  DCHECK_EQ(num_retvals, outputs.size());
  for (int i = 0; i < num_retvals; ++i) {
    if (retvals[i] == nullptr) {
      TF_RETURN_IF_ERROR(TensorHandle::CreateLocalHandle(
          outputs[i], /* d= */ kernel->OutputDevice(i),
          /* op_device= */ kernel->device(), ctx, &retvals[i]));
    } else {
      // In the async case, the retval is not a nullptr, and its device is
      // already set since all TensorHandles always have their device set
      // (potentially to nullptr) during construction.
      DCHECK_EQ(kernel->device(), retvals[i]->op_device());
      DCHECK_EQ(kernel->OutputDevice(i), retvals[i]->device());

      TF_RETURN_IF_ERROR(retvals[i]->SetTensor(outputs[i]));
    }
  }
  return Status::OK();
}

namespace {

Status LocalEagerCopyToDevice(TensorHandle* h, EagerContext* ctx, Device* dstd,
                              TensorHandle** result) {
  TF_RETURN_IF_ERROR(ctx->GetStatus());
  if (ctx->Async()) {
    TF_RETURN_IF_ERROR(TensorHandle::CreateAsyncLocalHandle(
        dstd, dstd, nullptr, h->dtype, ctx, result));
    // Note that `h` may not be currently ready. However execution order will
    // make sure that `h` is ready before the copy is actually done.
    std::unique_ptr<EagerNode> node(
        new CopyToDeviceNode(h, *result, dstd, ctx));
    // Note that calling Add makes `node` accessible by the EagerExecutor
    // thread. So further accesses need to be thread-safe.
    ctx->ExecutorAdd(std::move(node));
    return Status::OK();
  } else {
    tensorflow::Tensor tensor;
    TF_RETURN_IF_ERROR(h->CopyToDevice(ctx, dstd, &tensor));
    const bool dst_cpu = dstd->tensorflow_gpu_device_info() == nullptr;
    return TensorHandle::CreateLocalHandle(tensor, dst_cpu ? nullptr : dstd,
                                           ctx, result);
    return Status::OK();
  }
}

#if !defined(IS_MOBILE_PLATFORM)
Status CreateUncachedKernelAndDeviceOp(
    EagerOperation* op, core::RefCountPtr<KernelAndDevice>* kernel) {
  EagerContext* ctx = op->EagerContext();
  Device* device = op->Device();

  FunctionLibraryRuntime* flr = ctx->func_lib(device);
  if (flr == nullptr) {
    return errors::Unavailable(
        "Unable to find a FunctionLibraryRuntime corresponding to device ",
        device->name());
  }

  auto runner = (flr->runner() != nullptr) ? flr->runner() : ctx->runner();
  kernel->reset(new KernelAndDeviceOp(
      ctx->GetRendezvous(), ctx->LogMemory(), flr, runner,
      ctx->GetCollectiveExecutorHandle(), ctx->HostCPU()));

  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  return kernel->get()->Init(ndef, nullptr);
}

Status ExecuteSend(EagerContext* ctx, Device* device, TensorHandle* h,
                   StringPiece wire_id, Device* recv_device) {
  // TODO(gjn): We should consider just using the low-level SendOp::Compute()
  // functionality here instead of constructing an Op.
  const AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp("_Send", &types, &is_function));
  DCHECK(!is_function);
  EagerOperation op(ctx, "_Send", /*is_function=*/false, types);

  op.SetDevice(device);

  op.MutableAttrs()->Set("tensor_name", wire_id);
  op.MutableAttrs()->Set("send_device", device->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64>(device->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("T", h->dtype);

  DCHECK(device != nullptr);

  if (ctx->IsLocal(device)) {
    TF_RETURN_IF_ERROR(ctx->GetStatus());

    op.AddInput(h);

    core::RefCountPtr<KernelAndDevice> kernel;
    TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(&op, &kernel));

    gtl::InlinedVector<TensorValue, 4> input_vector(1);
    TF_RETURN_IF_ERROR(h->TensorValue(&input_vector[0]));

    TF_RETURN_IF_ERROR(
        kernel->Run(input_vector, nullptr, nullptr, nullptr, nullptr));
  } else {
    eager::EagerClient* eager_client;
    uint64 context_id = ctx->GetContextId();
    TF_RETURN_IF_ERROR(
        ctx->GetClient(device, &eager_client));

    std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
    eager::EnqueueResponse response;

    request->set_context_id(context_id);

    auto* remote_op = request->add_queue()->mutable_operation();
    TF_RETURN_IF_ERROR(AddRemoteInput(remote_op, h, h->device()));

    PrepareRemoteOp(remote_op, &op);

    if (ctx->Async()) {
      uint64 remote_node_id = ctx->NextId();

      std::unique_ptr<EagerNode> node(
          new eager::RemoteExecuteNode(remote_node_id, std::move(request),
                                       eager_client, op.Inputs(), nullptr, 0));
      ctx->ExecutorAdd(std::move(node));
    } else {
      TF_RETURN_IF_ERROR(EnqueueAndWait(eager_client, request, &response));
    }
  }

  return Status::OK();
}

// Execute a Recv to transfer a tensor handle to a specific device. The received
// tensor handle will be returned in result. If mirror_dst is provided, the
// tensor handle will be added as a mirror.
Status ExecuteRecv(EagerContext* ctx, Device* device, DataType dtype,
                   StringPiece wire_id, Device* send_device,
                   TensorHandle* mirror_dst, TensorHandle** result) {
  // TODO(gjn): We should consider just using the low-level RecvOp::Compute()
  // functionality here instead of constructing an Op.
  const AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp("_Recv", &types, &is_function));
  DCHECK(!is_function);
  EagerOperation op(ctx, "_Recv", /*is_function=*/false, types);

  op.SetDevice(device);

  op.MutableAttrs()->Set("tensor_name", wire_id);
  op.MutableAttrs()->Set("send_device", send_device->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64>(send_device->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", device->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("tensor_type", dtype);

  if (ctx->IsLocal(device)) {
    TF_RETURN_IF_ERROR(ctx->GetStatus());

    core::RefCountPtr<KernelAndDevice> kernel;
    TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(&op, &kernel));

    std::vector<Tensor> outputs;
    gtl::InlinedVector<TensorValue, 4> input_vector;
    TF_RETURN_IF_ERROR(
        kernel->Run(input_vector, &outputs, nullptr, nullptr, nullptr));

    // TODO(gjn): Add support for async mode
    TF_RETURN_IF_ERROR(TensorHandle::CreateLocalHandle(
        outputs[0], /* d= */ kernel->OutputDevice(0),
        /* op_device= */ kernel->device(), ctx, result));
  } else {
    eager::EagerClient* eager_client;
    uint64 context_id = ctx->GetContextId();
    TF_RETURN_IF_ERROR(
        ctx->GetClient(device, &eager_client));

    std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
    eager::EnqueueResponse response;

    request->set_context_id(context_id);

    auto* remote_op = request->add_queue()->mutable_operation();
    PrepareRemoteOp(remote_op, &op);

    const uint64 id = remote_op->id();
    if (ctx->Async()) {
      uint64 remote_node_id = ctx->NextId();

      TF_RETURN_IF_ERROR(TensorHandle::CreateUnshapedRemoteHandle(
          id, 0, eager_client, context_id, dtype, device,
          dtype == DT_RESOURCE ? device : nullptr, ctx, result));

      std::unique_ptr<EagerNode> node(
          new eager::RemoteExecuteNode(remote_node_id, std::move(request),
                                       eager_client, op.Inputs(), result, 1));
      ctx->ExecutorAdd(std::move(node));
    } else {
      TF_RETURN_IF_ERROR(EnqueueAndWait(eager_client, request, &response));

      auto tensor_handle_data = absl::make_unique<RemoteTensorHandleData>(
          id, 0, response.queue_response(0).shape(0), eager_client, context_id,
          ctx);
      if (mirror_dst != nullptr) {
        TF_RETURN_IF_ERROR(
            mirror_dst->AddRemoteMirror(std::move(tensor_handle_data), device));
        mirror_dst->Ref();
        *result = mirror_dst;
      } else {
        TF_RETURN_IF_ERROR(TensorHandle::CreateRemoteHandle(
            std::move(tensor_handle_data), dtype, device,
            dtype == DT_RESOURCE ? device : nullptr, ctx, result));
      }
    }
  }

  return Status::OK();
}

// This gets a unique wire ID. We add a random identifier so that if the
// worker has other clients that it is servicing, we don't have any collision.
string GetUniqueWireID() {
  static tensorflow::uint64 random_seed = random::New64();
  static tensorflow::mutex wireid_mutex(tensorflow::LINKER_INITIALIZED);
  static tensorflow::int64 wireid GUARDED_BY(wireid_mutex) = 0;
  tensorflow::mutex_lock l(wireid_mutex);
  return strings::StrCat(random_seed, "_", wireid++);
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace

Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                         const char* device_name, bool mirror,
                         TensorHandle** result) {
  profiler::TraceMe activity("EagerCopyToDevice",
                             profiler::TraceMeLevel::kInfo);
  Device* send_device = h->device();

  if (send_device == nullptr) {
    send_device = ctx->HostCPU();
  }

  bool sender_is_local = ctx->IsLocal(send_device);

  Device* recv_device;
  TF_RETURN_IF_ERROR(ctx->FindDeviceFromName(device_name, &recv_device));

  bool recver_is_local = ctx->IsLocal(recv_device);

  if (sender_is_local && recver_is_local) {
    return LocalEagerCopyToDevice(h, ctx, recv_device, result);
  } else {
#if defined(IS_MOBILE_PLATFORM)
    return errors::Unimplemented(
        "Eager's remote execution is not available on mobile devices.");
#else   // !IS_MOBILE_PLATFORM
    if (mirror) {
      if (h->HasRemoteMirror(recv_device)) {
        h->Ref();
        *result = h;
        return Status::OK();
      }
    }

    if (ctx->UseSendTensorRPC() && sender_is_local && !recver_is_local) {
      return EagerRemoteSendTensor(ctx, h, recv_device, mirror, result);
    } else {
      string wire_id = GetUniqueWireID();
      TF_RETURN_IF_ERROR(
          ExecuteSend(ctx, send_device, h, wire_id, recv_device));

      return ExecuteRecv(ctx, recv_device, h->dtype, wire_id, send_device,
                         mirror ? h : nullptr, result);
    }
#endif  // !IS_MOBILE_PLATFORM
  }
}
}  // namespace tensorflow
