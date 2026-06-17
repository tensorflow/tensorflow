/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {

absl::Status EagerKernelArgs::GetLocalArg(const FunctionArgIndex& index,
                                          Tensor* val) const {
  if (index.sub_index >= 0) {
    return errors::InvalidArgument("Got unexpected sub_index ", index.sub_index,
                                   " for argument ", index.index);
  }
  Tensor* arg = tensor_args_.at(index.index).tensor;
  if (arg) {
    *val = *arg;
    return absl::OkStatus();
  } else {
    return errors::NotFound("Argument ", index.index, " has no local tensor.");
  }
}

std::vector<Tensor> EagerKernelArgs::GetLocalTensors() const {
  std::vector<Tensor> local_inputs;
  local_inputs.reserve(tensor_args_.size());
  for (const TensorValue& tensor_value : tensor_args_) {
    local_inputs.push_back(*tensor_value.tensor);
  }
  return local_inputs;
}

std::function<void(std::function<void()>)>* KernelAndDevice::get_runner()
    const {
  if (runner_) {
    return runner_;
  } else {
    static auto* default_runner =
        new std::function<void(std::function<void()>)>(
            [](const std::function<void()>& f) { f(); });
    return default_runner;
  }
}

KernelAndDeviceFunc::~KernelAndDeviceFunc() {
  if (handle_ != kInvalidHandle) {
    absl::Status status = pflr_->ReleaseHandle(handle_);
    if (!status.ok()) {
      LOG(INFO) << "Ignoring error status when releasing multi-device function "
                   "handle "
                << status;
    }
  }
}

absl::Status KernelAndDeviceOp::Init(
    const bool log_device_placement, const NodeDef& ndef,
    GraphCollector* graph_collecto,
    const absl::optional<EagerFunctionParams>& eager_func_params) {
  if (eager_func_params.has_value()) {
    return absl::InternalError(
        "KernelAndDeviceOp does not support EagerFunctionParams.");
  }
  OpKernel* k = nullptr;
  if (flr_ == nullptr) {
    return errors::Internal(
        "A valid FunctionLibraryRuntime must be provided when running ops "
        "based on OpKernel.");
  }
  std::shared_ptr<const NodeProperties> props;
  TF_RETURN_IF_ERROR(NodeProperties::CreateFromNodeDef(
      ndef, flr_->GetFunctionLibraryDefinition(), &props));
  TF_RETURN_IF_ERROR(flr_->CreateKernel(props, &k));
  kernel_.reset(k);
  const auto* op_reg_data = OpRegistry::Global()->LookUp(ndef.op());
  if (op_reg_data != nullptr) {
    is_distributed_communication_op_ =
        op_reg_data->op_def.is_distributed_communication();
  }

  input_alloc_attrs_.resize(kernel_->num_inputs());
  input_devices_.resize(kernel_->num_inputs(), device_);
  for (size_t i = 0; i < input_alloc_attrs_.size(); ++i) {
    bool host = kernel_->input_memory_types()[i] == tensorflow::HOST_MEMORY;
    input_alloc_attrs_[i].set_on_host(host);
    if (host && input_devices_[i]->device_type() != DEVICE_CPU) {
      input_devices_[i] = host_cpu_device_;
    }
  }
  output_alloc_attrs_.resize(kernel_->num_outputs());
  for (size_t i = 0; i < output_alloc_attrs_.size(); ++i) {
    output_alloc_attrs_[i].set_on_host(kernel_->output_memory_types()[i] ==
                                       tensorflow::HOST_MEMORY);
  }

  return absl::OkStatus();
}

absl::Status KernelAndDeviceFunc::InstantiateFunc(
    const bool log_device_placement, const NodeDef& ndef,
    GraphCollector* graph_collector,
    const absl::optional<EagerFunctionParams>& eager_func_params) {
  const OpDef* op_def = nullptr;
  const FunctionLibraryDefinition* func_lib_def;
  FunctionLibraryRuntime::InstantiateOptions options;

  if (eager_func_params.has_value() &&
      eager_func_params.value().func_lib_def_override != nullptr) {
    func_lib_def = eager_func_params.value().func_lib_def_override;
    options.lib_def = func_lib_def;
  } else {
    if (flr_ == nullptr) {
      // If function is being executed without an explicit device request,
      // lookup the FunctionDef in the CPU's FLR. All FLRs share the same
      // library.
      func_lib_def = pflr_->GetFLR(host_cpu_device_->name())
                         ->GetFunctionLibraryDefinition();
    } else {
      func_lib_def = flr_->GetFunctionLibraryDefinition();
    }
  }

  const FunctionDef* function_def = func_lib_def->Find(ndef.op());
  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(ndef.op(), &op_def));
  }
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(ndef, *op_def, &input_dtypes_, &output_dtypes_));

  options.target = device_ == nullptr ? "" : device_->name();
  options.is_multi_device_function = true;
  for (const Device* device : input_devices_) {
    options.input_devices.push_back(device->name());
  }
  options.composite_devices = composite_devices_;
  options.input_resource_dtypes_and_shapes = input_resource_dtypes_and_shapes_;
  if (outputs_on_op_device_) {
    if (function_def == nullptr) {
      return errors::InvalidArgument("Failed to find function ", ndef.op());
    }
    for (int i = 0; i < function_def->signature().output_arg_size(); ++i) {
      options.output_devices.push_back(options.target);
    }
  }

  const auto& it = ndef.attr().find("executor_type");
  if (it != ndef.attr().end()) {
    options.executor_type = it->second.s();
  }
  const auto& is_component_fn_it = ndef.attr().find("is_component_function");
  if (is_component_fn_it != ndef.attr().end()) {
    options.is_component_function = is_component_fn_it->second.b();
  }
#if !defined(IS_MOBILE_PLATFORM)
  // Android tf library does not include grappler.
  const auto& config_it = ndef.attr().find("config_proto");
  if (config_it != ndef.attr().end()) {
    if (!options.config_proto.ParseFromString(config_it->second.s())) {
      return errors::InvalidArgument(
          "Failed to parse config_proto attribute as tensorflow::ConfigProto "
          "proto.");
    }
    grappler::GrapplerItem::OptimizationOptions optimization_options =
        grappler::CreateOptOptionsForEager();

    options.optimize_graph_fn = std::bind(
        grappler::OptimizeGraph, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        options.config_proto, function_def->signature().name(),
        optimization_options, std::placeholders::_6);
  }
#endif  // !IS_MOBILE_PLATFORM
  options.graph_collector = graph_collector;

  options.allow_small_function_optimizations =
      allow_small_function_optimizations_;

  options.allow_control_flow_sync_execution =
      allow_control_flow_sync_execution_;

  options.shape_inference_on_tfe_dialect_import =
      shape_inference_on_tfe_dialect_import_;

  // In Eager mode we always inline all functions into the top-level
  // function body graph, to get a single executable graph, that could be
  // optimized across function boundaries (e.g. prune unused inputs and
  // outputs in a function call chain). This is required to mimic graph mode
  // execution, with aggressive pruning of nodes not in the transitive fanin
  // of fetches.
  options.config_proto.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);

  options.config_proto.set_log_device_placement(log_device_placement);

  options.int_args_and_retvals_on_device = int_args_and_retvals_on_device_;
  options.function_runs_at_most_once = function_runs_at_most_once_;

  if (xla_compile_device_type_.has_value()) {
    options.xla_compile_device_type = xla_compile_device_type_.value();
  }

  options.allow_soft_placement = allow_soft_placement_;

  TF_RETURN_IF_ERROR(
      pflr_->Instantiate(ndef.op(), AttrSlice(ndef), options, &handle_));
  return pflr_->IsCrossProcess(handle_, &is_cross_process_);
}

absl::Status KernelAndDeviceFunc::Init(
    const bool log_device_placement, const NodeDef& ndef,
    GraphCollector* graph_collector,
    const absl::optional<EagerFunctionParams>& eager_func_params) {
  TF_RETURN_IF_ERROR(InstantiateFunc(log_device_placement, ndef,
                                     graph_collector, eager_func_params));
  return pflr_->GetOutputDevices(handle_, &output_devices_);
}

namespace {
// In certain contexts (e.g. TPU async executions), the CancellationManager is
// used to shut down the device in error scenarios (as opposed to using the
// AsyncCompute's DoneCallback). This is handled through the
// {inc,dec}_num_deferred_ops_function.
struct OpExecutionState : public core::RefCounted {
  // TODO(nareshmodi): consider refcounting the cancellation_manager.
  CancellationManager cancellation_manager;
};
}  // anonymous namespace

absl::Status KernelAndDeviceOp::Run(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const absl::optional<ManagedStackTrace>& stack_trace,
    tsl::CoordinationServiceAgent* coordination_service_agent) {
  OpKernelContext::Params params;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = *inputs.GetTensorValues();
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.input_alloc_attrs = input_alloc_attrs_;
  params.output_attr_array = output_alloc_attrs_.data();
  params.function_library = flr_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendezvous_;
  params.stack_trace = stack_trace;
  OpExecutionState* op_execution_state = nullptr;

  CancellationManager default_cancellation_manager;
  if (cancellation_manager) {
    params.cancellation_manager = cancellation_manager;
  } else if (kernel_->is_deferred()) {
    op_execution_state = new OpExecutionState;
    params.cancellation_manager = &op_execution_state->cancellation_manager;
    params.inc_num_deferred_ops_function = [op_execution_state]() {
      op_execution_state->Ref();
    };
    params.dec_num_deferred_ops_function = [op_execution_state]() {
      op_execution_state->Unref();
    };
  } else {
    params.cancellation_manager = &default_cancellation_manager;
  }

  params.log_memory = log_memory_;

  params.runner = get_runner();

  params.step_container = step_container;

  params.collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;

  params.coordination_service_agent = coordination_service_agent;

  OpKernelContext context(&params);

  {
    port::ScopedFlushDenormal flush;
    port::ScopedSetRound round(FE_TONEAREST);
    // 'AnnotatedTraceMe' will trace both scheduling time on host and execution
    // time on device of the OpKernel.
    profiler::AnnotatedTraceMe activity(
        [&] { return kernel_->TraceString(context, /*verbose=*/false); },
        tsl::profiler::TraceMeLevel::kInfo);
    device_->Compute(kernel_.get(), &context);
  }

  // Clean up execution op_execution_state if deferred ops aren't running.
  if (op_execution_state != nullptr) {
    op_execution_state->Unref();
  }

  absl::Status s = context.status();
  if (TF_PREDICT_FALSE(!s.ok())) {
    if (absl::IsUnavailable(s) && !is_distributed_communication_op_) {
      s = errors::ReplaceErrorFromNonCommunicationOps(s, kernel_->name());
    }
    return s;
  }

  if (outputs != nullptr) {
    outputs->clear();
    for (int i = 0; i < context.num_outputs(); ++i) {
      const auto* output_tensor = context.mutable_output(i);
      if (output_tensor != nullptr) {
        outputs->push_back(Tensor(*output_tensor));
      } else {
        outputs->push_back(Tensor());
      }
    }
  }
  return absl::OkStatus();
}

std::shared_ptr<FunctionLibraryRuntime::Options>
KernelAndDeviceFunc::PrepareForRun(
    ScopedStepContainer* step_container, std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const absl::optional<ManagedStackTrace>& stack_trace,
    tsl::CoordinationServiceAgent* coordination_service_agent,
    tsl::core::RefCountPtr<Rendezvous>* rendezvous) {
  std::shared_ptr<FunctionLibraryRuntime::Options> opts = nullptr;
  if (eager_func_params.has_value()) {
    const EagerFunctionParams& params = eager_func_params.value();
    if (params.step_id.has_value()) {
      // If the function is a remote component of a cross-process function,
      // re-use the step id as its parent function's.
      opts = std::make_shared<FunctionLibraryRuntime::Options>(
          params.step_id.value());
    } else {
      opts = std::make_shared<FunctionLibraryRuntime::Options>();
    }
    // Reuse the op id if it exists.
    if (params.op_id != kInvalidOpId) {
      opts->op_id = params.op_id;
    }
  } else {
    opts = std::make_shared<FunctionLibraryRuntime::Options>();
    if (get_op_id_ && is_cross_process_) {
      // If the function is a cross-process function and the remote execution
      // goes through eager service, create an eager op id for the function.
      opts->op_id = get_op_id_();
    }
  }

  // We don't pass rendezvous from eager context because we can get tensor
  // name collisions in send/recv ops when running multiple instances
  // of the same multi-device function concurrently.
  TF_CHECK_OK(rendezvous_factory_(opts->step_id, nullptr, rendezvous));
  opts->rendezvous = rendezvous->get();
  opts->create_rendezvous = false;

  // Create a cancellation manager to be used by FLR options if caller does not
  // pass in one. If the caller does provide one, pass it to process FLR and the
  // locally created one will be unused.
  std::shared_ptr<CancellationManager> local_cm;
  if (cancellation_manager) {
    opts->cancellation_manager = cancellation_manager;
  } else {
    opts->cancellation_manager = new CancellationManager;
  }
  opts->allow_dead_tensors = true;
  opts->step_container = step_container;
  opts->collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;
  opts->stack_trace = stack_trace;

  opts->stats_collector = nullptr;
  opts->runner = get_runner();
  opts->coordination_service_agent = coordination_service_agent;

  outputs->clear();
  return opts;
}

absl::Status KernelAndDeviceFunc::Run(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const absl::optional<ManagedStackTrace>& stack_trace,
    tsl::CoordinationServiceAgent* coordination_service_agent) {
  tsl::profiler::TraceMe activity("KernelAndDeviceFunc::Run",
                                  tsl::profiler::TraceMeLevel::kInfo);
  // Don't try to handle packed or remote inputs synchronously.
  if (inputs.HasRemoteOrPackedInputs() || eager_func_params.has_value()) {
    absl::Notification n;
    absl::Status status;
    RunAsync(step_container, inputs, outputs, cancellation_manager,
             eager_func_params, coordination_service_agent,
             [&status, &n](absl::Status s) {
               status = s;
               n.Notify();
             });
    n.WaitForNotification();
    return status;
  }
  tsl::core::RefCountPtr<Rendezvous> created_rendezvous;
  std::shared_ptr<FunctionLibraryRuntime::Options> opts = PrepareForRun(
      step_container, outputs, cancellation_manager, eager_func_params,
      stack_trace, coordination_service_agent, &created_rendezvous);

  std::vector<Tensor> rets;
  absl::Status s;
  {
    port::ScopedFlushDenormal flush;
    port::ScopedSetRound round(FE_TONEAREST);
    s.Update(pflr_->RunSync(*opts, handle_, inputs.GetLocalTensors(), &rets));
  }

  if (cancellation_manager == nullptr) {
    delete opts->cancellation_manager;
  }
  outputs->reserve(rets.size());
  for (auto& v : rets) {
    outputs->push_back(std::move(v));
  }
  return s;
}

void KernelAndDeviceFunc::RunAsync(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    tsl::CoordinationServiceAgent* coordination_service_agent,
    std::function<void(const absl::Status&)> done) {
  tsl::profiler::TraceMe activity(
      [] {
        return tsl::profiler::TraceMeEncode("KernelAndDeviceFunc::RunAsync",
                                            {{"_r", 1}});
      },
      tsl::profiler::TraceMeLevel::kInfo);
  tsl::core::RefCountPtr<Rendezvous> created_rendezvous;
  std::shared_ptr<FunctionLibraryRuntime::Options> opts = PrepareForRun(
      step_container, outputs, cancellation_manager, eager_func_params,
      std::nullopt, coordination_service_agent, &created_rendezvous);

  pflr_->Run(*opts, handle_, inputs, outputs,
             [opts, cancellation_manager, done = std::move(done),
              created_rendezvous =
                  created_rendezvous.release()](const absl::Status& s) {
               if (cancellation_manager == nullptr) {
                 delete opts->cancellation_manager;
               }
               created_rendezvous->Unref();
               done(s);
             });
}

tensorflow::Device* KernelAndDeviceOp::OutputDevice(int idx) const {
  if (kernel_->output_memory_types()[idx] == HOST_MEMORY) {
    return nullptr;
  }
  return device_;
}

tensorflow::Device* KernelAndDeviceFunc::OutputDevice(int idx) const {
  if (output_dtypes_[idx] == DT_RESOURCE) {
    return nullptr;
  }
  return output_devices_[idx];
}

tensorflow::Device* KernelAndDeviceOp::OutputResourceDevice(int idx) const {
  if (kernel_->output_type(idx) == DT_RESOURCE) {
    return device_;
  }
  return nullptr;
}

tensorflow::Device* KernelAndDeviceFunc::OutputResourceDevice(int idx) const {
  if (output_dtypes_[idx] == DT_RESOURCE) {
    return output_devices_[idx];
  }
  return nullptr;
}

Device* KernelAndDeviceOp::InputDevice(int i) const {
  return input_devices_[i];
}

Device* KernelAndDeviceFunc::InputDevice(int i) const {
  if ((input_dtypes_[i] == DT_RESOURCE) &&
      (composite_devices_.find(input_devices_[i]->name()) ==
       composite_devices_.end())) {
    return host_cpu_device_;
  } else {
    return input_devices_[i];
  }
}

}  // namespace tensorflow
