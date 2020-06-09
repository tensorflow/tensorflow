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

#include <memory>

#include "absl/strings/match.h"
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
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if !defined(IS_MOBILE_PLATFORM)
#if !defined(PLATFORM_WINDOWS)
#include "tensorflow/compiler/jit/xla_kernel_creator_util.h"
#endif  // !PLATFORM_WINDOWS
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {

Status EagerKernelArgs::GetLocalArg(const FunctionArgIndex& index,
                                    Tensor* val) const {
  if (index.sub_index >= 0) {
    return errors::InvalidArgument("Got unexpected sub_index ", index.sub_index,
                                   " for argument ", index.index);
  }
  Tensor* arg = tensor_args_.at(index.index).tensor;
  if (arg) {
    *val = *arg;
    return Status::OK();
  } else {
    return errors::NotFound("Argument ", index.index, " has no local tensor.");
  }
}

std::vector<Tensor> EagerKernelArgs::GetLocalTensors() const {
  std::vector<Tensor> lcoal_inputs;
  lcoal_inputs.reserve(tensor_args_.size());
  for (const TensorValue& tensor_value : tensor_args_) {
    lcoal_inputs.push_back(*tensor_value.tensor);
  }
  return lcoal_inputs;
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
    Status status = pflr_->ReleaseHandle(handle_);
    if (!status.ok()) {
      LOG(INFO) << "Ignoring error status when releasing multi-device function "
                   "handle "
                << status.ToString();
    }
  }
}

Status KernelAndDeviceOp::Init(const Context& ctx, const NodeDef& ndef,
                               GraphCollector* graph_collector) {
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

  input_alloc_attrs_.resize(kernel_->num_inputs());
  input_devices_.resize(kernel_->num_inputs(), device_);
  for (size_t i = 0; i < input_alloc_attrs_.size(); ++i) {
    bool host = kernel_->input_memory_types()[i] == tensorflow::HOST_MEMORY;
    input_alloc_attrs_[i].set_on_host(host);
    if (host) {
      input_devices_[i] = host_cpu_device_;
    }
  }
  output_alloc_attrs_.resize(kernel_->num_outputs());
  for (size_t i = 0; i < output_alloc_attrs_.size(); ++i) {
    output_alloc_attrs_[i].set_on_host(kernel_->output_memory_types()[i] ==
                                       tensorflow::HOST_MEMORY);
  }

  return Status::OK();
}

Status KernelAndDeviceFunc::InstantiateFunc(const Context& ctx,
                                            const NodeDef& ndef,
                                            GraphCollector* graph_collector) {
  const OpDef* op_def = nullptr;
  const FunctionDef* function_def;
  if (flr_ == nullptr) {
    // If function is being executed without an explicit device request,
    // lookup the FunctionDef in the CPU's FLR. All FLRs share the same
    // library.
    function_def = pflr_->GetFLR(host_cpu_device_->name())
                       ->GetFunctionLibraryDefinition()
                       ->Find(ndef.op());
  } else {
    function_def = flr_->GetFunctionLibraryDefinition()->Find(ndef.op());
  }

  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(ndef.op(), &op_def));
  }
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(ndef, *op_def, &input_dtypes_, &output_dtypes_));

  FunctionLibraryRuntime::InstantiateOptions options;
  options.target = device_ == nullptr ? "" : device_->name();
  options.is_multi_device_function = true;
  for (const Device* device : input_devices_) {
    options.input_devices.push_back(device->name());
  }
  options.composite_devices = composite_devices_;
  options.input_resource_dtypes_and_shapes = input_resource_dtypes_and_shapes_;

  const auto& it = ndef.attr().find("executor_type");
  if (it != ndef.attr().end()) {
    options.executor_type = it->second.s();
  }
#if !defined(IS_MOBILE_PLATFORM)
  // Android tf library does not include grappler.
  const auto& config_it = ndef.attr().find("config_proto");
  if (it != ndef.attr().end()) {
    if (!options.config_proto.ParseFromString(config_it->second.s())) {
      return errors::InvalidArgument(
          "Failed to parse config_proto attribute as tensorflow::ConfigProto "
          "proto.");
    }
    grappler::GrapplerItem::OptimizationOptions optimization_options;

    // Tensorflow 2.0 in eager mode with automatic control dependencies will
    // prune all nodes that are not in the transitive fanin of the fetch nodes.
    // However because the function will be executed via FunctionLibraryRuntime,
    // and current function implementation does not prune stateful and dataset
    // ops, we rely on Grappler to do the correct graph pruning.
    optimization_options.allow_pruning_stateful_and_dataset_ops = true;

    optimization_options.is_eager_mode = true;

    // All the nested function calls will be executed and optimized via
    // PartitionedCallOp, there is no need to optimize functions now.
    optimization_options.optimize_function_library = false;

    options.optimize_graph_fn = std::bind(
        grappler::OptimizeGraph, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        options.config_proto, function_def->signature().name(),
        optimization_options, std::placeholders::_6);
  }
#endif  // !IS_MOBILE_PLATFORM
  options.graph_collector = graph_collector;

  // In Eager mode we always inline all functions into the top-level
  // function body graph, to get a single executable graph, that could be
  // optimized across function boundaries (e.g. prune unused inputs and outputs
  // in a function call chain). This is required to mimic graph mode execution,
  // with aggressive pruning of nodes not in the transitive fanin of fetches.
  options.config_proto.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);

  options.config_proto.set_log_device_placement(ctx.log_device_placement);

  TF_RETURN_IF_ERROR(
      pflr_->Instantiate(ndef.op(), AttrSlice(ndef), options, &handle_));
  return pflr_->IsCrossProcess(handle_, &is_cross_process_);
}

Status KernelAndDeviceFunc::Init(const Context& ctx, const NodeDef& ndef,
                                 GraphCollector* graph_collector) {
  TF_RETURN_IF_ERROR(InstantiateFunc(ctx, ndef, graph_collector));
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

Status KernelAndDeviceOp::Run(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<Tensor>* outputs, CancellationManager* cancellation_manager,
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params) {
  OpKernelContext::Params params;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = inputs.GetTensorValues();
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.input_alloc_attrs = &input_alloc_attrs_;
  params.output_attr_array = output_alloc_attrs_.data();
  params.function_library = flr_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendezvous_;
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

  params.step_container =
      step_container == nullptr ? &step_container_ : step_container;
  auto step_container_cleanup = gtl::MakeCleanup([step_container, this] {
    if (step_container == nullptr) {
      this->step_container_.CleanUp();
    }
  });

  params.collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;

  OpKernelContext context(&params);

  {
    port::ScopedFlushDenormal flush;
    port::ScopedSetRound round(FE_TONEAREST);
    // 'AnnotatedTraceMe' will trace both scheduling time on host and execution
    // time on device of the OpKernel.
    profiler::AnnotatedTraceMe activity(
        [&] { return kernel_->TraceString(&context, /*verbose=*/false); },
        profiler::TraceMeLevel::kInfo);
    device_->Compute(kernel_.get(), &context);
  }

  // Clean up execution op_execution_state if deferred ops aren't running.
  if (op_execution_state != nullptr) {
    op_execution_state->Unref();
  }

  if (!context.status().ok()) return context.status();

  if (outputs != nullptr) {
    outputs->clear();
    for (int i = 0; i < context.num_outputs(); ++i) {
      outputs->push_back(Tensor(*context.mutable_output(i)));
    }
  }
  return Status::OK();
}

Status KernelAndDeviceFunc::Run(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<Tensor>* outputs, CancellationManager* cancellation_manager,
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params) {
  Notification n;
  Status status;
  RunAsync(step_container, inputs, outputs, cancellation_manager,
           remote_func_params, [&status, &n](const Status& s) {
             status = s;
             n.Notify();
           });
  n.WaitForNotification();
  return status;
}

void KernelAndDeviceFunc::RunAsync(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<Tensor>* outputs, CancellationManager* cancellation_manager,
    const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
    std::function<void(const Status&)> done) {
  std::shared_ptr<FunctionLibraryRuntime::Options> opts = nullptr;
  if (remote_func_params.has_value()) {
    const EagerRemoteFunctionParams& params = remote_func_params.value();
    if (params.step_id.has_value()) {
      // If the function is a remote component of a cross-process function,
      // re-use the step id as its parent function's.
      opts = std::make_shared<FunctionLibraryRuntime::Options>(
          params.step_id.value());
    } else {
      opts = std::make_shared<FunctionLibraryRuntime::Options>();
    }
    // Reuse the op id if it exists.
    opts->op_id = params.op_id;
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
  Rendezvous* rendezvous = rendezvous_creator_(opts->step_id);
  opts->rendezvous = rendezvous;
  opts->create_rendezvous = false;

  // Create a cancellation manager to be used by FLR options if caller does not
  // pass in one. If the caller does provide one, pass it to process FLR and the
  // locally created one will be unused.
  std::shared_ptr<CancellationManager> local_cm;
  if (cancellation_manager) {
    opts->cancellation_manager = cancellation_manager;
  } else {
    local_cm = std::make_shared<CancellationManager>();
    opts->cancellation_manager = local_cm.get();
  }
  opts->allow_dead_tensors = true;
  opts->step_container =
      step_container == nullptr ? &step_container_ : step_container;
  opts->collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;

  opts->stats_collector = nullptr;
  opts->runner = get_runner();

  outputs->clear();

  profiler::TraceMeProducer activity(
      // To TraceMeConsumers in ExecutorState::Process/Finish.
      [&] {
        return profiler::TraceMeEncode(
            "FunctionRun", {{"id", opts->step_id}, {"$r", 1} /*root_event*/});
      },
      profiler::ContextType::kTfExecutor, opts->step_id,
      profiler::TraceMeLevel::kInfo);
  pflr_->Run(*opts, handle_, inputs, outputs,
             [opts, rendezvous, local_cm, step_container, this,
              done = std::move(done)](const Status& s) {
               rendezvous->Unref();
               if (step_container == nullptr) {
                 this->step_container_.CleanUp();
               }
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
