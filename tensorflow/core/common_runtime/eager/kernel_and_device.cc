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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

// static
Status KernelAndDevice::Init(const NodeDef& ndef, FunctionLibraryRuntime* flr,
                             std::function<void(std::function<void()>)>* runner,
                             KernelAndDevice* out) {
  OpKernel* k = nullptr;
  TF_RETURN_IF_ERROR(flr->CreateKernel(ndef, &k));
  out->device_ = flr->device();
  out->kernel_.reset(k);
  out->flr_ = flr;
  out->runner_ = runner;
  out->default_runner_ = [](std::function<void()> f) { f(); };

  // Update output_dtypes_.
  const OpDef* op_def = nullptr;
  const FunctionDef* function_def =
      flr->GetFunctionLibraryDefinition()->Find(ndef.op());
  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(ndef.op().c_str(), &op_def));
  }
  return OutputTypesForNode(ndef, *op_def, &out->output_dtypes_);
}

Status KernelAndDevice::Run(const gtl::InlinedVector<TensorValue, 4>& inputs,
                            std::vector<Tensor>* outputs, NodeExecStats* stats,
                            StepStats* step_stats,
                            GraphCollector* graph_collector) {
  ScopedStepContainer step_container(0, [this](const string& name) {
    device_->resource_manager()->Cleanup(name).IgnoreError();
  });
  return this->Run(&step_container, inputs, outputs, stats, step_stats,
                   graph_collector);
}

Status KernelAndDevice::Run(ScopedStepContainer* step_container,
                            const gtl::InlinedVector<TensorValue, 4>& inputs,
                            std::vector<Tensor>* outputs, NodeExecStats* stats,
                            StepStats* step_stats,
                            GraphCollector* graph_collector) {
  std::vector<AllocatorAttributes> out_attrs(kernel_->num_outputs());
  for (size_t i = 0; i < out_attrs.size(); ++i) {
    out_attrs[i].set_on_host(kernel_->output_memory_types()[i] ==
                             tensorflow::HOST_MEMORY);
  }

  gtl::InlinedVector<DeviceContext*, 4> input_device_contexts;
  for (int i = 0; i < inputs.size(); i++) {
    DeviceContext* device_context = nullptr;
    if (device_->tensorflow_gpu_device_info() != nullptr) {
      device_context = device_->tensorflow_gpu_device_info()->default_context;
    }
    input_device_contexts.push_back(device_context);
  }

  OpKernelContext::Params params;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.output_attr_array = gtl::vector_as_array(&out_attrs);
  params.function_library = flr_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendez_;
  params.cancellation_manager = &cm_;
  params.log_memory = log_memory_;
  std::unique_ptr<StepStatsCollector> step_stats_collector;
  if (stats != nullptr) {
    step_stats_collector.reset(new StepStatsCollector(step_stats));
    params.track_allocations = true;
    params.stats_collector = step_stats_collector.get();
    params.graph_collector = graph_collector;
  }
  if (runner_ == nullptr) {
    params.runner = &default_runner_;
  } else {
    params.runner = runner_;
  }

  params.step_container = step_container;
  params.collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;
  params.input_device_contexts = &input_device_contexts;

  OpKernelContext context(&params);

  if (kernel_->def().op() == "_Recv") {
    // TODO(apassos) do not special-case _Recv. Currently the GPU device fails
    // if trying to run _Recv->Compute(), specifically checking for _Recv. To go
    // around this we call _Recv->ComputeAsync, to mimic graph mode behavior.
    AsyncOpKernel* async = kernel_->AsAsync();
    Notification done;
    device_->ComputeAsync(async, &context, [&done]() { done.Notify(); });
    done.WaitForNotification();
  } else {
    device_->Compute(kernel_.get(), &context);
  }
  if (!context.status().ok()) return context.status();

  outputs->clear();
  for (int i = 0; i < context.num_outputs(); ++i) {
    outputs->push_back(Tensor(*context.mutable_output(i)));
  }
  if (stats != nullptr) {
    for (const auto& allocator_pair : context.ConsumeWrappedAllocators()) {
      AllocatorMemoryUsed* memory = stats->add_memory();
      memory->set_allocator_name(allocator_pair.first->Name());
      auto sizes = allocator_pair.second->GetSizes();
      memory->set_total_bytes(std::get<0>(sizes));
      memory->set_peak_bytes(std::get<1>(sizes));
      memory->set_live_bytes(std::get<2>(sizes));

      AllocatorStats allocator_stats;
      allocator_pair.first->GetStats(&allocator_stats);
      memory->set_allocator_bytes_in_use(allocator_stats.bytes_in_use);
      allocator_pair.second->GetRecordsAndUnRef();
    }
    auto* ms = stats->mutable_memory_stats();
    ms->set_temp_memory_size(context.temp_memory_allocated());
    for (const auto& alloc_id : context.persistent_alloc_ids()) {
      ms->mutable_persistent_tensor_alloc_ids()->Add(alloc_id);
    }

    ms->set_persistent_memory_size(context.persistent_memory_allocated());
    step_stats_collector->Finalize();
  }
  return Status::OK();
}

tensorflow::Device* KernelAndDevice::OutputDevice(int idx) const {
  if (device_ != nullptr &&
      kernel_->output_memory_types()[idx] == HOST_MEMORY) {
    return nullptr;
  }
  return device_;
}

}  // namespace tensorflow
