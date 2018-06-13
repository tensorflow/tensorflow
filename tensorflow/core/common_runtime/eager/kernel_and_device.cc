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
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
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
Status KernelAndDevice::InitOp(Device* device, const NodeDef& ndef,
                               KernelAndDevice* out) {
  OpKernel* k = nullptr;
  Status s = CreateOpKernel(device->device_type().c_str(), device,
                            device->GetAllocator(AllocatorAttributes()),
                            nullptr, ndef, TF_GRAPH_DEF_VERSION, &k);
  out->device_ = device;
  out->kernel_.reset(k);
  out->flib_ = nullptr;
  return s;
}

// static
Status KernelAndDevice::Init(const NodeDef& ndef, FunctionLibraryRuntime* flib,
                             KernelAndDevice* out) {
  OpKernel* k = nullptr;
  Status s = flib->CreateKernel(ndef, &k);
  out->device_ = flib->device();
  out->kernel_.reset(k);
  out->flib_ = flib;
  return s;
}

Status KernelAndDevice::Run(std::vector<Tensor>* input_tensors,
                            std::vector<Tensor>* output_tensors,
                            NodeExecStats* stats) {
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (Tensor& t : *input_tensors) {
    inputs.push_back(TensorValue(&t));
  }

  std::vector<AllocatorAttributes> out_attrs(kernel_->num_outputs());
  for (size_t i = 0; i < out_attrs.size(); ++i) {
    out_attrs[i].set_on_host(kernel_->output_memory_types()[i] ==
                             tensorflow::HOST_MEMORY);
  }

  OpKernelContext::Params params;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.output_attr_array = gtl::vector_as_array(&out_attrs);
  params.function_library = flib_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendez_;
  params.cancellation_manager = &cm_;
  if (stats != nullptr) {
    params.track_allocations = true;
  }
  // TODO(apassos): use a thread pool.
  std::function<void(std::function<void()>)> runner =
      [](std::function<void()> f) { f(); };
  params.runner = &runner;

  ScopedStepContainer step_container(0, [this](const string& name) {
    device_->resource_manager()->Cleanup(name).IgnoreError();
  });
  params.step_container = &step_container;

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

  output_tensors->clear();
  for (int i = 0; i < context.num_outputs(); ++i) {
    output_tensors->push_back(Tensor(*context.mutable_output(i)));
  }
  if (stats != nullptr) {
    for (const auto& allocator_pair : context.wrapped_allocators()) {
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
  }
  return Status::OK();
}

}  // namespace tensorflow
