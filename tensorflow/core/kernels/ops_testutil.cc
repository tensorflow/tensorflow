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

#include "tensorflow/core/framework/node_properties.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace test {

void SetOutputAttrs(OpKernelContext::Params* params,
                    std::vector<AllocatorAttributes>* attrs) {
  attrs->clear();
  for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    attrs->push_back(attr);
  }
  params->output_attr_array = attrs->data();
}

}  // namespace test

OpsTestBase::OpsTestBase() : device_type_(DEVICE_CPU) {
  auto device = DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
  CHECK(device) << "Could not create CPU device";

  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), /*name=*/"default", /*num_threads=*/1);

  device_ = device.get();
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(device));

  allocator_ = device_->GetAllocator(AllocatorAttributes());

  flib_def_ = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), FunctionDefLibrary{});
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions());
}

OpsTestBase::~OpsTestBase() {
  for (auto& temp : tensors_) {
    delete temp;
  }
  for (auto& temp : managed_outputs_) {
    delete temp;
  }
  tensors_.clear();
  managed_outputs_.clear();
  context_.reset(nullptr);
  params_.reset(nullptr);
}

void OpsTestBase::SetDevice(const DeviceType& device_type,
                            std::unique_ptr<Device> device) {
  CHECK(device_) << "No device provided";

  device_ = device.get();
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(device));
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions(),
      thread_pool_.get());

  device_type_ = device_type;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (device_type == DEVICE_GPU) {
    managed_allocator_.reset(new GpuManagedAllocator());
    allocator_ = managed_allocator_.get();
  } else {
    managed_allocator_.reset();
    allocator_ = device_->GetAllocator(AllocatorAttributes());
  }
#else
  CHECK_NE(device_type, DEVICE_GPU)
      << "Requesting GPU on binary compiled without GOOGLE_CUDA or "
         "TENSORFLOW_USE_ROCM.";
  allocator_ = device_->GetAllocator(AllocatorAttributes());
#endif
}

void OpsTestBase::set_node_def(const NodeDef& node_def) {
  node_def_.CopyFrom(node_def);
}

NodeDef* OpsTestBase::node_def() { return &node_def_; }

Status OpsTestBase::InitOp() {
  return InitOpWithGraphVersion(TF_GRAPH_DEF_VERSION);
}

Status OpsTestBase::InitOpWithGraphVersion(int graph_def_version) {
  std::shared_ptr<const NodeProperties> props;
  TF_RETURN_IF_ERROR(NodeProperties::CreateFromNodeDef(
      node_def_, OpRegistry::Global(), &props));
  OpKernel* kernel;
  TF_RETURN_IF_ERROR(CreateOpKernel(
      device_type_, device_, allocator(), /*flib=*/nullptr,
      device_->resource_manager(), props, graph_def_version, &kernel));
  kernel_.reset(kernel);
  input_types_ = kernel_->input_types();
  return Status::OK();
}

Status OpsTestBase::RunOpKernel() {
  // Make sure the old OpKernelContext is deleted before the Params
  // it was using.
  context_.reset(nullptr);

  // Delete the output copies from previous runs.
  for (auto& temp : managed_outputs_) {
    delete temp;
  }
  managed_outputs_.clear();
  managed_outputs_.resize(0);

  params_.reset(new OpKernelContext::Params);
  params_->device = device_;
  params_->frame_iter = FrameAndIter(0, 0);
  params_->inputs = &inputs_;
  params_->op_kernel = kernel_.get();
  step_container_.reset(new ScopedStepContainer(0, [](const string&) {}));
  params_->step_container = step_container_.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(params_.get(), &attrs);
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
  params_->slice_reader_cache = &slice_reader_cache_wrapper;
  params_->resource_manager = device_->resource_manager();
  params_->function_library = pflr_->GetFLR(device_->name());

  context_.reset(new OpKernelContext(params_.get()));
  device_->Compute(kernel_.get(), context_.get());
  return context_->status();
}

const Tensor& OpsTestBase::GetInput(int input_index) const {
  CHECK_LT(input_index, context_->num_inputs());
  CHECK(!IsRefType(context_->input_dtype(input_index)));
  return context_->input(input_index);
}

TensorValue OpsTestBase::mutable_input(int input_index) {
  CHECK_LT(input_index, inputs_.size());
  return inputs_[input_index];
}

Tensor* OpsTestBase::GetOutput(int output_index) {
  CHECK_LT(output_index, context_->num_outputs());
  Tensor* output = context_->mutable_output(output_index);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (device_type_ == DEVICE_GPU) {
    managed_outputs_.resize(context_->num_outputs());
    // Copy the output tensor to managed memory if we haven't done so.
    if (!managed_outputs_[output_index]) {
      Tensor* managed_output =
          new Tensor(allocator(), output->dtype(), output->shape());
      auto src = output->tensor_data();
      auto dst = managed_output->tensor_data();
      context_->eigen_gpu_device().memcpyDeviceToHost(
          const_cast<char*>(dst.data()), src.data(), src.size());
      context_->eigen_gpu_device().synchronize();
      managed_outputs_[output_index] = managed_output;
    }
    output = managed_outputs_[output_index];
  }
#endif
  return output;
}

Allocator* OpsTestBase::allocator() { return allocator_; }

OpKernel* OpsTestBase::op_kernel() { return kernel_.get(); }

const DataTypeVector& OpsTestBase::output_types() const {
  return kernel_->output_types();
}

Tensor* OpsTestBase::AddInput(DataType dtype, const TensorShape& shape) {
  CHECK_GT(input_types_.size(), inputs_.size())
      << "Adding more inputs than types; perhaps you need to call MakeOp";
  bool is_ref = IsRefType(input_types_[inputs_.size()]);
  Tensor* input = new Tensor(allocator(), dtype, shape);
  tensors_.push_back(input);
  if (is_ref) {
    CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]), dtype);
    inputs_.push_back({&lock_for_refs_, input});
  } else {
    CHECK_EQ(input_types_[inputs_.size()], dtype);
    inputs_.push_back({nullptr, input});
  }
  return input;
}

void OpsTestBase::AddResourceInputInternal(const std::string& container_name,
                                           const std::string& name,
                                           const TypeIndex& type_index) {
  ResourceHandle handle;
  handle.set_device(device_->name());
  handle.set_container(container_name);
  handle.set_name(name);
  handle.set_hash_code(type_index.hash_code());
  handle.set_maybe_type_name(type_index.name());
  Tensor* input = new Tensor(allocator(), DT_RESOURCE, TensorShape({}));
  input->scalar<ResourceHandle>()() = handle;
  tensors_.push_back(input);
  inputs_.push_back({nullptr, input});
}

}  // namespace tensorflow
