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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#endif

#include "tensorflow/core/kernels/ops_testutil.h"

namespace tensorflow {

void OpsTestBase::SetDevice(const DeviceType& device_type,
                            std::unique_ptr<Device> device) {
  CHECK(device_) << "No device provided";

  device_ = device.get();
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(device));
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions());

  device_type_ = device_type;
#ifdef GOOGLE_CUDA
  if (device_type == DEVICE_GPU) {
    managed_allocator_.reset(new GpuManagedAllocator());
    allocator_ = managed_allocator_.get();
  } else {
    managed_allocator_.reset();
    allocator_ = device_->GetAllocator(AllocatorAttributes());
  }
#else
  CHECK_NE(device_type, DEVICE_GPU)
      << "Requesting GPU on binary compiled without GOOGLE_CUDA.";
  allocator_ = device_->GetAllocator(AllocatorAttributes());
#endif
}

Tensor* OpsTestBase::GetOutput(int output_index) {
  CHECK_LT(output_index, context_->num_outputs());
  Tensor* output = context_->mutable_output(output_index);
#ifdef GOOGLE_CUDA
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

}  // namespace tensorflow
