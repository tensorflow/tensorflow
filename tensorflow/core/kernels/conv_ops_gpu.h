/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <tuple>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Get the Dnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64 GetDnnWorkspaceLimit(const string& envvar_in_mb,
                           int64_t default_value_in_bytes);

// A class to provide scratch-space allocator for Stream-Executor Cudnn
// callback. TensorFlow is responsible for releasing the temporary buffers after
// the kernel finishes.
class DnnScratchAllocator : public se::ScratchAllocator {
 public:
  virtual ~DnnScratchAllocator() {}
  DnnScratchAllocator(int64_t memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  int64 GetMemoryLimitInBytes() override { return memory_limit_; }
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64_t byte_size) override {
    Tensor temporary_memory;
    if (byte_size < 0) {
      return se::port::Status{se::port::error::INVALID_ARGUMENT,
                              "Requested negative byte size!"};
    }
    if (byte_size > memory_limit_) {
      return se::port::Status{se::port::error::UNAVAILABLE,
                              absl::StrCat("Requested memory size (", byte_size,
                                           ") exceeds the max memory limit (",
                                           memory_limit_, ").")};
    }
    AllocationAttributes allocation_attr;
    allocation_attr.retry_on_failure = false;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return se::port::Status{
          se::port::error::UNAVAILABLE,
          absl::StrCat("Failed to allocate the requested memory size (",
                       byte_size, ").")};
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return se::port::StatusOr<se::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 memory_limit_;
  int64 total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

typedef Eigen::GpuDevice GPUDevice;

// Select an algorithm for the given convolution, either by running actual
// autotuning with a cache, or by falling back to a default if
// 'cudnn_use_autotune' is true and cuDNN is the statically-chosen DNN backend.
template <typename T>
StatusOr<AutotuneEntry<se::dnn::FusedConvOp>> AutotuneFusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_input_scale,
    double side_input_scale, se::DeviceMemory<T> input_ptr,
    se::DeviceMemory<T> filter_ptr, se::DeviceMemory<T> output_ptr,
    se::DeviceMemory<T> bias_ptr, se::DeviceMemory<T> side_input_ptr,
    int64_t scratch_size);

template <typename T>
StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<T> input_ptr, const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<T> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc, se::DeviceMemory<T> output_ptr,
    int64_t scratch_size_limit);

// Returns a pointer to the primary 'OpRunner' of 'runners' and allocated
// scratch memory if allocatable; else a pointer to its fallback
// no-scratch-space runner, and a null 'DeviceMemoryBase'.
template <typename Sig>
StatusOr<std::tuple<const se::dnn::OpRunner<Sig>*, se::DeviceMemoryBase>>
AllocateScratchOrFallback(se::ScratchAllocator* scratch_allocator,
                          const se::dnn::OpRunner<Sig>* primary,
                          const se::dnn::OpRunner<Sig>* no_scratch_fallback) {
  const se::dnn::OpRunner<Sig>* selected_runner = primary;

  auto workspace_size = selected_runner->GetWorkspaceSize();

  se::DeviceMemoryBase scratch_memory;
  if (workspace_size > 0) {
    auto scratch_or = scratch_allocator->AllocateBytes(workspace_size);
    if (scratch_or.ok()) {
      scratch_memory = scratch_or.ValueOrDie();
    } else if ((selected_runner = no_scratch_fallback)) {
      if (selected_runner->GetWorkspaceSize() > 0) {
        return errors::Internal(
            "No-scratch fallback runner requires nonzero scratch space");
      }
    } else {
      return errors::Unknown(
          "CUDNN failed to allocate the scratch space for the runner or to "
          "find a working no-scratch runner.");
    }
  }

  return std::make_tuple(selected_runner, scratch_memory);
}

template <typename T>
Status LaunchAutotunedConv(const AutotuneEntry<se::dnn::ConvOp>& autotune_entry,
                           DnnScratchAllocator* scratch_allocator,
                           se::dnn::ConvolutionKind kind, se::Stream* stream,
                           const se::dnn::BatchDescriptor& input_desc,
                           se::DeviceMemory<T> in_ptr,
                           const se::dnn::FilterDescriptor& filter_desc,
                           se::DeviceMemory<T> filter_ptr,
                           const se::dnn::ConvolutionDescriptor& conv_desc,
                           const se::dnn::BatchDescriptor& output_desc,
                           se::DeviceMemory<T> out_ptr) {
  if (!autotune_entry.is_algorithm_config()) {
    const auto& runners = autotune_entry.GetOpRunners();
    se::dnn::DataType element_type = se::dnn::ToDataType<T>::value;
    se::dnn::ConvOp::Config config{kind,       element_type, element_type,
                                   input_desc, filter_desc,  output_desc,
                                   conv_desc};
    TF_ASSIGN_OR_RETURN(auto* primary,
                        runners.primary->GetOrCreateRunner(config, stream));

    const se::dnn::ConvRunner* no_scratch_fallback = nullptr;
    if (runners.no_scratch_fallback) {
      TF_ASSIGN_OR_RETURN(
          no_scratch_fallback,
          runners.no_scratch_fallback->GetOrCreateRunner(config, stream));
    }

    TF_ASSIGN_OR_RETURN(auto runner_and_scratch,
                        AllocateScratchOrFallback<se::dnn::ConvOp::Signature>(
                            scratch_allocator, primary, no_scratch_fallback));
    auto& runner = *std::get<const se::dnn::ConvRunner*>(runner_and_scratch);
    return runner(stream, in_ptr, filter_ptr, out_ptr,
                  std::get<se::DeviceMemoryBase>(runner_and_scratch), nullptr);
  } else {
    return stream->ConvolveWithAlgorithm(
        kind, input_desc, in_ptr, filter_desc, filter_ptr, output_desc, out_ptr,
        conv_desc, scratch_allocator, autotune_entry.GetAlgorithmConfig(),
        nullptr);
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
