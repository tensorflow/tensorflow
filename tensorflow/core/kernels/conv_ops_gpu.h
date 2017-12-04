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

#if GOOGLE_CUDA

#include <tuple>
#include <unordered_map>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {


// Get the Cudnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64 GetCudnnWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes);

// A class to provide scratch-space allocator for Stream-Executor Cudnn
// callback. TensorFlow is responsible for releasing the temporary buffers after
// the kernel finishes.
class CudnnScratchAllocator : public perftools::gputools::ScratchAllocator {
 public:
  virtual ~CudnnScratchAllocator() {}
  CudnnScratchAllocator(int64 memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  int64 GetMemoryLimitInBytes(perftools::gputools::Stream* stream) override {
    return memory_limit_;
  }
  perftools::gputools::port::StatusOr<perftools::gputools::DeviceMemory<uint8>>
  AllocateBytes(perftools::gputools::Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;
    if (byte_size < 0) {
      return perftools::gputools::port::Status{
          perftools::gputools::port::error::INVALID_ARGUMENT,
          "Requested negative byte size!"};
    }
    if (byte_size > memory_limit_) {
      return perftools::gputools::port::StatusOr<
          perftools::gputools::DeviceMemory<uint8>>();
    }
    AllocationAttributes allocation_attr;
    allocation_attr.no_retry_on_failure = true;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return perftools::gputools::port::StatusOr<
          perftools::gputools::DeviceMemory<uint8>>();
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return perftools::gputools::port::StatusOr<
        perftools::gputools::DeviceMemory<uint8>>(
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

// Encapsulate all the shape information that is used in both forward and
// backward conv operations.
class ConvParameters {
 public:
  using SpatialArray = gtl::InlinedVector<int64, 3>;
  ConvParameters(int64 batch, int64 in_depths, const SpatialArray& in,
                 int64 out_depths, const SpatialArray& filter,
                 const SpatialArray& dilation, const SpatialArray& stride,
                 const SpatialArray& padding, DataType dtype, int device_id)
      : batch_(batch),
        in_depths_(in_depths),
        out_depths_(out_depths),
        in_(in),
        filter_(filter),
        dilation_(dilation),
        stride_(stride),
        padding_(padding),
        dtype_(dtype),
        device_id_(device_id) {
    hash_code_ = batch;
    hash_code_ = Hash64Combine(hash_code_, in_depths);
    for (int64 val : in) hash_code_ = Hash64Combine(hash_code_, val);
    hash_code_ = Hash64Combine(hash_code_, out_depths);
    for (int64 val : filter) hash_code_ = Hash64Combine(hash_code_, val);
    for (int64 val : dilation) hash_code_ = Hash64Combine(hash_code_, val);
    for (int64 val : stride) hash_code_ = Hash64Combine(hash_code_, val);
    for (int64 val : padding) hash_code_ = Hash64Combine(hash_code_, val);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const ConvParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const ConvParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    // clang-format off
    return strings::StrCat(
        batch_, ", ", in_depths_, ", ",
        "(", str_util::Join(in_, ", "), "), ",
        out_depths_, ", ",
        "(", str_util::Join(filter_, ", "), "), ",
        "(", str_util::Join(dilation_, ", "), "), ",
        "(", str_util::Join(stride_, ", "), "), ",
        "(", str_util::Join(padding_, ", "), "), ",
        dtype_, ", ",
        device_id_);
    // clang-format on
  }

  // TODO(yangzihao): The purpose of this function is to disable winograd
  // nonfused conv algorithm for certain input parameters so as to avoid a bug
  // in cuDNNv5 and cuDNNv6. Remove this once switch to cuDNNv7.
  template <typename T>
  bool ShouldIncludeWinogradNonfusedAlgo() const {
    int64 total_size = 16 * std::ceil(batch_ / 16.0) *
                       std::max(in_depths_, out_depths_) * in_[0] * in_[1] *
                       sizeof(T);
    int64 threshold = 1L << 31;
    if (total_size >= threshold) {
      return false;
    } else {
      return true;
    }
  }

 protected:
  using ParameterDataType =
      std::tuple<int64, int64, SpatialArray, int64, SpatialArray, SpatialArray,
                 SpatialArray, SpatialArray, DataType, int>;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(batch_, in_depths_, in_, out_depths_, filter_,
                           dilation_, stride_, padding_, dtype_, device_id_);
  }

  uint64 hash_code_;

 private:
  int64 batch_;
  int64 in_depths_;
  int64 out_depths_;
  SpatialArray in_;
  SpatialArray filter_;
  SpatialArray dilation_;
  SpatialArray stride_;
  SpatialArray padding_;
  DataType dtype_;
  int device_id_;
};

typedef Eigen::GpuDevice GPUDevice;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
