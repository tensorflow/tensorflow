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
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Get the Dnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64 GetDnnWorkspaceLimit(const string& envvar_in_mb,
                           int64 default_value_in_bytes);

// A class to provide scratch-space allocator for Stream-Executor Cudnn
// callback. TensorFlow is responsible for releasing the temporary buffers after
// the kernel finishes.
class DnnScratchAllocator : public se::ScratchAllocator {
 public:
  virtual ~DnnScratchAllocator() {}
  DnnScratchAllocator(int64 memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  int64 GetMemoryLimitInBytes() override { return memory_limit_; }
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64 byte_size) override {
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

// Encapsulate all the shape information that is used in both forward and
// backward conv operations.
class ConvParameters {
 public:
  using SpatialArray = gtl::InlinedVector<int64, 3>;
  ConvParameters(int64 batch, int64 in_depths, const SpatialArray& in,
                 TensorFormat data_format, int64 out_depths,
                 const SpatialArray& filter, const SpatialArray& dilation,
                 const SpatialArray& stride, const SpatialArray& padding,
                 DataType dtype, int device_id, int group_count = 1)
      : batch_(batch),
        in_depths_(in_depths),
        out_depths_(out_depths),
        in_(CheckSpatialArraySize(in)),
        data_format_(data_format),
        filter_(CheckSpatialArraySize(filter)),
        dilation_(CheckSpatialArraySize(dilation)),
        stride_(CheckSpatialArraySize(stride)),
        padding_(CheckSpatialArraySize(padding)),
        dtype_(dtype),
        device_id_(device_id),
        group_count_(group_count) {
    hash_code_ = batch;
    hash_code_ = Hash64Combine(hash_code_, in_depths);
    for (int64 val : in) hash_code_ = Hash64Combine(hash_code_, val);
    hash_code_ = Hash64Combine(hash_code_, data_format);
    hash_code_ = Hash64Combine(hash_code_, out_depths);
    for (int64 val : filter) hash_code_ = Hash64Combine(hash_code_, val);
    for (int64 val : dilation) hash_code_ = Hash64Combine(hash_code_, val);
    for (int64 val : stride) hash_code_ = Hash64Combine(hash_code_, val);
    for (int64 val : padding) hash_code_ = Hash64Combine(hash_code_, val);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
    hash_code_ = Hash64Combine(hash_code_, group_count);
  }

  bool operator==(const ConvParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const ConvParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    return absl::StrFormat(
        "batch=%d, in_depth=%d, in=(%s), out_depth=%d, filter=(%s), "
        "%s, %s, dilation=(%s), stride=(%s), padding=(%s), group_count=%d, "
        "device_id=%d",
        batch_, in_depths_, absl::StrJoin(in_, ","), out_depths_,
        absl::StrJoin(filter_, ","), DataTypeString(dtype_),
        ::tensorflow::ToString(data_format_), absl::StrJoin(dilation_, ","),
        absl::StrJoin(stride_, ","), absl::StrJoin(padding_, ","), group_count_,
        device_id_);
  }

 protected:
  using ParameterDataType =
      std::tuple<int64, int64, SpatialArray, TensorFormat, int64, SpatialArray,
                 SpatialArray, SpatialArray, SpatialArray, DataType, int, int>;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(batch_, in_depths_, in_, data_format_, out_depths_,
                           filter_, dilation_, stride_, padding_, dtype_,
                           device_id_, group_count_);
  }

  uint64 hash_code_;

 private:
  static const SpatialArray& CheckSpatialArraySize(const SpatialArray& array) {
    CHECK_LE(array.size(), 3);  // Catch corruptions related to b/124313574.
    return array;
  }

  int64 batch_;
  int64 in_depths_;
  int64 out_depths_;
  SpatialArray in_;
  TensorFormat data_format_;
  SpatialArray filter_;
  SpatialArray dilation_;
  SpatialArray stride_;
  SpatialArray padding_;
  DataType dtype_;
  int device_id_;
  int group_count_;
};

typedef Eigen::GpuDevice GPUDevice;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
