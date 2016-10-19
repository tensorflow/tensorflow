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
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// TODO(zhengxq): move this to gpu_util.h. The use of such wrappers is wide
// spread.
template <typename T>
inline perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                           uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}

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
  virtual int64 GetMemoryLimitInBytes(
      perftools::gputools::Stream* stream) override {
    return memory_limit_;
  }
  virtual perftools::gputools::port::StatusOr<
      perftools::gputools::DeviceMemory<uint8>>
  AllocateBytes(perftools::gputools::Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;
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
  ConvParameters(int64 batch, int64 in_depths, int64 in_rows, int64 in_cols,
                 int64 out_depths, int64 filter_rows, int64 filter_cols,
                 int64 stride_rows, int64 stride_cols, int64 padding_rows,
                 int64 padding_cols, int device_id)
      : batch_(batch),
        in_depths_(in_depths),
        in_rows_(in_rows),
        in_cols_(in_cols),
        out_depths_(out_depths),
        filter_rows_(filter_rows),
        filter_cols_(filter_cols),
        stride_rows_(stride_rows),
        stride_cols_(stride_cols),
        padding_rows_(padding_rows),
        padding_cols_(padding_cols),
        device_id_(device_id) {
    hash_code_ = batch;
    hash_code_ = Hash64Combine(hash_code_, in_depths);
    hash_code_ = Hash64Combine(hash_code_, in_rows);
    hash_code_ = Hash64Combine(hash_code_, in_cols);
    hash_code_ = Hash64Combine(hash_code_, out_depths);
    hash_code_ = Hash64Combine(hash_code_, filter_rows);
    hash_code_ = Hash64Combine(hash_code_, filter_cols);
    hash_code_ = Hash64Combine(hash_code_, stride_rows);
    hash_code_ = Hash64Combine(hash_code_, stride_cols);
    hash_code_ = Hash64Combine(hash_code_, padding_rows);
    hash_code_ = Hash64Combine(hash_code_, padding_cols);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const ConvParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const ConvParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

 private:
  typedef std::tuple<int64, int64, int64, int64, int64, int64, int64, int64,
                     int64, int64, int64, int>
      DataType;

  DataType get_data_as_tuple() const {
    return std::make_tuple(batch_, in_depths_, in_rows_, in_cols_, out_depths_,
                           filter_rows_, filter_cols_, stride_rows_,
                           stride_cols_, padding_rows_, padding_cols_,
                           device_id_);
  }

  int64 batch_;
  int64 in_depths_;
  int64 in_rows_;
  int64 in_cols_;
  int64 out_depths_;
  int64 filter_rows_;
  int64 filter_cols_;
  int64 stride_rows_;
  int64 stride_cols_;
  int64 padding_rows_;
  int64 padding_cols_;
  int device_id_;
  uint64 hash_code_;
};

typedef Eigen::GpuDevice GPUDevice;

// A helper class that looks up the best autotuned config from parameters.
template <typename Parameters, typename Config>
class AutoTuneMap {
 public:
  bool Find(const Parameters& params, Config* config) const {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    if (iter == params_config_map_.end()) {
      return false;
    }
    *config = iter->second;
    return true;
  }
  void Insert(const ConvParameters& params, const Config& config) {
    mutex_lock lock(mu_);
    params_config_map_[params] = config;
  }

 private:
  AutoTuneMap() {}

  template <class Group, class Params, class Cfg>
  friend class AutoTuneSingleton;

  struct Hasher {
    std::size_t operator()(const Parameters& parameter) const {
      return parameter.hash();
    }
  };
  mutable mutex mu_;
  std::unordered_map<Parameters, Config, Hasher> params_config_map_
      GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(AutoTuneMap);
};

// A Singleton helper that manages the global autotune results by groups.
// The caller specified arbitrary Group type that can distinguish between
// different autotune results, even if their Parameters and Configs are the
// same.
template <class Group, typename Parameters, typename Config>
class AutoTuneSingleton {
 public:
  typedef AutoTuneMap<Parameters, Config> AutoTuneType;
  static AutoTuneType* GetInstance() {
    static AutoTuneType* instance = new AutoTuneType;
    return instance;
  }
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
