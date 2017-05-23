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
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
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
  using SpatialArray = gtl::InlinedVector<int64, 3>;
  ConvParameters(int64 batch, int64 in_depths, const SpatialArray& in,
                 int64 out_depths, const SpatialArray& filter,
                 const SpatialArray& stride, const SpatialArray& padding,
                 const DataType& dtype, int device_id)
      : batch_(batch),
        in_depths_(in_depths),
        in_(in),
        out_depths_(out_depths),
        filter_(filter),
        stride_(stride),
        padding_(padding),
        dtype_(dtype),
        device_id_(device_id) {
    hash_code_ = batch;
    hash_code_ = Hash64Combine(hash_code_, in_depths);
    for (int64 val : in) hash_code_ = Hash64Combine(hash_code_, val);
    hash_code_ = Hash64Combine(hash_code_, out_depths);
    for (int64 val : filter) hash_code_ = Hash64Combine(hash_code_, val);
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
        "(", str_util::Join(stride_, ", "), "), ",
        "(", str_util::Join(padding_, ", "), "), ",
        dtype_, ", ", device_id_);
    // clang-format on
  }

 private:
  typedef std::tuple<int64, int64, SpatialArray, int64, SpatialArray,
                     SpatialArray, SpatialArray, DataType, int>
      ParameterDataType;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(batch_, in_depths_, in_, out_depths_, filter_,
                           stride_, padding_, dtype_, device_id_);
  }

  int64 batch_;
  int64 in_depths_;
  SpatialArray in_;
  int64 out_depths_;
  SpatialArray filter_;
  SpatialArray stride_;
  SpatialArray padding_;
  DataType dtype_;
  int device_id_;
  uint64 hash_code_;
};

typedef Eigen::GpuDevice GPUDevice;

// A helper class that looks up the best autotuned config from parameters.
// Due to the noisy nature of autotune, especially with multiple devices, it
// only accepts a config if its margin exceeds a threshold.
// For the same shape configs, if a new best config matches the previous best,
// they get promoted; otherwise, the winner gets demoted. This process stops
// when the winner's score exceeds the threshold.
// In a bad case when two configs are very close to each other and flips
// back and forth randomly, the expected number of experiments before autotune
// settles is O(threshold ^ 2). So we recommend that number of warmup runs
// for any benchmarks.
template <typename Parameters, typename Config>
class AutoTuneMap {
 public:
  bool Find(const Parameters& params, Config* config) const {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    if (iter == params_config_map_.end() ||
        iter->second.score < min_score_threshold_) {
      return false;
    }
    *config = iter->second.config;
    return true;
  }
  void Insert(const ConvParameters& params, const Config& config) {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    int new_score = 0;
    if (iter == params_config_map_.end()) {
      // Create a new entry if params is new.
      VLOG(1) << GetActionSummary("creates", params, config);
      params_config_map_.insert(std::make_pair(params, ValueType{config, 1}));
      new_score = 1;
    } else if (iter->second.score < min_score_threshold_) {
      DCHECK(iter->second.score > 0);
      if (iter->second.config != config) {
        // If it is different from the current winner, demotes the winner.
        VLOG(1) << GetActionSummary("demotes", params, config);
        new_score = --iter->second.score;
        if (new_score <= 0) {
          VLOG(1) << GetActionSummary("erases", params, config);
          params_config_map_.erase(iter);
        }
      } else {
        // If it is the same as the current winner, promotes the winner.
        VLOG(1) << GetActionSummary("promotes", params, config);
        new_score = ++iter->second.score;
      }
    }
    if (new_score >= min_score_threshold_) {
      VLOG(1) << GetActionSummary("accepts", params, config);
    }
  }

 private:
  AutoTuneMap(const string& name) : name_(name) {
    min_score_threshold_ = 1;
    const char* threshold_str = getenv("TF_AUTOTUNE_THRESHOLD");
    if (threshold_str != nullptr) {
      strings::safe_strto32(threshold_str, &min_score_threshold_);
    }
    min_score_threshold_ = std::max(min_score_threshold_, 1);
  }

  template <class Group, class Params, class Cfg>
  friend class AutoTuneSingleton;

  struct Hasher {
    std::size_t operator()(const Parameters& parameter) const {
      return parameter.hash();
    }
  };

  string GetActionSummary(StringPiece action, const Parameters& params,
                          const Config& config) {
    return strings::Printf("autotune_map %s %s: %s -> (%s)", name_.c_str(),
                           action.ToString().c_str(), params.ToString().c_str(),
                           config.ToString().c_str());
  }

  mutable mutex mu_;
  struct ValueType {
    Config config;
    int32 score;
  };
  std::unordered_map<Parameters, ValueType, Hasher> params_config_map_
      GUARDED_BY(mu_);
  string name_;
  int32 min_score_threshold_;

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
    static AutoTuneType* instance = new AutoTuneType(Group::name());
    return instance;
  }
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
