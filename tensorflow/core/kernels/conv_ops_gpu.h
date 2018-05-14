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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

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

  // The purpose of this function is to disable winograd nonfused conv algorithm
  // for certain input parameters so as to avoid a bug in cuDNNv5 and cuDNNv6.
  template <typename T>
  bool ShouldIncludeWinogradNonfusedAlgo(
      se::StreamExecutor* stream_exec) const {
    // Skip this check for cuDNN 7 and newer.
    auto version = stream_exec->AsDnn()->GetVersion();
    if (version.ok() && version.ValueOrDie().major_version() >= 7) {
      return true;
    }
    return ShouldIncludeWinogradNonfusedAlgoPreDnn7<T>();
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
  friend struct ConvParametersPeer;  // For testing purposes.

  template <typename T>
  bool ShouldIncludeWinogradNonfusedAlgoPreDnn7() const {
    int64 total_size = 16 * std::ceil(batch_ / 16.0) *
                       std::max(in_depths_, out_depths_) * in_[0] * in_[1] *
                       sizeof(T);
    int64 threshold = 1LL << 31;
    if (total_size >= threshold) {
      return false;
    } else {
      return true;
    }
  }

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

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
