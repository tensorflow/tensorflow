/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "absl/types/optional.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {
// Uniquely identifies a convolution operation that runs on a particular device
// model.
//
// This can serve as a hashtable key, where the value might be the autotuned
// algorithm we choose for the conv.
//
// All of the data in this class other than the device_id is stored in the
// ConvParametersProto, so it can be easily serialized (for the purposes of
// ahead-of-time autotuning).
//
// When using the cudnn frontend API, two autotuning results for two different
// GPUs of the same model are not interchangeable, because an autotuning result
// includes a cudnn execution plan, which is tied to the GPU.  As a result, we
// need to create separate ConvParameters objects for them.
class ConvParameters {
 public:
  struct FusionInfo {
    // For some implementations (e.g. cuDNN new backend) these scales are part
    // of the algorithm, not part of the parameters an algorithm take. They need
    // to be used to distinguish different algorithms.
    double conv_scale;
    double side_input_scale;
    double leakyrelu_alpha;
    stream_executor::dnn::ActivationMode activation_mode;
    bool is_contrib;
  };

  // LINT.IfChange(conv_parameters_version)
  // A positive number that denotes the version of this class. Should be
  // incremented everytime this class or ConvParametersProto are updated in a
  // way that may invalidate autotune results.
  static constexpr int kVersion = 3;
  // LINT.ThenChange()

  // We have three kinds of convolutions today.  Vanilla unfused convolutions,
  // fused convolutions, and fused convolutions as implemented in the `contrib`
  // directory.  The two fused convolutions ultimately correspond to the same
  // cudnn calls, but have slightly different semantics (e.g. they interpret
  // padding differently).
  ConvParameters(
      se::StreamExecutor* stream_exec, int64_t batch, int64_t in_depths,
      absl::Span<const int64_t> in, int data_format, int64_t out_depths,
      absl::Span<const int64_t> filter, absl::Span<const int64_t> dilation,
      absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
      DataType dtype, int group_count,
      absl::optional<FusionInfo> fusion_info = absl::optional<FusionInfo>(),
      // This argument should be set only for test use.
      int version = kVersion);

  ConvParameters(se::StreamExecutor* stream_exec,
                 const ConvParametersProto& proto);

  bool operator==(const ConvParameters& other) const;

  bool operator!=(const ConvParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const;

  const ConvParametersProto& proto() const { return proto_; }

 private:
  int device_id_;
  ConvParametersProto proto_;
  uint64 hash_code_;
};

class MatmulParameters {
 public:
  // LINT.IfChange(matmul_parameters_version)
  // A positive number that denotes the version of this class. Should be
  // incremented everytime this class or ConvParametersProto are updated in a
  // way that may invalidate autotune results.
  static constexpr int kVersion = 2;
  // LINT.ThenChange()

  MatmulParameters(se::StreamExecutor* stream_exec, DataType ab_dtype,
                   DataType c_dtype, bool trans_a, bool trans_b, uint64_t m,
                   uint64_t n, uint64_t k, int64_t lda, int64_t ldb,
                   int64_t ldc,
                   stream_executor::dnn::ActivationMode activation_mode,
                   // This argument should be set only for test use.
                   int version = kVersion);

  MatmulParameters(se::StreamExecutor* stream_exec,
                   const MatmulParametersProto& proto);

  bool operator==(const MatmulParameters& other) const;

  bool operator!=(const MatmulParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const;

  const MatmulParametersProto& proto() const { return proto_; }

 private:
  int device_id_;
  MatmulParametersProto proto_;
  uint64 hash_code_;
};

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
