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

// For Google-internal use only.
//
// This file defines the map data structure for storing autotuning results for
// fused_conv2d_bias_activation_op_kernels.
//
// The key of the map uniquely identifies a convolution operation that runs on a
// particular device model while the value might be the autotuned algorithm we
// choose for the conv.
//
// This map will be merged after fused_conv2d_bias_activation_op_kernels is
// merged into conv_ops_fused_impl.h (b/177365158, b/189530096)

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_FUSED_CONV_BIAS_ACTIVATION_AUTOTUNE_MAP_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_FUSED_CONV_BIAS_ACTIVATION_AUTOTUNE_MAP_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include <string>

#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"

namespace tensorflow {

// A dummy type to group forward convolution autotune results together.
struct ConvAutotuneGroup {
  static string name() { return "Conv"; }
};

using ConvAutotuneMap = AutotuneSingleton<ConvAutotuneGroup, ConvParameters,
                                          AutotuneEntry<se::dnn::ConvOp>>;

// A dummy type to group fused convolution autotune results together.
struct ConvFusedAutotuneGroup {
  static string name() { return "FusedConv"; }
};

using FusedConvAutotuneMap =
    AutotuneSingleton<ConvAutotuneGroup, ConvParameters,
                      AutotuneEntry<se::dnn::FusedConvOp>>;

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_FUSED_CONV_BIAS_ACTIVATION_AUTOTUNE_MAP_H_
