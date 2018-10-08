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

#ifdef TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/gpu_fusion_ops.h"

namespace tensorflow {

se::dnn::ActivationMode GetDnnActivationMode(ActivationMode activation_mode) {
  se::dnn::ActivationMode dnn_activation_mode;
  switch (activation_mode) {
    case ActivationMode::NONE:
      dnn_activation_mode = se::dnn::ActivationMode::kNone;
      break;
    case ActivationMode::SIGMOID:
      dnn_activation_mode = se::dnn::ActivationMode::kSigmoid;
      break;
    case ActivationMode::RELU:
      dnn_activation_mode = se::dnn::ActivationMode::kRelu;
      break;
    case ActivationMode::RELU6:
      dnn_activation_mode = se::dnn::ActivationMode::kRelu6;
      break;
    case ActivationMode::RELUX:
      dnn_activation_mode = se::dnn::ActivationMode::kReluX;
      break;
    case ActivationMode::TANH:
      dnn_activation_mode = se::dnn::ActivationMode::kTanh;
      break;
    case ActivationMode::BANDPASS:
      dnn_activation_mode = se::dnn::ActivationMode::kBandPass;
      break;
    default:
      LOG(FATAL) << "Activation mode " << activation_mode << " not supported";
  }

  return dnn_activation_mode;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
