/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_PAD_FEATURES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_PAD_FEATURES_H_

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {

// Zero-pad input/output feature the desired shapes
// before the HloCustomCallInstruction calling cuDNN convolution and remove the
// unecessary output after it
class CudnnConvPadFeatures {
 public:
  // This is the main function of the transform.  It runs on a given module.
  // It then finds individual custom call nodes to cuDNN convolution in the
  // module, calls resolve_pad_shapes to resolve the desired input/output
  // feature map shapes, and adds necessary padding and slicing nodes around
  // them.
  //
  // resolve_pad_shapes points to a function.  It takes conv, a custom call
  // instruction to cuDNN convolution that may need padding to figure out the
  // desired input and output tensor shapes after padding and store the desired
  // shapes in new_input_shapes and new_input_shapes.  Notice that
  // new_input_shapes is a vector for multiple input tesnsors. This function
  // shall return true, if padding is necessary or false otherwise in addition
  // status.
  StatusOr<bool> Run(
      HloModule* module,
      StatusOr<bool> (*resolve_pad_shapes)(HloCustomCallInstruction* conv,
                                           std::vector<Shape>* new_input_shapes,
                                           Shape* new_result_shape));
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_PAD_FEATURES_H_
