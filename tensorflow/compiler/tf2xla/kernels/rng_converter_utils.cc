/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/rng_alg.h"

namespace tensorflow {

Algorithm ToTensorflowAlgorithm(xla::RandomAlgorithm alg) {
  switch (alg) {
    case xla::RandomAlgorithm::RNG_PHILOX:
      return RNG_ALG_PHILOX;
    case xla::RandomAlgorithm::RNG_THREE_FRY:
      return RNG_ALG_THREEFRY;
    case xla::RandomAlgorithm::RNG_DEFAULT:  // fall through
    default:
      // The output counter will have the maximal size, so it's safe to let
      // downstream RNG ops choose the algorithm.
      return RNG_ALG_AUTO_SELECT;
  }
}

xla::RandomAlgorithm DefaultRngAlgForDeviceType(
    absl::string_view device_type_string) {
  // The Philox algorithm may cause performance regression on other devices.
  // Turn on the Philox algorithm for the CPU and GPU backends only.
  if (device_type_string == DEVICE_GPU_XLA_JIT ||
      device_type_string == DEVICE_CPU_XLA_JIT) {
    return xla::RandomAlgorithm::RNG_PHILOX;
  } else {
    return xla::RandomAlgorithm::RNG_DEFAULT;
  }
}

}  // namespace tensorflow
