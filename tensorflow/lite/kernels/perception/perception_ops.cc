/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/perception/perception_ops.h"

#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {

extern "C" void AddPerceptionOps(::tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("DenseImageWarp",
                      tflite::ops::custom::RegisterDenseImageWarp());
  resolver->AddCustom("MaxPoolWithArgmax",
                      tflite::ops::custom::RegisterMaxPoolWithArgmax());
  resolver->AddCustom("MaxUnpooling2D",
                      tflite::ops::custom::RegisterMaxUnpooling2D());
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
