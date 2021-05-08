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

/*
GENERIC FAST
This optimized kernel directory contains optimized kernels.
The kernels are portable to every hardware, no custom instructions are used.
The kernels take advantage of precomputations, smaller tweaks and the prepare
phase to reduce runtime and memory overhead.
==============================================================================*/

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/generic_fast/reduce/reduce_impl.h"

namespace tflite {
namespace ops {
namespace micro {

TfLiteRegistration Register_MEAN() {
  return {/*init=*/reduce::InitReduce,
          /*free=*/nullptr,
          /*prepare=*/reduce::PrepareMeanOrSum,
          /*invoke=*/reduce::EvalMeanMax,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}
TfLiteRegistration Register_REDUCE_MAX() {
  return {/*init=*/reduce::InitReduce,
          /*free=*/nullptr,
          /*prepare=*/reduce::PrepareMax,
          /*invoke=*/reduce::EvalMeanMax,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
