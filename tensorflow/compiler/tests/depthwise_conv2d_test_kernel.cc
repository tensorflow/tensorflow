/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/types.h"

using tensorflow::int64;

// A dummy implementation that fills the output with 0, 1, 2,...
// to test the custom call implementation of DepthwiseConv2dNative op.
// TODO(keveman): Test this after adding a real implementation for the kernel.
extern "C" void DummyDepthwiseConv2dKernel(float* output, void** inputs) {
  const int64* output_size = reinterpret_cast<const int64*>(inputs[4]);
  const int64 total_size =
      output_size[0] * output_size[1] * output_size[2] * output_size[3];
  for (int64 i = 0; i < total_size; ++i) {
    *(output + i) = i;
  }
}
