/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace fuzzing {

class FuzzEncodeJpeg : public FuzzSession {
  SINGLE_INPUT_OP_BUILDER(DT_UINT8, EncodeJpeg);

  void FuzzImpl(const uint8_t* data, size_t size) final {
    if (size < 6) return;

    // Pick random channels and aspect ratio, and then set the
    // input based upon the aspect ratio and size.
    int64 channels = (data[0] % 2) * 2 + 1;  // 1, 3
    int64 height = data[1] + (data[2] << 8);
    int64 width = data[2] + (data[3] << 8);
    if (width == 0) return;

    // TODO(dga): kcc@ notes: better to use actual supplied h, w and then
    // trim them if needed to ensure w*h <= size-4.
    double hw_ratio = height / width;
    int64 remaining_bytes = size - 5;
    int64 pixels = remaining_bytes / channels;
    height = static_cast<int64>(floor(sqrt(hw_ratio * pixels)));
    if (height == 0) return;
    width = static_cast<int64>(floor(pixels / height));
    if (width == 0) return;
    size_t actual_pixels = height * width * channels;
    if (actual_pixels == 0) return;

    // TODO(dga):  Generalize this by borrowing the AsTensor logic
    // from tf testing, once we have a few more fuzzers written.
    Tensor input_tensor(tensorflow::DT_UINT8,
                        TensorShape({height, width, channels}));
    auto flat_tensor = input_tensor.flat<uint8>();
    for (size_t i = 0; i < actual_pixels; i++) {
      flat_tensor(i) = data[i];
    }
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    RunOneInput(input_tensor).IgnoreError();
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzEncodeJpeg);

}  // end namespace fuzzing
}  // end namespace tensorflow
