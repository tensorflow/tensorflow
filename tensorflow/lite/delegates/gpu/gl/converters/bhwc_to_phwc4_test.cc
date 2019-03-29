/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/converters/bhwc_to_phwc4.h"

#include <algorithm>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

inline std::vector<float> GenerateFloats(float multiplier, int size) {
  std::vector<float> v(size);
  for (int i = 0; i < size; ++i) {
    v[i] = multiplier * i * (i % 2 == 0 ? -1 : 1);
  }
  return v;
}

Status RunTest(const BHWC& shape) {
  // Create random input and calculate expected output for it.
  std::vector<float> input = GenerateFloats(0.01, shape.DimensionsProduct());
  std::vector<float> output(GetElementsSizeForPHWC4(shape), 0);
  RETURN_IF_ERROR(
      ConvertToPHWC4(absl::MakeConstSpan(input.data(), input.size()), shape,
                     absl::MakeSpan(output.data(), output.size())));

  std::unique_ptr<EglEnvironment> env;
  RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env));

  // Create input and output buffers
  GlBuffer input_buffer;
  RETURN_IF_ERROR(CreateReadOnlyShaderStorageBuffer(
      absl::MakeConstSpan(input.data(), input.size()), &input_buffer));

  GlBuffer output_buffer;
  RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
      GetElementsSizeForPHWC4(shape), &output_buffer));

  // Create converter and run it.
  ConverterBhwcToPhwc4 converter;
  RETURN_IF_ERROR(ConverterBhwcToPhwc4::Create(&converter));
  RETURN_IF_ERROR(
      converter.Convert(shape, input_buffer, nullptr, &output_buffer));

  std::vector<float> converted_output(output.size(), 0);
  RETURN_IF_ERROR(output_buffer.Read(
      absl::MakeSpan(converted_output.data(), converted_output.size())));
  if (output != converted_output) {
    return InternalError("Outputs don't match");
  }
  return OkStatus();
}

TEST(HwcToPhwc4, Smoke) {
  for (int32_t h : {1, 2, 3, 7, 20}) {
    for (int32_t w : {1, 2, 4, 5, 11}) {
      for (int32_t c : {1, 2, 4, 5, 8, 9}) {
        BHWC shape(1, h, w, c);
        EXPECT_TRUE(RunTest(shape).ok())
            << shape.h << " " << shape.w << " " << shape.c;
      }
    }
  }
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
