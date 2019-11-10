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

#include "tensorflow/lite/delegates/gpu/gl/kernels/converter.h"

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

Dimensions ToDimensions(const BHWC& shape) {
  return Dimensions(shape.b, shape.h, shape.w, shape.c);
}

Status RunFromTensorTest(const BHWC& shape) {
  // Create random input and calculate expected output for it.
  std::vector<float> input =
      GenerateFloats(0.01, GetElementsSizeForPHWC4(shape));
  std::vector<float> output(shape.DimensionsProduct(), 0);
  RETURN_IF_ERROR(
      ConvertFromPHWC4(absl::MakeConstSpan(input.data(), input.size()), shape,
                       absl::MakeSpan(output.data(), output.size())));

  std::unique_ptr<EglEnvironment> env;
  RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env));

  // Create input and output buffers
  GlBuffer input_buffer;
  RETURN_IF_ERROR(CreateReadOnlyShaderStorageBuffer(
      absl::MakeConstSpan(input.data(), input.size()), &input_buffer));

  GlBuffer output_buffer;
  RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
      shape.DimensionsProduct(), &output_buffer));

  // Create converter and run it.
  auto builder = NewConverterBuilder(nullptr);
  TensorObjectDef input_def;
  input_def.object_def.data_type = DataType::FLOAT32;
  input_def.object_def.data_layout = DataLayout::DHWC4;
  input_def.object_def.object_type = ObjectType::OPENGL_SSBO;
  input_def.dimensions = ToDimensions(shape);
  TensorObjectDef output_def = input_def;
  output_def.object_def.data_layout = DataLayout::BHWC;
  std::unique_ptr<TensorObjectConverter> converter;
  RETURN_IF_ERROR(builder->MakeConverter(input_def, output_def, &converter));
  RETURN_IF_ERROR(converter->Convert(OpenGlBuffer{input_buffer.id()},
                                     OpenGlBuffer{output_buffer.id()}));

  // Compare outputs.
  std::vector<float> converted_output(output.size(), 0);
  RETURN_IF_ERROR(output_buffer.Read(
      absl::MakeSpan(converted_output.data(), converted_output.size())));
  if (output != converted_output) {
    return InternalError("Outputs don't match");
  }
  return OkStatus();
}

TEST(FromTensor, Smoke) {
  for (int32_t h : {1, 2, 3, 7, 20}) {
    for (int32_t w : {1, 2, 4, 5, 11}) {
      for (int32_t c : {1, 2, 4, 5, 8, 9}) {
        BHWC shape(1, h, w, c);
        auto status = RunFromTensorTest(shape);
        EXPECT_TRUE(status.ok()) << status << ", shape = " << shape.h << " "
                                 << shape.w << " " << shape.c;
      }
    }
  }
}

Status RunToTensorTest(const BHWC& shape) {
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
  auto builder = NewConverterBuilder(nullptr);
  TensorObjectDef input_def;
  input_def.object_def.data_type = DataType::FLOAT32;
  input_def.object_def.data_layout = DataLayout::BHWC;
  input_def.object_def.object_type = ObjectType::OPENGL_SSBO;
  input_def.dimensions = ToDimensions(shape);
  TensorObjectDef output_def = input_def;
  output_def.object_def.data_layout = DataLayout::DHWC4;
  std::unique_ptr<TensorObjectConverter> converter;
  RETURN_IF_ERROR(builder->MakeConverter(input_def, output_def, &converter));
  RETURN_IF_ERROR(converter->Convert(OpenGlBuffer{input_buffer.id()},
                                     OpenGlBuffer{output_buffer.id()}));

  // Compare outputs.
  std::vector<float> converted_output(output.size(), 0);
  RETURN_IF_ERROR(output_buffer.Read(
      absl::MakeSpan(converted_output.data(), converted_output.size())));
  if (output != converted_output) {
    return InternalError("Outputs don't match");
  }
  return OkStatus();
}

TEST(ToTensor, Smoke) {
  for (int32_t h : {1, 2, 3, 7, 20}) {
    for (int32_t w : {1, 2, 4, 5, 11}) {
      for (int32_t c : {1, 2, 4, 5, 8, 9}) {
        BHWC shape(1, h, w, c);
        auto status = RunToTensorTest(shape);
        EXPECT_TRUE(status.ok()) << status << ", shape = " << shape.h << " "
                                 << shape.w << " " << shape.c;
      }
    }
  }
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
