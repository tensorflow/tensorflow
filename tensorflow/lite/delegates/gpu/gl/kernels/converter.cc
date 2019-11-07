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

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Wraps given SSBO into GlBuffer object that does not have ownership.
Status WrapSSBO(OpenGlBuffer ssbo, GlBuffer* buffer) {
  int64_t size_bytes;
  RETURN_IF_ERROR(GetSSBOSize(ssbo.id, &size_bytes));
  *buffer = GlBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.id, size_bytes, 0, false);
  return OkStatus();
}

std::string GetShaderHeader(const uint3& localsize) {
  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

class OpenGlConverterImpl : public TensorObjectConverter {
 public:
  explicit OpenGlConverterImpl(CommandQueue* command_queue)
      : command_queue_(command_queue) {}

  virtual Status Init(const TensorObjectDef& input_def,
                      const TensorObjectDef& output_def) = 0;

 protected:
  Status InitializeProgram(const uint3& workgroup_size,
                           const std::string& shader_source) {
    workgroup_size_ = workgroup_size;
    GlShader shader;
    RETURN_IF_ERROR(GlShader::CompileShader(
        GL_COMPUTE_SHADER, GetShaderHeader(workgroup_size) + shader_source,
        &shader));
    return GlProgram::CreateWithShader(shader, &program_);
  }

  Status Dispatch(const uint3& workload) {
    uint3 num_workgroups = IntegralDivideRoundUp(workload, workgroup_size_);
    if (command_queue_) {
      return command_queue_->Dispatch(program_, num_workgroups);
    }
    return program_.Dispatch(num_workgroups);
  }

  GlProgram program_;
  uint3 workgroup_size_;
  CommandQueue* command_queue_;
};

bool IsSupportedDataType(DataType type) { return type == DataType::FLOAT32; }

uint32_t SizeInBytesDHWC4(const BHWC& shape) {
  return shape.b * shape.h * shape.w * AlignByN(shape.c, 4) * sizeof(float);
}

uint32_t SizeInBytesBHWC(const BHWC& shape) {
  return shape.DimensionsProduct() * sizeof(float);
}

// Implements conversion from OpenGL-specific tensor layout to BHWC.
class FromTensorConverter : public OpenGlConverterImpl {
 public:
  explicit FromTensorConverter(CommandQueue* command_queue)
      : OpenGlConverterImpl(command_queue) {}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Output is always SSBO/BHWC
           output.object_type == ObjectType::OPENGL_SSBO &&
           output.data_layout == DataLayout::BHWC &&
           // SSBO/DHWC4 ->
           input.object_type == ObjectType::OPENGL_SSBO &&
           input.data_layout == DataLayout::DHWC4;
  }

  Status Init(const TensorObjectDef& input_def,
              const TensorObjectDef& output_def) final {
    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    if (shape_.b != 1) {
      return UnimplementedError(
          "FromTensorConverter: Batch size != 1 is not supported.");
    }

    return InitializeProgram(uint3(8, 4, 2), R"(
    layout(std430) buffer;
    precision highp float;

    layout(binding = 0) readonly buffer B0 {
      vec4 elements[];
    } input_data;

    layout(binding = 1) writeonly buffer B1 {
      float elements[];
    } output_data;

    uniform ivec4 sizes;

    void main() {
      ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
      if (gid.x >= sizes.x || gid.y >= sizes.y || gid.z >= sizes.z) {
        return;
      }
      output_data.elements[(gid.y * sizes.x + gid.x) * sizes.z + gid.z] = input_data.elements[(gid.z / 4 * sizes.y + gid.y) * sizes.x + gid.x][gid.z % 4];
    })");
  }

  Status Convert(const TensorObject& input_obj,
                 const TensorObject& output_obj) override {
    auto output = absl::get_if<OpenGlBuffer>(&output_obj);
    if (!output || !output->id) {
      return InvalidArgumentError("Missing output in converter");
    }
    auto input = absl::get_if<OpenGlBuffer>(&input_obj);
    if (!input || !input->id) {
      return InvalidArgumentError("Missing input in converter");
    }
    if (input->id == output->id) {
      return InvalidArgumentError("Can not execute inplace conversion");
    }
    GlBuffer input_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*input, &input_ssbo));
    GlBuffer output_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*output, &output_ssbo));

    if (input_ssbo.bytes_size() != SizeInBytesDHWC4(shape_)) {
      return InvalidArgumentError(
          "FromTensorConverter: input data size does not match expected size.");
    }
    if (output_ssbo.bytes_size() != SizeInBytesBHWC(shape_)) {
      return InvalidArgumentError(
          "FromTensorConverter: output data size does not match expected "
          "size.");
    }
    RETURN_IF_ERROR(program_.SetParameter(
        {"sizes",
         int4(static_cast<int32_t>(shape_.w), static_cast<int32_t>(shape_.h),
              static_cast<int32_t>(shape_.c), 0)}));
    RETURN_IF_ERROR(input_ssbo.BindToIndex(0));
    RETURN_IF_ERROR(output_ssbo.BindToIndex(1));
    return Dispatch(uint3(shape_.w, shape_.h, shape_.c));
  }

  BHWC shape_;
};

// Implements conversion from BHWC to OpenGL-specific tensor layout.
class ToTensorConverter : public OpenGlConverterImpl {
 public:
  explicit ToTensorConverter(CommandQueue* command_queue)
      : OpenGlConverterImpl(command_queue) {}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Input is always SSBO/BHWC
           input.object_type == ObjectType::OPENGL_SSBO &&
           input.data_layout == DataLayout::BHWC &&
           // -> SSBO/DHWC4
           output.object_type == ObjectType::OPENGL_SSBO &&
           output.data_layout == DataLayout::DHWC4;
  }

  Status Init(const TensorObjectDef& input_def,
              const TensorObjectDef& output_def) final {
    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    if (shape_.b != 1) {
      return UnimplementedError(
          "FromTensorConverter: Batch size != 1 is not supported.");
    }

    return InitializeProgram(uint3(8, 4, 2), R"(
    layout(std430) buffer;
    precision highp float;

    layout(binding = 0) readonly buffer B0 {
      float elements[];
    } input_data;

    layout(binding = 1) writeonly buffer B1 {
      vec4 elements[];
    } output_data;

    uniform ivec4 sizes;

    void main() {
      ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
      if (gid.x >= sizes.x || gid.y >= sizes.y || gid.z >= sizes.w) {
        return;
      }
      vec4 v = vec4(0);
      int dst_channel = gid.z * 4;
      int index = (gid.y * sizes.x + gid.x) * sizes.z + dst_channel;
      for (int i = 0; i < 4; ++i, ++index, ++dst_channel) {
        if (dst_channel >= sizes.z) break;
        v[i] = input_data.elements[index];
      }
      output_data.elements[(gid.z * sizes.y + gid.y) * sizes.x + gid.x] = v;
    })");
  }

  Status Convert(const TensorObject& input_obj,
                 const TensorObject& output_obj) override {
    auto output = absl::get_if<OpenGlBuffer>(&output_obj);
    if (!output || !output->id) {
      return InvalidArgumentError("Missing output in converter");
    }
    auto input = absl::get_if<OpenGlBuffer>(&input_obj);
    if (!input || !input->id) {
      return InvalidArgumentError("Missing input in converter");
    }
    if (input->id == output->id) {
      return InvalidArgumentError("Can not execute inplace conversion");
    }
    GlBuffer input_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*input, &input_ssbo));
    GlBuffer output_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*output, &output_ssbo));

    if (input_ssbo.bytes_size() != SizeInBytesBHWC(shape_)) {
      return InvalidArgumentError(
          "ToTensorConverter: input data size does not match expected size.");
    }
    if (output_ssbo.bytes_size() != SizeInBytesDHWC4(shape_)) {
      return InvalidArgumentError(
          "ToTensorConverter: output data size does not match expected size.");
    }
    auto d = IntegralDivideRoundUp(shape_.c, 4);
    RETURN_IF_ERROR(program_.SetParameter(
        {"sizes",
         int4(static_cast<int32_t>(shape_.w), static_cast<int32_t>(shape_.h),
              static_cast<int32_t>(shape_.c), static_cast<int32_t>(d))}));
    RETURN_IF_ERROR(input_ssbo.BindToIndex(0));
    RETURN_IF_ERROR(output_ssbo.BindToIndex(1));
    return Dispatch(uint3(shape_.w, shape_.h, d));
  }

  BHWC shape_;
};

// Copies data from one object of the same type and layout to another object.
class TrivialCopier : public TensorObjectConverter {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.object_type == ObjectType::OPENGL_SSBO &&
           input.data_type == output.data_type &&
           input.object_type == output.object_type &&
           input.data_layout == output.data_layout;
  }

  Status Convert(const TensorObject& input_obj,
                 const TensorObject& output_obj) override {
    auto ssbo_input = absl::get_if<OpenGlBuffer>(&input_obj);
    auto ssbo_output = absl::get_if<OpenGlBuffer>(&output_obj);
    if (ssbo_input && ssbo_output) {
      return Copy(*ssbo_input, *ssbo_output);
    }
    return InternalError("Unexpected object");
  }

  Status Copy(OpenGlBuffer input, OpenGlBuffer output) {
    if (input.id == output.id) {
      return OkStatus();
    }
    GlBuffer input_obj;
    RETURN_IF_ERROR(WrapSSBO(input, &input_obj));
    GlBuffer output_obj;
    RETURN_IF_ERROR(WrapSSBO(output, &output_obj));
    return CopyBuffer(input_obj, output_obj);
  }
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public TensorObjectConverter {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             output.object_type == ObjectType::OPENGL_SSBO) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             input.object_type == ObjectType::OPENGL_SSBO));
  }

  Status Convert(const TensorObject& input_obj,
                 const TensorObject& output_obj) override {
    auto cpu_input = absl::get_if<CpuMemory>(&input_obj);
    auto cpu_output = absl::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      auto ssbo_output = absl::get_if<OpenGlBuffer>(&output_obj);
      if (ssbo_output) {
        GlBuffer gl_buffer;
        RETURN_IF_ERROR(WrapSSBO(*ssbo_output, &gl_buffer));
        return gl_buffer.Write(
            absl::MakeConstSpan(static_cast<const uint8_t*>(cpu_input->data),
                                cpu_input->size_bytes));
      }
    } else if (cpu_output) {
      auto ssbo_input = absl::get_if<OpenGlBuffer>(&input_obj);
      if (ssbo_input) {
        GlBuffer gl_buffer;
        RETURN_IF_ERROR(WrapSSBO(*ssbo_input, &gl_buffer));
        return gl_buffer.Read(absl::MakeSpan(
            static_cast<uint8_t*>(cpu_output->data), cpu_output->size_bytes));
      }
    }
    return InternalError("Unexpected object");
  }
};

class TensorConverterBuilderImpl : public TensorObjectConverterBuilder {
 public:
  explicit TensorConverterBuilderImpl(CommandQueue* command_queue)
      : command_queue_(command_queue) {}

  bool IsSupported(const TensorObjectDef& input,
                   const TensorObjectDef& output) const final {
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    return input.dimensions == output.dimensions &&
           (TrivialCopier::IsSupported(input_def, output_def) ||
            CpuCopier::IsSupported(input_def, output_def) ||
            FromTensorConverter::IsSupported(input_def, output_def) ||
            ToTensorConverter::IsSupported(input_def, output_def));
  }

  Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
    std::unique_ptr<OpenGlConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    if (TrivialCopier::IsSupported(input_def, output_def)) {
      *converter = absl::make_unique<TrivialCopier>();
      return OkStatus();
    } else if (CpuCopier::IsSupported(input_def, output_def)) {
      *converter = absl::make_unique<CpuCopier>();
      return OkStatus();
    } else if (FromTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<FromTensorConverter>(command_queue_);
    } else if (ToTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<ToTensorConverter>(command_queue_);
    } else {
      return UnimplementedError("Unsupported conversion");
    }
    RETURN_IF_ERROR(impl->Init(input, output));
    *converter = std::move(impl);
    return OkStatus();
  }

 private:
  CommandQueue* command_queue_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    CommandQueue* command_queue) {
  return absl::make_unique<TensorConverterBuilderImpl>(command_queue);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
