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

#include "tensorflow/lite/delegates/gpu/cl/kernels/converter.h"

#include <algorithm>
#include <array>
#include <string>

#include "tensorflow/lite/delegates/gpu/cl/arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

class OpenClConverterImpl : public TensorObjectConverter {
 public:
  virtual absl::Status Init(const TensorObjectDef& input_def,
                            const TensorObjectDef& output_def,
                            Environment* environment) = 0;

 protected:
  absl::Status DispatchKernel(cl_mem buffer_mem, Tensor* tensor) {
    kernel_.ResetBindingCounter();
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(buffer_mem));
    RETURN_IF_ERROR(args_.SetObjectRef("tensor", tensor));
    RETURN_IF_ERROR(args_.Bind(kernel_.kernel(), kernel_.GetBindingCounter()));
    int3 grid = int3(tensor->Width() * tensor->Batch(), tensor->Height(),
                     tensor->Slices());
    return queue_->DispatchImplicit(kernel_, grid, {16, 8, 1});
  }

  Arguments args_;
  BHWC shape_;
  CLKernel kernel_;
  TensorDescriptor tensor_descriptor_;
  CLCommandQueue* queue_ = nullptr;
  const CLContext* context_ = nullptr;
};

bool IsSupportedDataType(DataType type) {
  return type == DataType::FLOAT16 || type == DataType::FLOAT32;
}

// Implements conversion from OpenCL-specific tensor layout to BHWC.
class FromTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Output is always Buffer/(BHWC|DHWC4)
           output.object_type == ObjectType::OPENCL_BUFFER &&
           (output.data_layout == DataLayout::BHWC ||
            output.data_layout == DataLayout::DHWC4) &&
           // Texture2D/HDWC4 ->
           ((input.object_type == ObjectType::OPENCL_TEXTURE &&
             input.data_layout == DataLayout::HDWC4) ||
            // SingleTextureArray/BHWC ->
            (input.object_type == ObjectType::OPENCL_TEXTURE &&
             input.data_layout == DataLayout::BHWC) ||
            // TextureArray/DHWC4 ->
            (input.object_type == ObjectType::OPENCL_TEXTURE &&
             input.data_layout == DataLayout::DHWC4) ||
            // Buffer/DHWC4 ->
            (input.object_type == ObjectType::OPENCL_BUFFER &&
             input.data_layout == DataLayout::DHWC4));
  }

  std::pair<std::string, std::string> GetToDhwc4Kernel(
      const TensorObjectDef& input_def,
      const TensorObjectDef& output_def) const {
    return std::make_pair("__global " +
                              ToCLDataType(output_def.object_def.data_type, 4) +
                              "* dst",
                          "dst[(d * args.tensor.Height() + y) * "
                          "args.tensor.Width() + x] = input;");
  }

  std::pair<std::string, std::string> GetToBhwcKernel(
      const TensorObjectDef& input_def,
      const TensorObjectDef& output_def) const {
    return std::make_pair(
        "__global " + ToCLDataType(output_def.object_def.data_type) + "* dst",
        R"(
  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;

  dst[index] = input.x;
  if (c + 1 < args.tensor.Channels()) {
    dst[index + 1] = input.y;
  }
  if (c + 2 < args.tensor.Channels()) {
    dst[index + 2] = input.z;
  }
  if (c + 3 < args.tensor.Channels()) {
    dst[index + 3] = input.w;
  })");
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    auto params_kernel = output_def.object_def.data_layout == DataLayout::BHWC
                             ? GetToBhwcKernel(input_def, output_def)
                             : GetToDhwc4Kernel(input_def, output_def);

    TensorStorageType src_tensor_type = ToTensorStorageType(
        input_def.object_def.object_type, input_def.object_def.data_layout);
    tensor_descriptor_.layout = Layout::BHWC;
    tensor_descriptor_.storage_type = src_tensor_type;
    tensor_descriptor_.data_type = input_def.object_def.data_type;
    args_.AddObjectRef("tensor", AccessType::READ,
                       absl::make_unique<TensorDescriptor>(tensor_descriptor_));
    std::string shader_src =
        R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

const sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void from_tensor()" +
        params_kernel.first + R"(, $0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  )" + ToCLDataType(output_def.object_def.data_type, 4) +
        " input = args.tensor.Read<" +
        ToCLDataType(output_def.object_def.data_type) + ">(x, y, d, b);\n" +
        params_kernel.second + "\n}";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(args_.TransformToCLCode(environment->device().GetInfo(), {},
                                            &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "from_tensor", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto output = absl::get_if<OpenClBuffer>(&output_obj);
    if (!output || !output->memobj) {
      return absl::InvalidArgumentError(
          "Missing output in from_tensor converter");
    }
    cl_mem memory = nullptr;
    auto input_texture = absl::get_if<OpenClTexture>(&input_obj);
    if (input_texture && input_texture->memobj) {
      memory = input_texture->memobj;
    }
    auto input_buffer = absl::get_if<OpenClBuffer>(&input_obj);
    if (input_buffer && input_buffer->memobj) {
      memory = input_buffer->memobj;
    }
    if (!memory) {
      return absl::InvalidArgumentError(
          "Missing input in from_tensor converter");
    }
    Tensor tensor;
    RETURN_IF_ERROR(CreateSharedTensor(*context_, memory, shape_,
                                       tensor_descriptor_, &tensor));
    return DispatchKernel(output->memobj, &tensor);
  }
};

// Implements conversion from BHWC to OpenCL-specific tensor layout.
class ToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Input is always Buffer/BHWC
           input.object_type == ObjectType::OPENCL_BUFFER &&
           (input.data_layout == DataLayout::BHWC ||
            input.data_layout == DataLayout::DHWC4) &&
           // -> Texture2D/HDWC4
           ((output.object_type == ObjectType::OPENCL_TEXTURE &&
             output.data_layout == DataLayout::HDWC4) ||
            // -> TextureArray/DHWC4
            (output.object_type == ObjectType::OPENCL_TEXTURE &&
             output.data_layout == DataLayout::DHWC4) ||
            // -> SingleTextureArray/BHWC
            (output.object_type == ObjectType::OPENCL_TEXTURE &&
             output.data_layout == DataLayout::BHWC) ||
            // -> Buffer/DHWC4
            (output.object_type == ObjectType::OPENCL_BUFFER &&
             output.data_layout == DataLayout::DHWC4));
  }

  std::pair<std::string, std::string> GetFromDhwc4Kernel(
      const TensorObjectDef& input_def,
      const TensorObjectDef& output_def) const {
    return std::make_pair(
        "__global " + ToCLDataType(input_def.object_def.data_type, 4) + "* src",
        output_def.object_def.data_type == input_def.object_def.data_type
            ? "result = src[(d * args.tensor.Height() + y) * "
              "args.tensor.Width() + x];"
            : "result = convert_" +
                  ToCLDataType(output_def.object_def.data_type, 4) +
                  "(src[(d * args.tensor.Height() + y) * args.tensor.Width() + "
                  "x]);");
  }

  std::pair<std::string, std::string> GetFromBhwcKernel(
      const TensorObjectDef& input_def,
      const TensorObjectDef& output_def) const {
    return std::make_pair(
        "__global " + ToCLDataType(input_def.object_def.data_type) + "* src",
        R"(int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  result.x = src[index];
  result.y = c + 1 < args.tensor.Channels() ? src[index + 1] : 1;
  result.z = c + 2 < args.tensor.Channels() ? src[index + 2] : 2;
  result.w = c + 3 < args.tensor.Channels() ? src[index + 3] : 3;
)");
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    auto params_kernel = input_def.object_def.data_layout == DataLayout::BHWC
                             ? GetFromBhwcKernel(input_def, output_def)
                             : GetFromDhwc4Kernel(input_def, output_def);
    TensorStorageType dst_tensor_type = ToTensorStorageType(
        output_def.object_def.object_type, output_def.object_def.data_layout);
    tensor_descriptor_.layout = Layout::BHWC;
    tensor_descriptor_.storage_type = dst_tensor_type;
    tensor_descriptor_.data_type = output_def.object_def.data_type;
    args_.AddObjectRef("tensor", AccessType::WRITE,
                       absl::make_unique<TensorDescriptor>(tensor_descriptor_));
    std::string shader_src =
        R"(
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void to_tensor()" +
        params_kernel.first + R"(, $0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);

  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  )" + ToCLDataType(output_def.object_def.data_type, 4) +
        " result;\n" + params_kernel.second + "\n  " +
        "args.tensor.Write(result, x, y, d, b);\n}";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    RETURN_IF_ERROR(args_.TransformToCLCode(environment->device().GetInfo(), {},
                                            &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "to_tensor", environment->context(), environment->device(),
        &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto input = absl::get_if<OpenClBuffer>(&input_obj);
    if (!input || !input->memobj) {
      return absl::InvalidArgumentError("Missing input in to_tensor converter");
    }
    cl_mem memory = nullptr;
    auto output_texture = absl::get_if<OpenClTexture>(&output_obj);
    if (output_texture && output_texture->memobj) {
      memory = output_texture->memobj;
    }
    auto output_buffer = absl::get_if<OpenClBuffer>(&output_obj);
    if (output_buffer && output_buffer->memobj) {
      memory = output_buffer->memobj;
    }
    if (!memory) {
      return absl::InvalidArgumentError(
          "Missing output in to_tensor converter");
    }
    Tensor tensor;
    RETURN_IF_ERROR(CreateSharedTensor(*context_, memory, shape_,
                                       tensor_descriptor_, &tensor));
    return DispatchKernel(input->memobj, &tensor);
  }
};

std::array<size_t, 3> CalculateTextureRegion(const TensorObjectDef& def) {
  const auto& dims = def.dimensions;
  std::array<size_t, 3> region = {0, 0, 1};
  switch (ToTensorStorageType(def.object_def.object_type,
                              def.object_def.data_layout)) {
    case TensorStorageType::SINGLE_TEXTURE_2D:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h);
      break;
    case TensorStorageType::TEXTURE_2D:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h * dims.d());
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h);
      region[2] = static_cast<size_t>(dims.d());
      break;
    default:
      break;
  }
  return region;
}

bool IsOpenClTextureOrBuffer(ObjectType type) {
  return type == ObjectType::OPENCL_BUFFER ||
         type == ObjectType::OPENCL_TEXTURE;
}

// Copies data from one object of the same type and layout to another object.
class TrivialCopier : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsOpenClTextureOrBuffer(input.object_type) &&
           input.data_type == output.data_type &&
           input.object_type == output.object_type &&
           input.data_layout == output.data_layout;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    data_type_ = input_def.object_def.data_type;
    queue_ = environment->queue();
    region_ = CalculateTextureRegion(output_def);
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto texture_input = absl::get_if<OpenClTexture>(&input_obj);
    auto texture_output = absl::get_if<OpenClTexture>(&output_obj);
    if (texture_input && texture_output) {
      return Copy(*texture_input, *texture_output);
    }
    auto buffer_input = absl::get_if<OpenClBuffer>(&input_obj);
    auto buffer_output = absl::get_if<OpenClBuffer>(&output_obj);
    if (buffer_input && buffer_output) {
      return Copy(*buffer_input, *buffer_output);
    }
    return absl::InternalError("Unexpected object");
  }

  absl::Status Copy(const OpenClBuffer& input, const OpenClBuffer& output) {
    if (input.memobj == output.memobj) {
      return absl::OkStatus();
    }
    return GetOpenCLError(
        clEnqueueCopyBuffer(queue_->queue(), input.memobj, output.memobj, 0, 0,
                            SizeOf(data_type_) * shape_.w * shape_.h *
                                AlignByN(shape_.c, 4) * shape_.b,
                            0, nullptr, nullptr));
  }

  absl::Status Copy(const OpenClTexture& input, const OpenClTexture& output) {
    if (input.memobj == output.memobj) {
      return absl::OkStatus();
    }
    size_t origin[3] = {0, 0, 0};
    return GetOpenCLError(
        clEnqueueCopyImage(queue_->queue(), input.memobj, output.memobj, origin,
                           origin, region_.data(), 0, nullptr, nullptr));
  }

 private:
  DataType data_type_ = DataType::UNKNOWN;
  std::array<size_t, 3> region_;
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             IsOpenClTextureOrBuffer(output.object_type)) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             IsOpenClTextureOrBuffer(input.object_type)));
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    region_ = CalculateTextureRegion(
        input_def.object_def.object_type == ObjectType::CPU_MEMORY ? output_def
                                                                   : input_def);
    queue_ = environment->queue();
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto cpu_input = absl::get_if<CpuMemory>(&input_obj);
    auto cpu_output = absl::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      auto texture_output = absl::get_if<OpenClTexture>(&output_obj);
      if (texture_output) {
        return queue_->EnqueueWriteImage(
            texture_output->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_input->data);
      }
      auto buffer_output = absl::get_if<OpenClBuffer>(&output_obj);
      if (buffer_output) {
        return queue_->EnqueueWriteBuffer(
            buffer_output->memobj, cpu_input->size_bytes, cpu_input->data);
      }
    } else if (cpu_output) {
      auto texture_input = absl::get_if<OpenClTexture>(&input_obj);
      if (texture_input) {
        return queue_->EnqueueReadImage(
            texture_input->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_output->data);
      }
      auto buffer_input = absl::get_if<OpenClBuffer>(&input_obj);
      if (buffer_input) {
        return queue_->EnqueueReadBuffer(
            buffer_input->memobj, cpu_output->size_bytes, cpu_output->data);
      }
    }
    return absl::InternalError("Unexpected object");
  }

 private:
  std::array<size_t, 3> region_;
};

class OpenClTensorConverterBuilder : public TensorObjectConverterBuilder {
 public:
  explicit OpenClTensorConverterBuilder(Environment* environment)
      : environment_(environment) {}

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

  absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
    std::unique_ptr<OpenClConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    if (TrivialCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<TrivialCopier>();
    } else if (CpuCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<CpuCopier>();
    } else if (FromTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<FromTensorConverter>();
    } else if (ToTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<ToTensorConverter>();
    } else {
      return absl::UnimplementedError("Unsupported conversion");
    }
    RETURN_IF_ERROR(impl->Init(input, output, environment_));
    *converter = std::move(impl);
    return absl::OkStatus();
  }

  Environment* environment_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment) {
  return absl::make_unique<OpenClTensorConverterBuilder>(environment);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
