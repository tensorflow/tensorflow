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
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
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

  void SetGpuInfo(const GpuInfo& info) { gpu_info_ = info; }

 protected:
  absl::Status DispatchKernel(cl_mem buffer_mem, Tensor* tensor) {
    kernel_.ResetBindingCounter();
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(buffer_mem));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("tensor", tensor));
    RETURN_IF_ERROR(
        cl_args_.Bind(kernel_.kernel(), kernel_.GetBindingCounter()));
    const int3 grid = int3(tensor->Width() * tensor->Batch(), tensor->Height(),
                           tensor->Slices());
    std::vector<int3> work_groups;
    GetPossibleWorkGroupsConv(TuningType::kFast, gpu_info_, kernel_.info_, grid,
                              &work_groups);
    const int3 work_group_size = work_groups[0];
    const int3 work_groups_count = GetWorkGroupsCount(grid, work_group_size);
    return queue_->Dispatch(kernel_, work_groups_count, work_group_size);
  }

  CLArguments cl_args_;
  BHWC shape_;
  CLKernel kernel_;
  TensorDescriptor tensor_descriptor_;
  GpuInfo gpu_info_;
  CLCommandQueue* queue_ = nullptr;
  const CLContext* context_ = nullptr;
};

bool IsSupportedDataType(DataType type) {
  return type == DataType::FLOAT16 || type == DataType::FLOAT32;
}

bool IsBHWCOpenCLBuffer(const ObjectDef& def) {
  return IsSupportedDataType(def.data_type) &&
         def.object_type == ObjectType::OPENCL_BUFFER &&
         def.data_layout == DataLayout::BHWC;
}

bool IsOpenCLTensor(const ObjectDef& def) {
  const bool is_buffer_tensor = def.object_type == ObjectType::OPENCL_BUFFER &&
                                def.data_layout == DataLayout::DHWC4;
  const bool is_image2d_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::HDWC4;
  const bool is_image2d_array_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::DHWC4;
  const bool is_single_image_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::BHWC;
  return IsSupportedDataType(def.data_type) &&
         (is_buffer_tensor || is_image2d_tensor || is_image2d_array_tensor ||
          is_single_image_tensor);
}

absl::Status GetOpenCLMemory(const TensorObject& obj, cl_mem* memory) {
  auto texture = std::get_if<OpenClTexture>(&obj);
  auto buffer = std::get_if<OpenClBuffer>(&obj);
  if (texture && texture->memobj) {
    *memory = texture->memobj;
  } else if (buffer && buffer->memobj) {
    *memory = buffer->memobj;
  } else {
    return absl::InvalidArgumentError("Missing OpenCL object.");
  }
  return absl::OkStatus();
}

// Implements conversion from OpenCL tensor to another OpenCL tensor.
class TensorToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsOpenCLTensor(input) && IsOpenCLTensor(output);
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    src_tensor_descriptor_ =
        TensorDescriptor(input_def.object_def.data_type,
                         ToTensorStorageType(input_def.object_def.object_type,
                                             input_def.object_def.data_layout),
                         Layout::BHWC);
    Arguments args;
    args.AddObjectRef(
        "src_tensor", AccessType::READ,
        std::make_unique<TensorDescriptor>(src_tensor_descriptor_));

    dst_tensor_descriptor_ =
        TensorDescriptor(output_def.object_def.data_type,
                         ToTensorStorageType(output_def.object_def.object_type,
                                             output_def.object_def.data_layout),
                         Layout::BHWC);
    args.AddObjectRef(
        "dst_tensor", AccessType::WRITE,
        std::make_unique<TensorDescriptor>(dst_tensor_descriptor_));

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    const std::string out_data_type =
        ToCLDataType(output_def.object_def.data_type);
    std::string shader_src;
    if (need_fp16_support) {
      shader_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    shader_src +=
        R"(__kernel void tensor_to_tensor($0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.dst_tensor.Batch();
  int b = linear_id % args.dst_tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.dst_tensor.Width() || y >= args.dst_tensor.Height() || d >= args.dst_tensor.Slices()) return;
)";
    shader_src += "  " + out_data_type + "4 input = args.src_tensor.Read<" +
                  out_data_type + ">(x, y, d, b);\n";
    shader_src += "  args.dst_tensor.Write(input, x, y, d, b);\n}";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(
        args.Compile(environment->device().GetInfo(), {}, &shader_src));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &args, &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "tensor_to_tensor", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    cl_mem in_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(input_obj, &in_memory));
    cl_mem out_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(output_obj, &out_memory));

    Tensor src_tensor;
    TensorDescriptor descriptor_with_shape = src_tensor_descriptor_;
    descriptor_with_shape.SetBHWCShape(shape_);
    RETURN_IF_ERROR(CreateTensorShared(*context_, in_memory,
                                       descriptor_with_shape, &src_tensor));
    Tensor dst_tensor;
    descriptor_with_shape = dst_tensor_descriptor_;
    descriptor_with_shape.SetBHWCShape(shape_);
    RETURN_IF_ERROR(CreateTensorShared(*context_, out_memory,
                                       descriptor_with_shape, &dst_tensor));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("src_tensor", &src_tensor));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("dst_tensor", &dst_tensor));
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    const int3 grid = int3(dst_tensor.Width() * dst_tensor.Batch(),
                           dst_tensor.Height(), dst_tensor.Slices());
    const int3 work_group_size = {16, 8, 1};
    const int3 work_groups_count = GetWorkGroupsCount(grid, work_group_size);
    return queue_->Dispatch(kernel_, work_groups_count, work_group_size);
  }

 private:
  TensorDescriptor src_tensor_descriptor_;
  TensorDescriptor dst_tensor_descriptor_;
};

// Implements conversion from OpenCL-specific tensor layout to BHWC OpenCL
// buffer.
class TensorToBHWCBufferConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsOpenCLTensor(input) && IsBHWCOpenCLBuffer(output);
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    TensorStorageType src_tensor_type = ToTensorStorageType(
        input_def.object_def.object_type, input_def.object_def.data_layout);
    tensor_descriptor_ = TensorDescriptor(input_def.object_def.data_type,
                                          src_tensor_type, Layout::BHWC);
    Arguments args;
    args.AddObjectRef("tensor", AccessType::READ,
                      std::make_unique<TensorDescriptor>(tensor_descriptor_));

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    std::string shader_src;
    if (need_fp16_support) {
      shader_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    const std::string out_data_type =
        ToCLDataType(output_def.object_def.data_type);
    shader_src += "__kernel void tensor_to_bhwc(";
    shader_src += "__global " + out_data_type + "* dst, $0) {\n";
    shader_src += R"(  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
)";
    shader_src += "  " + out_data_type + "4 input = args.tensor.Read<" +
                  out_data_type + ">(x, y, d, b);\n";
    shader_src += R"(  int c = d * 4;
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
  }
})";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(
        args.Compile(environment->device().GetInfo(), {}, &shader_src));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &args, &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "tensor_to_bhwc", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto output = std::get_if<OpenClBuffer>(&output_obj);
    if (!output || !output->memobj) {
      return absl::InvalidArgumentError(
          "Missing output in tensor_to_bhwc converter");
    }

    cl_mem in_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(input_obj, &in_memory));
    Tensor tensor;
    TensorDescriptor descriptor_with_shape = tensor_descriptor_;
    descriptor_with_shape.SetBHWCShape(shape_);
    RETURN_IF_ERROR(CreateTensorShared(*context_, in_memory,
                                       descriptor_with_shape, &tensor));
    return DispatchKernel(output->memobj, &tensor);
  }
};

// Implements conversion from BHWC OpenCL buffer to OpenCL-specific tensor
// layout.
class BHWCBufferToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsBHWCOpenCLBuffer(input) && IsOpenCLTensor(output);
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
    auto params_kernel = GetFromBhwcKernel(input_def, output_def);

    TensorStorageType dst_tensor_type = ToTensorStorageType(
        output_def.object_def.object_type, output_def.object_def.data_layout);
    tensor_descriptor_ = TensorDescriptor(output_def.object_def.data_type,
                                          dst_tensor_type, Layout::BHWC);
    Arguments args;
    args.AddObjectRef("tensor", AccessType::WRITE,
                      std::make_unique<TensorDescriptor>(tensor_descriptor_));

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    std::string shader_src;
    if (need_fp16_support) {
      shader_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    const std::string in_data_type =
        ToCLDataType(input_def.object_def.data_type);
    const std::string out_data_type =
        ToCLDataType(output_def.object_def.data_type);
    shader_src += "__kernel void bhwc_to_tensor(";
    shader_src += "__global " + in_data_type + "* src, $0) {\n";

    shader_src += R"(  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);

  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
)";
    shader_src += "  " + out_data_type + "4 result;\n";
    shader_src += R"(  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  result.x = src[index];
  result.y = c + 1 < args.tensor.Channels() ? src[index + 1] : 1;
  result.z = c + 2 < args.tensor.Channels() ? src[index + 2] : 2;
  result.w = c + 3 < args.tensor.Channels() ? src[index + 3] : 3;
)";
    shader_src += "  args.tensor.Write(result, x, y, d, b);\n}";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    RETURN_IF_ERROR(
        args.Compile(environment->device().GetInfo(), {}, &shader_src));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &args, &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "bhwc_to_tensor", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto input = std::get_if<OpenClBuffer>(&input_obj);
    if (!input || !input->memobj) {
      return absl::InvalidArgumentError(
          "Missing input in bhwc_to_tensor converter");
    }
    cl_mem out_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(output_obj, &out_memory));
    Tensor tensor;
    TensorDescriptor descriptor_with_shape = tensor_descriptor_;
    descriptor_with_shape.SetBHWCShape(shape_);
    RETURN_IF_ERROR(CreateTensorShared(*context_, out_memory,
                                       descriptor_with_shape, &tensor));
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
    auto texture_input = std::get_if<OpenClTexture>(&input_obj);
    auto texture_output = std::get_if<OpenClTexture>(&output_obj);
    if (texture_input && texture_output) {
      return Copy(*texture_input, *texture_output);
    }
    auto buffer_input = std::get_if<OpenClBuffer>(&input_obj);
    auto buffer_output = std::get_if<OpenClBuffer>(&output_obj);
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
  explicit CpuCopier(bool asynchronous = false) : async_(asynchronous) {}
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
    auto cpu_input = std::get_if<CpuMemory>(&input_obj);
    auto cpu_output = std::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      auto texture_output = std::get_if<OpenClTexture>(&output_obj);
      if (texture_output) {
        return queue_->EnqueueWriteImage(
            texture_output->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_input->data, async_);
      }
      auto buffer_output = std::get_if<OpenClBuffer>(&output_obj);
      if (buffer_output) {
        return queue_->EnqueueWriteBuffer(buffer_output->memobj,
                                          cpu_input->size_bytes,
                                          cpu_input->data, async_);
      }
    } else if (cpu_output) {
      auto texture_input = std::get_if<OpenClTexture>(&input_obj);
      if (texture_input) {
        return queue_->EnqueueReadImage(
            texture_input->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_output->data, async_);
      }
      auto buffer_input = std::get_if<OpenClBuffer>(&input_obj);
      if (buffer_input) {
        return queue_->EnqueueReadBuffer(buffer_input->memobj,
                                         cpu_output->size_bytes,
                                         cpu_output->data, async_);
      }
    }
    return absl::InternalError("Unexpected object");
  }

 private:
  std::array<size_t, 3> region_;
  bool async_;
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
            TensorToTensorConverter::IsSupported(input_def, output_def) ||
            CpuCopier::IsSupported(input_def, output_def) ||
            TensorToBHWCBufferConverter::IsSupported(input_def, output_def) ||
            BHWCBufferToTensorConverter::IsSupported(input_def, output_def));
  }

  absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
    std::unique_ptr<OpenClConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    if (TrivialCopier::IsSupported(input_def, output_def)) {
      impl = std::make_unique<TrivialCopier>();
    } else if (TensorToTensorConverter::IsSupported(input_def, output_def)) {
      impl = std::make_unique<TensorToTensorConverter>();
    } else if (CpuCopier::IsSupported(input_def, output_def)) {
      impl = std::make_unique<CpuCopier>(/*asynchronous*/ true);
    } else if (TensorToBHWCBufferConverter::IsSupported(input_def,
                                                        output_def)) {
      impl = std::make_unique<TensorToBHWCBufferConverter>();
    } else if (BHWCBufferToTensorConverter::IsSupported(input_def,
                                                        output_def)) {
      impl = std::make_unique<BHWCBufferToTensorConverter>();
    } else {
      return absl::UnimplementedError("Unsupported conversion");
    }
    RETURN_IF_ERROR(impl->Init(input, output, environment_));
    impl->SetGpuInfo(environment_->GetDevicePtr()->GetInfo());
    *converter = std::move(impl);
    return absl::OkStatus();
  }

  Environment* environment_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment) {
  return std::make_unique<OpenClTensorConverterBuilder>(environment);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
