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
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conversion.h"
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
  absl::Status DispatchKernel(Buffer* buffer, Tensor* tensor) {
    RETURN_IF_ERROR(cl_args_.SetObjectRef("buffer", buffer));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("tensor", tensor));
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
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
  return type == DataType::FLOAT16 || type == DataType::FLOAT32 ||
         type == DataType::INT32 || type == DataType::BOOL;
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

    dst_tensor_descriptor_ =
        TensorDescriptor(output_def.object_def.data_type,
                         ToTensorStorageType(output_def.object_def.object_type,
                                             output_def.object_def.data_layout),
                         Layout::BHWC);

    GPUOperation gpu_op =
        CreateTensorToTensorOp(environment->GetDevicePtr()->GetInfo(),
                               src_tensor_descriptor_, dst_tensor_descriptor_);
    gpu_op.code_ =
        "#define MAIN_FUNCTION __kernel void tensor_to_tensor\n" + gpu_op.code_;

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    if (need_fp16_support) {
      gpu_op.code_ =
          "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" + gpu_op.code_;
    }
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(gpu_op.AssembleCode(environment->device().GetInfo()));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &gpu_op.args_, &gpu_op.code_));
    return environment->program_cache()->GetOrCreateCLKernel(
        gpu_op.code_, "tensor_to_tensor", environment->context(),
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

    BufferDescriptor buffer_desc;
    buffer_desc.element_type = output_def.object_def.data_type;
    buffer_desc.element_size = 1;
    buffer_desc.memory_type = MemoryType::GLOBAL;

    GPUOperation gpu_op =
        CreateTensorToBhwcBufferOp(environment->GetDevicePtr()->GetInfo(),
                                   tensor_descriptor_, buffer_desc);

    gpu_op.code_ =
        "#define MAIN_FUNCTION __kernel void tensor_to_bhwc\n" + gpu_op.code_;
    if (output_def.object_def.data_type == DataType::BOOL ||
        input_def.object_def.data_type == DataType::BOOL) {
      gpu_op.code_ =
          "#define convert_bool4(value) (convert_uchar4((value) != 0) & "
          "(uchar4) 1)\n#define bool4 uchar4\n" +
          gpu_op.code_;
    }

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    if (need_fp16_support) {
      gpu_op.code_ =
          "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" + gpu_op.code_;
    }
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(gpu_op.AssembleCode(environment->device().GetInfo()));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &gpu_op.args_, &gpu_op.code_));
    return environment->program_cache()->GetOrCreateCLKernel(
        gpu_op.code_, "tensor_to_bhwc", environment->context(),
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
    Buffer buffer = CreateBufferShared(output->memobj);
    return DispatchKernel(&buffer, &tensor);
  }
};

// Implements conversion from BHWC OpenCL buffer to OpenCL-specific tensor
// layout.
class BHWCBufferToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsBHWCOpenCLBuffer(input) && IsOpenCLTensor(output);
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    TensorStorageType dst_tensor_type = ToTensorStorageType(
        output_def.object_def.object_type, output_def.object_def.data_layout);
    tensor_descriptor_ = TensorDescriptor(output_def.object_def.data_type,

                                          dst_tensor_type, Layout::BHWC);

    BufferDescriptor buffer_desc;
    buffer_desc.element_type = input_def.object_def.data_type;
    buffer_desc.element_size = 1;
    buffer_desc.memory_type = MemoryType::GLOBAL;

    GPUOperation gpu_op =
        CreateBhwcBufferToTensorOp(environment->GetDevicePtr()->GetInfo(),
                                   buffer_desc, tensor_descriptor_);

    gpu_op.code_ =
        "#define MAIN_FUNCTION __kernel void bhwc_to_tensor\n" + gpu_op.code_;
    if (output_def.object_def.data_type == DataType::BOOL ||
        input_def.object_def.data_type == DataType::BOOL) {
      gpu_op.code_ =
          "#define convert_bool4(value) (convert_uchar4((value) != 0) & "
          "(uchar4) 1)\n#define bool4 uchar4\n" +
          gpu_op.code_;
    }
    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    if (need_fp16_support) {
      gpu_op.code_ =
          "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" + gpu_op.code_;
    }
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    RETURN_IF_ERROR(gpu_op.AssembleCode(environment->device().GetInfo()));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &gpu_op.args_, &gpu_op.code_));
    return environment->program_cache()->GetOrCreateCLKernel(
        gpu_op.code_, "bhwc_to_tensor", environment->context(),
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
    Buffer buffer = CreateBufferShared(input->memobj);
    return DispatchKernel(&buffer, &tensor);
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
    input_data_type_ = input_def.object_def.data_type;
    output_data_type_ = output_def.object_def.data_type;
    queue_ = environment->queue();
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto cpu_input = std::get_if<CpuMemory>(&input_obj);
    auto cpu_output = std::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      if (output_data_type_ == DataType::BOOL) {
        return CopyFromBoolCpu(cpu_input, output_obj);
      }
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
      if (input_data_type_ == DataType::BOOL) {
        return CopyToBoolCpu(input_obj, cpu_output);
      }
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
  absl::Status CopyToBoolCpu(const TensorObject& tensor_obj,
                             const CpuMemory* cpu_memory) {
    const size_t num_elements = cpu_memory->size_bytes;
    std::vector<uint8_t> tmp_data(num_elements);
    auto texture_input = std::get_if<OpenClTexture>(&tensor_obj);
    if (texture_input) {
      RETURN_IF_ERROR(queue_->EnqueueReadImage(
          texture_input->memobj, int3(region_[0], region_[1], region_[2]),
          tmp_data.data(), false));
    } else {
      auto buffer_input = std::get_if<OpenClBuffer>(&tensor_obj);
      if (!buffer_input) {
        return absl::InternalError("Unexpected object");
      }
      RETURN_IF_ERROR(queue_->EnqueueReadBuffer(
          buffer_input->memobj, tmp_data.size(), tmp_data.data(), false));
    }
    bool* output_data = reinterpret_cast<bool*>(cpu_memory->data);
    for (int i = 0; i < num_elements; ++i) {
      output_data[i] = tmp_data[i];
    }
    return absl::OkStatus();
  }

  absl::Status CopyFromBoolCpu(const CpuMemory* cpu_memory,
                               const TensorObject& tensor_obj) {
    const size_t num_elements = cpu_memory->size_bytes;
    const bool* bool_data = reinterpret_cast<bool*>(cpu_memory->data);
    tmp_bool_data_ = std::make_unique<std::vector<uint8_t>>();
    tmp_bool_data_->reserve(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      tmp_bool_data_->push_back(bool_data[i]);
    }
    auto texture_output = std::get_if<OpenClTexture>(&tensor_obj);
    if (texture_output) {
      return queue_->EnqueueWriteImage(texture_output->memobj,
                                       int3(region_[0], region_[1], region_[2]),
                                       tmp_bool_data_->data(), async_);
    }
    auto buffer_output = std::get_if<OpenClBuffer>(&tensor_obj);
    if (buffer_output) {
      return queue_->EnqueueWriteBuffer(buffer_output->memobj,
                                        tmp_bool_data_->size(),
                                        tmp_bool_data_->data(), async_);
    }
    return absl::InternalError("Unexpected object");
  }

  std::array<size_t, 3> region_;
  bool async_;
  DataType input_data_type_;
  DataType output_data_type_;
  std::unique_ptr<std::vector<uint8_t>> tmp_bool_data_;
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
