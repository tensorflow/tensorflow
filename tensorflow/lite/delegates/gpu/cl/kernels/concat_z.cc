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

#include "tensorflow/lite/delegates/gpu/cl/kernels/concat_z.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {
bool IsAllChannelsX4(const std::vector<int>& channels) {
  for (int channel : channels) {
    if (channel % 4 != 0) {
      return false;
    }
  }
  return true;
}

std::string GetConcatKernelCode(
    const OperationDef& definition, const std::vector<int>& channels,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::vector<std::shared_ptr<TensorCodeGenerator>> srcs(channels.size());
  for (int i = 0; i < channels.size(); ++i) {
    const std::string tensor_name = "src_data_" + std::to_string(i);
    const std::string uniform_name = "src_size_" + std::to_string(i);
    srcs[i] = std::shared_ptr<TensorCodeGenerator>(new TensorCodeGenerator(
        tensor_name, uniform_name, definition.src_tensors[i]));
  }
  TensorCodeGenerator dst("dst_data", "dst_size", definition.dst_tensors[0]);

  std::string code = GetCommonDefines(definition.precision);
  const std::string postfix[] = {".x", ".y", ".z", ".w"};

  code += "__kernel void main_function(\n";
  for (const auto& src : srcs) {
    code += src->GetDeclaration(AccessType::READ) + ",\n";
  }
  code += dst.GetDeclaration(AccessType::WRITE);
  code += GetArgsDeclaration(linked_operations);
  for (int i = 0; i < channels.size(); ++i) {
    const std::string uniform_name = "src_size_" + std::to_string(i);
    code += "    int4 " + uniform_name + ",\n";
  }
  code += "    int4 dst_size\n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  if (X >= dst_size.x || Y >= dst_size.y) { \n";
  code += "    return; \n";
  code += "  } \n";

  if (IsAllChannelsX4(channels)) {
    // When all channels % 4 == 0 we can read/assign/write FLT4 elements easily.
    // Also it is easy to write a loop in this case, to prevent long kernel
    // generation.
    code += "  int Z = 0;\n";
    for (int i = 0; i < channels.size(); ++i) {
      const std::string uniform_name = "src_size_" + std::to_string(i);
      const int depth = IntegralDivideRoundUp(channels[i], 4);
      if (depth % 2 == 0) {
        // We can read more at once inside of loop in case depth % 2 == 0
        // it should be better for reading latency hiding
        code += "  for (int i = 0; i < " + uniform_name + ".w; i += 2) {\n";
        code += "    FLT4 result0 = " +
                srcs[i]->Read3D("X", "Y", "i", TextureAddressMode::DONT_CARE) +
                ";\n";
        code +=
            "    FLT4 result1 = " +
            srcs[i]->Read3D("X", "Y", "i + 1", TextureAddressMode::DONT_CARE) +
            ";\n";
        code += "    " + dst.GetAddress("dst_adr0", "X", "Y", "Z") + "\n";
        code += "    " + dst.GetAddress("dst_adr1", "X", "Y", "Z + 1") + "\n";
        const LinkingContext context_0{"result0", "X", "Y", "Z"};
        const LinkingContext context_1{"result1", "X", "Y", "Z + 1"};
        code += PostProcess(linked_operations, context_0);
        code += PostProcess(linked_operations, context_1);
        code += "    " + dst.Write3D("result0", "X", "Y", "Z");
        code += "    " + dst.Write3D("result1", "X", "Y", "Z + 1");
        code += "    Z += 2;\n";
        code += "  }\n";
      } else {
        code += "  for (int i = 0; i < " + uniform_name + ".w; ++i) {\n";
        code += "    FLT4 result = " +
                srcs[i]->Read3D("X", "Y", "i", TextureAddressMode::DONT_CARE) +
                ";\n";
        const LinkingContext context{"result", "X", "Y", "Z"};
        code += PostProcess(linked_operations, context);
        code += "    " + dst.Write3D("result", "X", "Y", "Z");
        code += "    Z++;\n";
        code += "  }\n";
      }
    }
  } else {
    code += "  FLT4 result = (FLT4)(0.0);\n";
    int out_channel = 0;
    int read_index = 0;
    int z = 0;
    for (int i = 0; i < channels.size(); ++i) {
      const int depth = IntegralDivideRoundUp(channels[i], 4);
      for (int d = 0; d < depth; ++d) {
        const int channels_in_group = std::min(4, channels[i] - d * 4);
        const std::string temp_name = "t" + std::to_string(read_index);
        code += "  FLT4 " + temp_name + " = ";
        code += srcs[i]->Read3D("X", "Y", std::to_string(d),
                                TextureAddressMode::DONT_CARE) +
                ";\n";
        for (int c = 0; c < channels_in_group; ++c) {
          code += "  result" + postfix[out_channel] + " = ";
          code += temp_name + postfix[c] + ";\n";
          out_channel++;
          if (out_channel == 4) {
            out_channel = 0;
            code += "  {\n";
            const LinkingContext context{"result", "X", "Y", std::to_string(z)};
            code += PostProcess(linked_operations, context);
            code += "  " + dst.Write3D("result", "X", "Y", std::to_string(z));
            code += "  }\n";
            z++;
          }
        }
        read_index++;
      }
    }
    if (out_channel != 0) {
      code += "  {\n";
      const LinkingContext context{"result", "X", "Y", std::to_string(z)};
      code += PostProcess(linked_operations, context);
      code += "  " + dst.Write3D("result", "X", "Y", std::to_string(z));
      code += "  }\n";
    }
  }
  code += "}\n";
  return code;
}
}  // namespace

ConcatZ::ConcatZ(ConcatZ&& kernel)
    : GPUOperation(std::move(kernel)),
      channels_(std::move(kernel.channels_)),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

ConcatZ& ConcatZ::operator=(ConcatZ&& kernel) {
  if (this != &kernel) {
    channels_ = std::move(kernel.channels_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status ConcatZ::Compile(const CreationContext& creation_context) {
  const auto code =
      GetConcatKernelCode(definition_, channels_, linked_operations_);
  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F32 &&
      creation_context.device->IsPowerVR() && !IsAllChannelsX4(channels_)) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    options.push_back(CompilerOptions::CL_OPT_DISABLE);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConcatZ::BindArguments() {
  kernel_.ResetBindingCounter();
  for (int i = 0; i < channels_.size(); ++i) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[i]->GetMemoryPtr()));
  }
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  for (int i = 0; i < channels_.size(); ++i) {
    int4 size(src_[i]->Width(), src_[i]->Height(), channels_[i],
              IntegralDivideRoundUp(channels_[i], 4));
    RETURN_IF_ERROR(kernel_.SetBytesAuto(size));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  return OkStatus();
}

int3 ConcatZ::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

Status ConcatZ::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status ConcatZ::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

ConcatZ CreateConcatZ(const OperationDef& definition,
                      const std::vector<int>& channels) {
  return ConcatZ(definition, channels);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
