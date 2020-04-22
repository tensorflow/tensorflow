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

std::string GetSrcDepthSizeVar(int src_index) {
  return "src_size_" + std::to_string(src_index) + "_depth";
}

std::string GetConcatKernelCode(
    const OperationDef& op_def, const std::vector<int>& channels,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::vector<TensorCodeGenerator> srcs(channels.size());
  for (int i = 0; i < channels.size(); ++i) {
    const std::string tensor_name = "src_data_" + std::to_string(i);
    srcs[i] = TensorCodeGenerator(
        tensor_name,
        WHSPoint{"dst_size.x", "dst_size.y", GetSrcDepthSizeVar(i)},
        op_def.src_tensors[i]);
  }
  TensorCodeGenerator dst("dst_data",
                          WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
                          op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  const std::string postfix[] = {".x", ".y", ".z", ".w"};

  c += "__kernel void main_function(\n";
  for (const auto& src : srcs) {
    c += src.GetDeclaration(AccessType::READ) + ",\n";
  }
  c += dst.GetDeclaration(AccessType::WRITE);
  c += GetArgsDeclaration(linked_operations);
  for (int i = 0; i < channels.size(); ++i) {
    c += "    int " + GetSrcDepthSizeVar(i) + ",\n";
  }
  c += "    int4 dst_size\n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y) return; \n";

  if (IsAllChannelsX4(channels)) {
    // When all channels % 4 == 0 we can read/assign/write FLT4 elements easily.
    // Also it is easy to write a loop in this case, to prevent long kernel
    // generation.
    c += "  int Z = 0;\n";
    for (int i = 0; i < channels.size(); ++i) {
      const int depth = DivideRoundUp(channels[i], 4);
      if (depth % 2 == 0) {
        // We can read more at once inside of loop in case depth % 2 == 0
        // it should be better for reading latency hiding
        c += "  for (int i = 0; i < " + GetSrcDepthSizeVar(i) + "; i += 2) {\n";
        c += "    FLT4 result0 = " + srcs[i].ReadWHS("X", "Y", "i") + ";\n";
        c += "    FLT4 result1 = " + srcs[i].ReadWHS("X", "Y", "i + 1") + ";\n";
        c += "    " + dst.GetAddressWHS("dst_adr0", "X", "Y", "Z") + "\n";
        c += "    " + dst.GetAddressWHS("dst_adr1", "X", "Y", "Z + 1") + "\n";
        const LinkingContext context_0{"result0", "X", "Y", "Z"};
        const LinkingContext context_1{"result1", "X", "Y", "Z + 1"};
        c += PostProcess(linked_operations, context_0);
        c += PostProcess(linked_operations, context_1);
        c += "    " + dst.WriteWHS("result0", "X", "Y", "Z");
        c += "    " + dst.WriteWHS("result1", "X", "Y", "Z + 1");
        c += "    Z += 2;\n";
        c += "  }\n";
      } else {
        c += "  for (int i = 0; i < " + GetSrcDepthSizeVar(i) + "; ++i) {\n";
        c += "    FLT4 result = " + srcs[i].ReadWHS("X", "Y", "i") + ";\n";
        const LinkingContext context{"result", "X", "Y", "Z"};
        c += PostProcess(linked_operations, context);
        c += "    " + dst.WriteWHS("result", "X", "Y", "Z");
        c += "    Z++;\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  FLT4 result = (FLT4)(0.0);\n";
    int out_channel = 0;
    int read_index = 0;
    int z = 0;
    for (int i = 0; i < channels.size(); ++i) {
      const int depth = DivideRoundUp(channels[i], 4);
      for (int d = 0; d < depth; ++d) {
        const int channels_in_group = std::min(4, channels[i] - d * 4);
        const std::string temp_name = "t" + std::to_string(read_index);
        c += "  FLT4 " + temp_name + " = ";
        c += srcs[i].ReadWHS("X", "Y", std::to_string(d)) + ";\n";
        for (int ch = 0; ch < channels_in_group; ++ch) {
          c += "  result" + postfix[out_channel] + " = ";
          c += temp_name + postfix[ch] + ";\n";
          out_channel++;
          if (out_channel == 4) {
            out_channel = 0;
            c += "  {\n";
            const LinkingContext context{"result", "X", "Y", std::to_string(z)};
            c += PostProcess(linked_operations, context);
            c += "  " + dst.WriteWHS("result", "X", "Y", std::to_string(z));
            c += "  }\n";
            z++;
          }
        }
        read_index++;
      }
    }
    if (out_channel != 0) {
      c += "  {\n";
      const LinkingContext context{"result", "X", "Y", std::to_string(z)};
      c += PostProcess(linked_operations, context);
      c += "  " + dst.WriteWHS("result", "X", "Y", std::to_string(z));
      c += "  }\n";
    }
  }
  c += "}\n";
  return c;
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

absl::Status ConcatZ::Compile(const CreationContext& creation_context) {
  const auto code =
      GetConcatKernelCode(definition_, channels_, linked_operations_);
  std::vector<CompilerOptions> options;
  if (creation_context.device->IsPowerVR() &&
      definition_.precision == CalculationsPrecision::F32 &&
      !IsAllChannelsX4(channels_)) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    options.push_back(CompilerOptions::CL_OPT_DISABLE);
  }
  if (creation_context.device->IsAMD() &&
      definition_.precision != CalculationsPrecision::F32 &&
      definition_.src_tensors[0].storage_type != TensorStorageType::BUFFER &&
      !IsAllChannelsX4(channels_)) {
    // BUG, some AMD gpus crashe without it
    options.push_back(CompilerOptions::CL_OPT_DISABLE);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ConcatZ::BindArguments() {
  kernel_.ResetBindingCounter();
  for (int i = 0; i < channels_.size(); ++i) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[i]->GetMemoryPtr()));
  }
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  for (int i = 0; i < channels_.size(); ++i) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[i]->Slices()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  return absl::OkStatus();
}

int3 ConcatZ::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

absl::Status ConcatZ::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status ConcatZ::AddToQueue(CLCommandQueue* queue) {
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
