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

std::string GetConcatKernelCode(const OperationDef& op_def,
                                const std::vector<int>& channels,
                                Arguments* args) {
  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
    auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[i]);
    if (op_def.IsBatchSupported()) {
      src_desc->SetStateVar("BatchedWidth", "true");
    }
    args->AddObjectRef(tensor_names[i], AccessType::READ, std::move(src_desc));
  }
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int Z = get_global_id(2);\n";
    c += "  if (Z >= args.dst_tensor.Depth()) return;\n";
    coords = "X, Y, Z";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";

  if (IsAllChannelsX4(channels)) {
    // When all channels % 4 == 0 we can read/assign/write FLT4 elements easily.
    // Also it is easy to write a loop in this case, to prevent long kernel
    // generation.
    c += "  int S = 0;\n";
    for (int i = 0; i < channels.size(); ++i) {
      std::string t_name = "args." + tensor_names[i];
      const int depth = DivideRoundUp(channels[i], 4);
      if (depth % 2 == 0) {
        // We can read more at once inside of loop in case depth % 2 == 0
        // it should be better for reading latency hiding
        c += "  for (int i = 0; i < " + t_name + ".Slices(); i += 2) {\n";
        c += "    FLT4 result0 = " + t_name + ".Read(" + coords + ", i);\n";
        c += "    FLT4 result1 = " + t_name + ".Read(" + coords + ", i + 1);\n";
        c += "    args.dst_tensor.Write(result0, " + coords + ", S);\n";
        c += "    args.dst_tensor.Write(result1, " + coords + ", S + 1);\n";
        c += "    S += 2;\n";
        c += "  }\n";
      } else {
        c += "  for (int i = 0; i < " + t_name + ".Slices(); ++i) {\n";
        c += "    FLT4 result = " + t_name + ".Read(" + coords + ", i);\n";
        c += "    args.dst_tensor.Write(result, " + coords + ", S);\n";
        c += "    S++;\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  FLT4 result = (FLT4)(0.0);\n";
    int out_channel = 0;
    int read_index = 0;
    int z = 0;
    const std::string postfix[] = {".x", ".y", ".z", ".w"};
    for (int i = 0; i < channels.size(); ++i) {
      std::string tensor_name = "args." + tensor_names[i];
      const int depth = DivideRoundUp(channels[i], 4);
      for (int d = 0; d < depth; ++d) {
        const int channels_in_group = std::min(4, channels[i] - d * 4);
        const std::string temp_name = "t" + std::to_string(read_index);
        c += "  FLT4 " + temp_name + " = " + tensor_name + ".Read(" + coords +
             ", " + std::to_string(d) + ");\n";
        for (int ch = 0; ch < channels_in_group; ++ch) {
          c += "  result" + postfix[out_channel] + " = ";
          c += temp_name + postfix[ch] + ";\n";
          out_channel++;
          if (out_channel == 4) {
            out_channel = 0;
            c += "  args.dst_tensor.Write(result, " + coords + ", " +
                 std::to_string(z) + ");\n";
            z++;
          }
        }
        read_index++;
      }
    }
    if (out_channel != 0) {
      c += "  args.dst_tensor.Write(result, " + coords + ", " +
           std::to_string(z) + ");\n";
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
  std::string code = GetConcatKernelCode(definition_, channels_, &args_);
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
    // BUG, some AMD gpus crash without it
    options.push_back(CompilerOptions::CL_OPT_DISABLE);
  }
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ConcatZ::BindArguments() {
  for (int i = 0; i < definition_.src_tensors.size(); ++i) {
    RETURN_IF_ERROR(
        args_.SetObjectRef("src_tensor_" + std::to_string(i), src_[i]));
  }
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 ConcatZ::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
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
