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

#include "tensorflow/lite/delegates/gpu/cl/kernels/depthwise_conv_3x3.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateDepthwiseConvCode(const OperationDef& op_def,
                                      const CLDevice& device,
                                      bool weights_are_buffer,
                                      bool local_mem_uploads, Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  src_desc->SetTextureAddressMode(GetFastestZeroMode(device));
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));
  const auto src_tensor_type = op_def.src_tensors[0].storage_type;

  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  std::string c = GetCommonDefines(op_def.precision);
  if (local_mem_uploads) {
    c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
  }
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0) * 2;\n";
  c += "  int Y = get_global_id(1) * 2;\n";
  c += "  int S = get_global_id(2);\n";
  c += "   ACCUM_FLT4 r0 = (ACCUM_FLT4)(0.0f);\n";
  c += "   ACCUM_FLT4 r1 = (ACCUM_FLT4)(0.0f);\n";
  c += "   ACCUM_FLT4 r2 = (ACCUM_FLT4)(0.0f);\n";
  c += "   ACCUM_FLT4 r3 = (ACCUM_FLT4)(0.0f);\n";
  if (!local_mem_uploads) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| S >= args.dst_tensor.Slices()) { \n";
    c += "    return; \n";
    c += "  } \n";
  }
  if (local_mem_uploads) {
    c += "  __local FLT4 f[10];\n";
    c += "  event_t e = async_work_group_copy(f, args.weights.GetPtr() + S * "
         "10, 10, 0);\n";
    c += "  wait_group_events(1, &e);\n";
  } else if (weights_are_buffer) {
    c += "  __global FLT4* f = args.weights.GetPtr() + S * 10;\n";
  }
  c += "  FLT4 s0;\n";
  c += "  FLT4 s1;\n";
  c += "  FLT4 s2;\n";
  c += "  FLT4 s3;\n";
  std::string W[9] = {"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"};
  std::string bias = "bias";
  std::string xc[4] = {"X - 1", "X", "X + 1", "X + 2"};
  std::string yc[4] = {"Y - 1", "Y", "Y + 1", "Y + 2"};
  if (!weights_are_buffer) {
    c += "   FLT4 f0 = args.weights.Read(0, S);\n";
    c += "   FLT4 f1 = args.weights.Read(1, S);\n";
    c += "   FLT4 f2 = args.weights.Read(2, S);\n";
    c += "   FLT4 f3 = args.weights.Read(3, S);\n";
    c += "   FLT4 f4 = args.weights.Read(4, S);\n";
    c += "   FLT4 f5 = args.weights.Read(5, S);\n";
    c += "   FLT4 f6 = args.weights.Read(6, S);\n";
    c += "   FLT4 f7 = args.weights.Read(7, S);\n";
    c += "   FLT4 f8 = args.weights.Read(8, S);\n";
  }
  if (manual_clamp) {
    c += "  int x0 = X - 1;\n";
    c += "  int x1 = X;\n";
    c += "  int x2 = X + 1;\n";
    c += "  int x3 = X + 2;\n";
    c += "  int y0 = Y - 1;\n";
    c += "  int y1 = Y;\n";
    c += "  int y2 = Y + 1;\n";
    c += "  int y3 = Y + 2;\n";
    c += "  bool x0_in = x0 >= 0 && x0 < args.dst_tensor.Width();\n";
    c += "  bool x1_in = x1 >= 0 && x1 < args.dst_tensor.Width();\n";
    c += "  bool x2_in = x2 >= 0 && x2 < args.dst_tensor.Width();\n";
    c += "  bool x3_in = x3 >= 0 && x3 < args.dst_tensor.Width();\n";
    c += "  bool y0_in = y0 >= 0 && y0 < args.dst_tensor.Height();\n";
    c += "  bool y1_in = y1 >= 0 && y1 < args.dst_tensor.Height();\n";
    c += "  bool y2_in = y2 >= 0 && y2 < args.dst_tensor.Height();\n";
    c += "  bool y3_in = y3 >= 0 && y3 < args.dst_tensor.Height();\n";
    c += "  x0 = clamp(x0, 0, args.dst_tensor.Width() - 1);\n";
    c += "  x1 = clamp(x1, 0, args.dst_tensor.Width() - 1);\n";
    c += "  x2 = clamp(x2, 0, args.dst_tensor.Width() - 1);\n";
    c += "  x3 = clamp(x3, 0, args.dst_tensor.Width() - 1);\n";
    c += "  y0 = clamp(y0, 0, args.dst_tensor.Height() - 1);\n";
    c += "  y1 = clamp(y1, 0, args.dst_tensor.Height() - 1);\n";
    c += "  y2 = clamp(y2, 0, args.dst_tensor.Height() - 1);\n";
    c += "  y3 = clamp(y3, 0, args.dst_tensor.Height() - 1);\n";
    if (src_tensor_type == TensorStorageType::BUFFER) {
      c += "  __global FLT4* src_loc = "
           "args.src_tensor.GetPtrWithSliceOffset(S);\n";
    }
    xc[0] = "x0";
    xc[1] = "x1";
    xc[2] = "x2";
    xc[3] = "x3";
    yc[0] = "y0";
    yc[1] = "y1";
    yc[2] = "y2";
    yc[3] = "y3";
  }
  if (local_mem_uploads || weights_are_buffer) {
    W[0] = "f[0]";
    W[1] = "f[1]";
    W[2] = "f[2]";
    W[3] = "f[3]";
    W[4] = "f[4]";
    W[5] = "f[5]";
    W[6] = "f[6]";
    W[7] = "f[7]";
    W[8] = "f[8]";
    bias = "f[9]";
  }
  auto read_4x_line = [&](int y) {
    if (src_tensor_type == TensorStorageType::BUFFER) {
      const std::string y_in = "y" + std::to_string(y) + "_in";
      c += "    s0 = src_loc[args.src_tensor.GetWHOffset(" + xc[0] + ", " +
           yc[y] + ")] * (FLT)(x0_in && " + y_in + ");\n";
      c += "    s1 = src_loc[args.src_tensor.GetWHOffset(" + xc[1] + ", " +
           yc[y] + ")] * (FLT)(x1_in && " + y_in + ");\n";
      c += "    s2 = src_loc[args.src_tensor.GetWHOffset(" + xc[2] + ", " +
           yc[y] + ")] * (FLT)(x2_in && " + y_in + ");\n";
      c += "    s3 = src_loc[args.src_tensor.GetWHOffset(" + xc[3] + ", " +
           yc[y] + ")] * (FLT)(x3_in && " + y_in + ");\n";
    } else if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
      const std::string y_in = "y" + std::to_string(y) + "_in";
      c += "    s0 = args.src_tensor.Read(" + xc[0] + ", " + yc[y] +
           ", S) * (FLT)(x0_in && " + y_in + ");\n";
      c += "    s1 = args.src_tensor.Read(" + xc[1] + ", " + yc[y] +
           ", S) * (FLT)(x1_in && " + y_in + ");\n";
      c += "    s2 = args.src_tensor.Read(" + xc[2] + ", " + yc[y] +
           ", S) * (FLT)(x2_in && " + y_in + ");\n";
      c += "    s3 = args.src_tensor.Read(" + xc[3] + ", " + yc[y] +
           ", S) * (FLT)(x3_in && " + y_in + ");\n";
    } else {
      c += "    s0 = args.src_tensor.Read(" + xc[0] + ", " + yc[y] + ", S);\n";
      c += "    s1 = args.src_tensor.Read(" + xc[1] + ", " + yc[y] + ", S);\n";
      c += "    s2 = args.src_tensor.Read(" + xc[2] + ", " + yc[y] + ", S);\n";
      c += "    s3 = args.src_tensor.Read(" + xc[3] + ", " + yc[y] + ", S);\n";
    }
  };
  c += "  {\n";
  read_4x_line(0);
  c += "    r0 += TO_ACCUM_TYPE(" + W[0] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[1] + " * s1);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[0] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[2] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[1] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[2] + " * s3);\n";
  c += "  }\n";
  c += "  {\n";
  read_4x_line(1);
  c += "    r0 += TO_ACCUM_TYPE(" + W[3] + " * s0);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[0] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[4] + " * s1);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[3] + " * s1);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[1] + " * s1);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[0] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[5] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[4] + " * s2);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[2] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[1] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[5] + " * s3);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[2] + " * s3);\n";
  c += "  }\n";
  c += "  {\n";
  read_4x_line(2);
  c += "    r0 += TO_ACCUM_TYPE(" + W[6] + " * s0);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[3] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[7] + " * s1);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[6] + " * s1);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[4] + " * s1);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[3] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[8] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[7] + " * s2);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[5] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[4] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[8] + " * s3);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[5] + " * s3);\n";
  c += "  }\n";
  c += "  {\n";
  read_4x_line(3);
  c += "    r2 += TO_ACCUM_TYPE(" + W[6] + " * s0);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[7] + " * s1);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[6] + " * s1);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[8] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[7] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[8] + " * s3);\n";
  c += "  }\n";
  if (!weights_are_buffer) {
    c += "   FLT4 bias = args.weights.Read(9, S);\n";
  }
  c += "  r0 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  r1 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  r2 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  r3 += TO_ACCUM_TYPE(" + bias + ");\n";
  if (local_mem_uploads) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| "
         "S >= args.dst_tensor.Slices()) { \n";
    c += "    return; \n";
    c += "  } \n";
  }
  c += "  if(X + 0 < args.dst_tensor.Width() && Y + 0 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r0);\n";
  c += "    args.dst_tensor.Write(result, X + 0, Y + 0, S)\n";
  c += "  }\n";
  c += "  if(X + 1 < args.dst_tensor.Width() && Y + 0 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r1);\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y + 0, S)\n";
  c += "  }\n";
  c += "  if(X + 0 < args.dst_tensor.Width() && Y + 1 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r2);\n";
  c += "    args.dst_tensor.Write(result, X + 0, Y + 1, S)\n";
  c += "  }\n";
  c += "  if(X + 1 < args.dst_tensor.Width() && Y + 1 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r3);\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y + 1, S)\n";
  c += "  }\n";
  c += "}\n";

  return c;
}

}  // namespace

DepthwiseConv3x3::DepthwiseConv3x3(const OperationDef& definition,
                                   bool weights_are_buffer,
                                   bool local_mem_uploads)
    : GPUOperation(definition),
      weights_are_buffer_(weights_are_buffer),
      local_mem_uploads_(local_mem_uploads) {}

DepthwiseConv3x3::DepthwiseConv3x3(DepthwiseConv3x3&& operation)
    : GPUOperation(std::move(operation)),
      weights_are_buffer_(operation.weights_are_buffer_),
      local_mem_uploads_(operation.local_mem_uploads_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

DepthwiseConv3x3& DepthwiseConv3x3::operator=(DepthwiseConv3x3&& operation) {
  if (this != &operation) {
    std::swap(weights_are_buffer_, operation.weights_are_buffer_);
    std::swap(local_mem_uploads_, operation.local_mem_uploads_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status DepthwiseConv3x3::Compile(
    const CreationContext& creation_context) {
  std::string code = GenerateDepthwiseConvCode(
      definition_, *creation_context.device, weights_are_buffer_,
      local_mem_uploads_, &args_);
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));

  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsPowerVR()) {
    options.push_back(CompilerOptions::POWERVR_FP16);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status DepthwiseConv3x3::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 DepthwiseConv3x3::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width(), 2);
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status DepthwiseConv3x3::Tune(const TuningParameters& params) {
  if (local_mem_uploads_) {
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status DepthwiseConv3x3::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsDepthwiseConv3x3Supported(const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.dilations.w == 1 &&
         attr.dilations.h == 1 && attr.weights.shape.w == 3 &&
         attr.weights.shape.h == 3 && attr.strides.w == 1 &&
         attr.strides.h == 1 && attr.padding.prepended.w == 1 &&
         attr.padding.prepended.h == 1 && attr.padding.appended.w == 1 &&
         attr.padding.appended.h == 1;
}

absl::Status CreateDepthwiseConv3x3(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr, DepthwiseConv3x3* result) {
  if (!IsDepthwiseConv3x3Supported(attr)) {
    return absl::InvalidArgumentError(
        "DepthwiseConv3x3 doesn't support this attributes");
  }
  bool weights_are_buffer =
      creation_context.device->IsPowerVR() || creation_context.device->IsMali();
  bool local_mem_uploads =
      weights_are_buffer && creation_context.device->IsPowerVR();
  *result = DepthwiseConv3x3(definition, weights_are_buffer, local_mem_uploads);
  return result->UploadWeightsAndBiases(attr.weights, attr.bias,
                                        creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
