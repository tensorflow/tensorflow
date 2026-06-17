/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/resize.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

Resize::Resize(const OperationDef& definition, const Resize2DAttributes& attr)
    : GPUOperation(definition), attr_(attr) {
  code_ = GetResizeCode(definition_, attr_);
}

Resize::Resize(Resize&& operation)
    : GPUOperation(std::move(operation)), attr_(operation.attr_) {}

Resize& Resize::operator=(Resize&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Resize::GetResizeCode(const OperationDef& op_def,
                                  const Resize2DAttributes& attr) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| Z >= args.dst_tensor.Slices()) return;\n";
  if (attr.half_pixel_centers) {
    c += "  float f_coords_x = (INIT_FLOAT(X) + 0.5f) * args.scale_factor_x;\n";
    c += "  float f_coords_y = (INIT_FLOAT(Y) + 0.5f) * args.scale_factor_y;\n";
  } else {
    c += "  float f_coords_x = INIT_FLOAT(X) * args.scale_factor_x;\n";
    c += "  float f_coords_y = INIT_FLOAT(Y) * args.scale_factor_y;\n";
  }
  c += "  FLT4 r0;\n";
  if (attr.type == SamplingType::NEAREST) {
    if (attr.align_corners) {
      c += "  f_coords_x += 0.5f;";
      c += "  f_coords_y += 0.5f;";
    }
    c += "  args.src_tensor.ReadNearest(r0, f_coords_x, f_coords_y, Z);\n";
  } else {
    if (attr.half_pixel_centers) {
      c += "  f_coords_x -= 0.5f;";
      c += "  f_coords_y -= 0.5f;";
    }
    c += "  args.src_tensor.ReadBilinear(r0, f_coords_x, f_coords_y, Z);\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z);\n";
  c += "}\n";
  return c;
}

absl::Status Resize::BindArguments(ArgumentsBinder* args) {
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  return absl::OkStatus();
}

int3 Resize::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Resize CreateResize(const OperationDef& definition,
                    const Resize2DAttributes& attr) {
  return Resize(definition, attr);
}

Resize3D::Resize3D(const OperationDef& definition,
                   const Resize3DAttributes& attr)
    : GPUOperation(definition), attr_(attr) {
  code_ = GetResize3DCode(definition_, attr_);
}

Resize3D::Resize3D(Resize3D&& operation)
    : GPUOperation(std::move(operation)), attr_(operation.attr_) {}

Resize3D& Resize3D::operator=(Resize3D&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Resize3D::GetResize3DCode(const OperationDef& op_def,
                                      const Resize3DAttributes& attr) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");
  args_.AddFloat("scale_factor_z");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int linear_id_z = GLOBAL_ID_2;\n";
  c += "  int S = linear_id_z % args.dst_tensor.Slices();\n";
  c += "  int Z = linear_id_z / args.dst_tensor.Slices();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| Z >= args.dst_tensor.Depth()) return;\n";
  if (attr.half_pixel_centers) {
    c += "  float f_coords_x = (INIT_FLOAT(X) + 0.5f) * args.scale_factor_x;\n";
    c += "  float f_coords_y = (INIT_FLOAT(Y) + 0.5f) * args.scale_factor_y;\n";
    c += "  float f_coords_z = (INIT_FLOAT(Z) + 0.5f) * args.scale_factor_z;\n";
  } else {
    c += "  float f_coords_x = INIT_FLOAT(X) * args.scale_factor_x;\n";
    c += "  float f_coords_y = INIT_FLOAT(Y) * args.scale_factor_y;\n";
    c += "  float f_coords_z = INIT_FLOAT(Z) * args.scale_factor_z;\n";
  }
  c += "  FLT4 r0;\n";
  if (attr.type == SamplingType::NEAREST) {
    if (attr.align_corners) {
      c += "  f_coords_x += 0.5f;";
      c += "  f_coords_y += 0.5f;";
      c += "  f_coords_z += 0.5f;";
    }
    c += "  args.src_tensor.ReadNearest(r0, f_coords_x, f_coords_y, "
         "f_coords_z, S);\n";
  } else {
    if (attr.half_pixel_centers) {
      c += "  f_coords_x -= 0.5f;";
      c += "  f_coords_y -= 0.5f;";
      c += "  f_coords_z -= 0.5f;";
    }
    c += "  args.src_tensor.ReadBilinear(r0, f_coords_x, f_coords_y, "
         "f_coords_z, S);\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z, S);\n";
  c += "}\n";
  return c;
}

absl::Status Resize3D::BindArguments(ArgumentsBinder* args) {
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_z",
      CalculateResizeScale(src_[0]->Depth(), dst_[0]->Depth(), attr_)));
  return absl::OkStatus();
}

int3 Resize3D::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Resize3D CreateResize3D(const OperationDef& definition,
                        const Resize3DAttributes& attr) {
  return Resize3D(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
