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

#include "tensorflow/lite/delegates/gpu/common/tasks/prelu.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

ElementwiseDescriptor CreatePReLU(const PReLUAttributes& attr,
                                  TensorDescriptor tensor_desc) {
  ElementwiseDescriptor op_desc;
  std::string alpha_read;
  auto alpha_linear =
      absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (alpha_linear) {
    TensorLinearDescriptor desc;
    desc.storage_type = DeduceLinearStorageType(tensor_desc.GetStorageType());
    desc.element_type = tensor_desc.GetDataType();
    desc.UploadLinearData(*alpha_linear);
    op_desc.args.AddObject(
        "alpha", std::make_unique<TensorLinearDescriptor>(std::move(desc)));
    alpha_read = "FLT4 alpha_val = args.alpha.Read(S_COORD);\n";
  }

  auto alpha_hwc =
      absl::get_if<tflite::gpu::Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (alpha_hwc) {
    const BHWC shape =
        BHWC(1, alpha_hwc->shape.h, alpha_hwc->shape.w, alpha_hwc->shape.c);
    TensorDescriptor const_tensor_desc = tensor_desc;
    const_tensor_desc.UploadData(*alpha_hwc);
    op_desc.args.AddObject("alpha", std::make_unique<TensorDescriptor>(
                                        std::move(const_tensor_desc)));
    const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
    const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
    const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
    alpha_read = absl::StrCat("FLT4 alpha_val = args.alpha.Read(", x_coord,
                              ", ", y_coord, ", ", s_coord, ");\n");
    if (shape.c == 1) {
      alpha_read += "  alpha_val.y = alpha_val.x;\n";
      alpha_read += "  alpha_val.z = alpha_val.x;\n";
      alpha_read += "  alpha_val.w = alpha_val.x;\n";
    }
  }

  op_desc.code = alpha_read +
                 "out_value = max(INIT_FLT4(0.0f), in_value) + "
                 "min(INIT_FLT4(0.0f), "
                 "in_value) * alpha_val;";
  return op_desc;
}

GPUOperation CreatePReLU(const GpuInfo& gpu_info,
                         const OperationDef& definition,
                         const PReLUAttributes& attr) {
  ElementwiseDescriptor op_desc = CreatePReLU(attr, definition.src_tensors[0]);
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace gpu
}  // namespace tflite
