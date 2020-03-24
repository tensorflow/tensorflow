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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/flt_type.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

// Performs the operation: {Quantize, Dequantize} on floating-point data.
// We need this operation to emulate the error introduced by quantization
// on the GPU, which cannot represent int8 tensors.
//
// Implemented as:
// qvalue = round((min(qmax, max(qmin, src_val)) - qmin) * (1/qscale) + 0.5)
// dq_value = qvalue * qscale + qmin
// Here, qmin, qmax & qscale refer to the quantization values as implemented in
// TensorFlow Lite's 'FakeQuant' kernel. round(x + 0.5) ensures we round away
// from zero.
//
// NOTE: We do not need to nudge min/max values in this op, since they would
// already be adjusted while generating the quantized model.
class QuantizeAndDequantize : public ElementwiseOperation {
 public:
  QuantizeAndDequantize() = default;
  // Move only
  QuantizeAndDequantize(QuantizeAndDequantize&& operation);
  QuantizeAndDequantize& operator=(QuantizeAndDequantize&& operation);
  QuantizeAndDequantize(const QuantizeAndDequantize&) = delete;
  QuantizeAndDequantize& operator=(const QuantizeAndDequantize&) = delete;

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;
  std::string GetArgsDeclaration() const override;
  Status BindArguments(CLKernel* kernel) override;

  friend Status CreateQuantizeAndDequantize(
      const CreationContext& creation_context, const OperationDef& definition,
      const QuantizeAndDequantizeAttributes& attr,
      QuantizeAndDequantize* result);

 private:
  QuantizeAndDequantize(const OperationDef& definition,
                        const QuantizeAndDequantizeAttributes& attr,
                        CalculationsPrecision scalar_precision);

  template <DataType T>
  Status UploadParameters(const ::tflite::gpu::Tensor<Linear, T>& parameters,
                          CLContext* context);

  FLT min_;
  FLT max_;
  FLT scale_;
};

Status CreateQuantizeAndDequantize(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const QuantizeAndDequantizeAttributes& attr,
                                   QuantizeAndDequantize* result);

template <DataType T>
Status QuantizeAndDequantize::UploadParameters(
    const ::tflite::gpu::Tensor<Linear, T>& parameters, CLContext* context) {
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition_.GetPrimaryStorageType());
  create_info.data_type = definition_.GetPrimaryDataType();
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_
