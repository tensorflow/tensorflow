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

#include "tensorflow/lite/delegates/gpu/gl/workgroups/ideal_workgroup_picker.h"

#include <map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// This code employs the results the workgroup performance reseach
// (b/117291356).

// Describes the ideal convolution for the specific operation case
// Case here means specific "kernel + strides" conbination for specific
// operatoins type, not sizes of input and output tensors, they can be any.
struct IdealByCase {
  bool ParamsAccepted(OperationType in_op_type, HW in_kernel,
                      HW in_strides) const {
    return operation_type == in_op_type && kernel == in_kernel &&
           strides == in_strides;
  }
  OperationType operation_type;
  HW kernel;
  HW strides;
  uint3 ideal_workgroup;
};

// Describes the ideal convolution for the type of operations. It means that
// any configuration of operation of this type will be working with top 10%
// performance with the particular GPU.
struct IdealByType {
  bool ParamsAccepted(OperationType in_op_type) const {
    return operation_type == in_op_type;
  }
  OperationType operation_type;
  uint3 ideal_workgroup;
};

// Describes ideal workgroups for the particular GPU model.
struct IdealWorkgroups {
  std::vector<IdealByType> by_type;
  std::vector<IdealByCase> by_case;
};

// List of Ideal workgroups which is received after the research mentioned
// above.

// Ideal workgroups for Adreno 630.
std::vector<IdealByType>* kIdealByTypeAdreno630Ptr =
    new std::vector<IdealByType>{
        {OperationType::CONVOLUTION_2D, uint3(4, 8, 4)},
        {OperationType::DEPTHWISE_CONVOLUTION, uint3(4, 4, 8)},
    };

std::vector<IdealByCase>* kIdealByCaseAdreno630Ptr =
    new std::vector<IdealByCase>{
        {OperationType::CONVOLUTION_2D, HW(1, 1), HW(1, 1), uint3(4, 8, 4)},
        {OperationType::CONVOLUTION_2D, HW(3, 3), HW(2, 2), uint3(8, 4, 4)},
        {OperationType::DEPTHWISE_CONVOLUTION, HW(1, 1), HW(1, 1),
         uint3(8, 4, 4)},
        {OperationType::DEPTHWISE_CONVOLUTION, HW(3, 3), HW(2, 2),
         uint3(4, 4, 4)},
    };

// Ideal workgroups for Adreno 540.
std::vector<IdealByType>* kIdealByTypeAdreno540Ptr =
    new std::vector<IdealByType>{
        {OperationType::CONVOLUTION_2D, uint3(8, 2, 2)},
        {OperationType::DEPTHWISE_CONVOLUTION, uint3(8, 8, 2)},
    };

std::vector<IdealByCase>* kIdealByCaseAdreno540Ptr =
    new std::vector<IdealByCase>{
        {OperationType::CONVOLUTION_2D, HW(1, 1), HW(1, 1), uint3(4, 2, 8)},
        {OperationType::CONVOLUTION_2D, HW(3, 3), HW(2, 2), uint3(8, 2, 8)},
        {OperationType::DEPTHWISE_CONVOLUTION, HW(1, 1), HW(1, 1),
         uint3(8, 4, 8)},
        {OperationType::DEPTHWISE_CONVOLUTION, HW(3, 3), HW(2, 2),
         uint3(4, 4, 8)},
    };

// Ideal workgroups for Adreno 510.
std::vector<IdealByType>* kIdealByTypeAdreno510Ptr =
    new std::vector<IdealByType>{
        {OperationType::CONVOLUTION_2D, uint3(8, 4, 4)},
        {OperationType::DEPTHWISE_CONVOLUTION, uint3(8, 4, 4)},
    };

std::vector<IdealByCase>* kIdealByCaseAdreno510Ptr =
    new std::vector<IdealByCase>{
        {OperationType::CONVOLUTION_2D, HW(1, 1), HW(1, 1), uint3(4, 2, 8)},
        {OperationType::CONVOLUTION_2D, HW(3, 3), HW(2, 2), uint3(8, 2, 8)},
        {OperationType::DEPTHWISE_CONVOLUTION, HW(1, 1), HW(1, 1),
         uint3(8, 4, 8)},
        {OperationType::DEPTHWISE_CONVOLUTION, HW(3, 3), HW(2, 2),
         uint3(4, 4, 8)},
    };

// Ideal workgroups for Adreno 509.
std::vector<IdealByType>* kIdealByTypeAdreno509Ptr =
    new std::vector<IdealByType>{
        {OperationType::CONVOLUTION_2D, uint3(8, 4, 8)},
        {OperationType::DEPTHWISE_CONVOLUTION, uint3(8, 8, 2)},
    };

// Ideal workgroups for Adreno 508, 506, 505, 418, 405
std::vector<IdealByType>* kIdealByTypeAdreno508Ptr =
    new std::vector<IdealByType>{
        {OperationType::CONVOLUTION_2D, uint3(8, 4, 8)},
        {OperationType::DEPTHWISE_CONVOLUTION, uint3(8, 4, 8)},
    };
std::vector<IdealByType>* kIdealByTypeAdreno506Ptr = kIdealByTypeAdreno508Ptr;
std::vector<IdealByType>* kIdealByTypeAdreno505Ptr = kIdealByTypeAdreno508Ptr;
std::vector<IdealByType>* kIdealByTypeAdreno418Ptr = kIdealByTypeAdreno508Ptr;
std::vector<IdealByType>* kIdealByTypeAdreno405Ptr = kIdealByTypeAdreno508Ptr;

// Put all ideal workgroups from the list together.
const std::map<GpuModel, IdealWorkgroups>* kIdealWorkgroupsInfoPtr =
    new std::map<GpuModel, IdealWorkgroups>{
        {GpuModel::ADRENO630,
         {*kIdealByTypeAdreno630Ptr, *kIdealByCaseAdreno630Ptr}},
        {GpuModel::ADRENO540, {*kIdealByTypeAdreno540Ptr, {}}},
        {GpuModel::ADRENO510,
         {*kIdealByTypeAdreno510Ptr, *kIdealByCaseAdreno510Ptr}},
        {GpuModel::ADRENO509, {*kIdealByTypeAdreno509Ptr, {}}},
        {GpuModel::ADRENO508, {*kIdealByTypeAdreno508Ptr, {}}},
        {GpuModel::ADRENO506, {*kIdealByTypeAdreno506Ptr, {}}},
        {GpuModel::ADRENO505, {*kIdealByTypeAdreno505Ptr, {}}},
        {GpuModel::ADRENO418, {*kIdealByTypeAdreno418Ptr, {}}},
        {GpuModel::ADRENO405, {*kIdealByTypeAdreno405Ptr, {}}},
    };

}  // namespace

uint3 GetIdealWorkgroupIfPossible(GpuModel gpu_model, OperationType op_type,
                                  HW kernel, HW strides, uint3 default_wg,
                                  OHWI workload) {
  // Research showed that ideal workgroup approach doesn't work well with
  // convolutions, which have small amount of output channels or output
  // height/width dimensions
  if (workload.o < 32 || workload.h <= 5 || workload.w <= 5) return default_wg;

  // If GPU was investigated
  if (!kIdealWorkgroupsInfoPtr->count(gpu_model)) {
    return default_wg;
  }

  // Try to find the ideal workgroup by the specific operation case, cause they
  // are expected to be better tuned than default "by type" cases
  for (const auto& specific_case :
       kIdealWorkgroupsInfoPtr->at(gpu_model).by_case) {
    if (specific_case.ParamsAccepted(op_type, kernel, strides)) {
      return specific_case.ideal_workgroup;
    }
  }

  // Try to find the ideal workgroup by the operation type
  for (const auto& default_case :
       kIdealWorkgroupsInfoPtr->at(gpu_model).by_type) {
    if (default_case.ParamsAccepted(op_type)) {
      return default_case.ideal_workgroup;
    }
  }

  // If no ideal workgroup is found, use the default workgroup suggested by each
  // operation.
  return default_wg;
}

uint3 GetIdealWorkgroupIfPossible(GpuModel gpu_model, OperationType op_type,
                                  HW kernel, HW strides, OHWI workload) {
  return GetIdealWorkgroupIfPossible(gpu_model, op_type, kernel, strides,
                                     kEmptyWorkgroupSize, workload);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
