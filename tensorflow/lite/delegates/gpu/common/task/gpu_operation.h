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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/kernel_info.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/compiler_options.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tflite_serialization_base_generated.h"
#include "tensorflow/lite/delegates/gpu/common/task/tuning_type.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
// kCustom: default value
//   GPUOperation::GetGridSize must be overloaded
// kWBToX_HDToY_SToZ:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height() * dst_[0]->Depth();
//   grid_z = dst_[0]->Slices();
// kWBToX_HDToY_ZIs1:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height() * dst_[0]->Depth();
//   grid_z = 1;
// kWBToX_HToY_DToZ:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height();
//   grid_z = dst_[0]->Depth();
// kBToX_YIs1_ZIs1:
//   grid_x = dst_[0]->Batch();
//   grid_y = 1;
//   grid_z = 1;
enum class TensorToGrid {
  kCustom,
  kWBToX_HDToY_SToZ,
  kWBToX_HDToY_ZIs1,
  kWBToX_HToY_DToZ,
  kBToX_YIs1_ZIs1
};

struct OperationDef {
  CalculationsPrecision precision;
  std::vector<TensorDescriptor> src_tensors;
  std::vector<TensorDescriptor> dst_tensors;

  // returns FLOAT32 for F32 precision and FLOAT16 for F16 precision
  DataType GetDataType() const;
  // Primary means the first src tensor, because first tensor usually defines
  // the structure of kernel, all other resources(biases) types and etc.
  DataType GetPrimaryDataType() const;
  TensorStorageType GetPrimaryStorageType() const;
  bool IsBatchSupported() const;
};

struct ElementwiseDescriptor {
  Arguments args;
  std::string code;
};

class GPUOperation {
 public:
  GPUOperation() = default;
  explicit GPUOperation(const OperationDef& definition);
  virtual ~GPUOperation() = default;
  // Move only
  GPUOperation(GPUOperation&& operation);
  GPUOperation& operator=(GPUOperation&& operation);
  GPUOperation(const GPUOperation&) = delete;
  GPUOperation& operator=(const GPUOperation&) = delete;

  absl::Status AddOperation(const GpuInfo& gpu_info, GPUOperation* operation);

  int GetElementwiseInputsCount() const { return elementwise_inputs_; }

  void SetSrc(GpuSpatialTensor* ptr, int index = 0);
  void SetDst(GpuSpatialTensor* ptr, int index = 0);

  struct DispatchInfo {
    int3 work_group_size;
    int3 work_groups_count;
  };
  void GetPossibleDispatches(TuningType tuning_type, const GpuInfo& gpu_info,
                             const KernelInfo& kernel_info,
                             std::vector<DispatchInfo>* dispatches) const;

  const std::vector<std::string>& GetSrcTensorsNames() const {
    return src_tensors_names_;
  }
  const std::vector<std::string>& GetDstTensorsNames() const {
    return dst_tensors_names_;
  }
  const std::vector<GpuSpatialTensor*>& GetSrcTensors() const { return src_; }
  const std::vector<GpuSpatialTensor*>& GetDstTensors() const { return dst_; }
  const int3& GetWorkGroupsCount() const { return work_groups_count_; }

  absl::Status AssembleCode(const GpuInfo& gpu_info);

  const OperationDef& GetDefinition() const { return definition_; }
  CalculationsPrecision GetPrecision() const { return definition_.precision; }

  void AddSrcTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddSrcBuffer(const std::string& buffer_name,
                    const BufferDescriptor& desc);
  void AddDstTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);

  bool IsLinkable() const { return elementwise_; }

  virtual absl::Status BindArguments(ArgumentsBinder* args) {
    return absl::OkStatus();
  }
  void RecalculateGridSize() { grid_size_ = GetGridSize(); }
  void RecalculateWorkGroupsCount();

  Arguments args_;
  std::string code_;
  int3 work_group_size_ = int3(8, 4, 1);
  std::vector<CompilerOptions> compiler_options_;
  // not applicable to elementwise
  TensorToGrid tensor_to_grid_ = TensorToGrid::kCustom;

  // for profiling
  uint64_t flops_ = 0;
  // size in bytes of constant gpu_objects inside args_
  uint64_t const_args_size_ = 0;

  // Must be called before const generic objects in args_ released.
  void CalculateConstArgsSize();

 protected:
  friend flatbuffers::Offset<tflite::gpu::data::GPUOperation> Encode(
      const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const tflite::gpu::data::GPUOperation* fb_op,
                             GPUOperation* op);
  friend GPUOperation CreateGpuOperation(const OperationDef& definition,
                                         ElementwiseDescriptor&& descriptor);
  friend GPUOperation CreateGpuOperation(const OperationDef& definition,
                                         ElementwiseDescriptor&& descriptor,
                                         const BHWC& second_shape);

  friend absl::Status FuseElemWithElemInternal(
      const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
      const std::vector<std::pair<std::string, std::string>>& replacements,
      GPUOperation* result);
  friend absl::Status FuseSimpleElemWithSimpleElem(const GpuInfo& gpu_info,
                                                   GPUOperation&& elem0,
                                                   GPUOperation&& elem1,
                                                   GPUOperation* result);
  friend absl::Status Fuse2InputElemWithSimpleElemAsFirstInput(
      const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
      GPUOperation* result);
  friend absl::Status Fuse2InputElemWithSimpleElemAsSecondInput(
      const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
      GPUOperation* result);
  friend absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                                    GPUOperation&& elem0,
                                                    GPUOperation&& elem1,
                                                    GPUOperation&& elem_root,
                                                    GPUOperation* result);

  virtual int3 GetGridSize() const;
  virtual void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info, std::vector<int3>* work_groups) const;

  // Defines operation calculation precision and format of src/dst tensors.
  OperationDef definition_;
  std::vector<GpuSpatialTensor*> src_;
  std::vector<GpuSpatialTensor*> dst_;
  int grid_dimension_ = 3;  // can be 1, 2 or 3
  int3 work_group_launch_order_ = int3(0, 1, 2);
  int3 grid_size_ = int3(0, 0, 0);
  std::vector<std::string> src_tensors_names_;
  std::vector<std::string> dst_tensors_names_;

 private:
  absl::Status GetTensorDescriptor(const std::string& tensor_name,
                                   TensorDescriptor** resutl);
  absl::Status ResolveSecondElementwiseInput();
  int3 work_groups_count_ = int3(0, 0, 0);
  bool elementwise_ = false;      // temporary, used during op construction
  int elementwise_inputs_ = 0;    // can be {0, 1, 2}
  std::string
      second_elementwise_tensor_name_;  // used with elementwise_inputs_ = 2
  int linkable_count_ = 0;        // temporary, used during op construction
  std::string elementwise_code_;  // temporary, used during op construction
};

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor);

// For creating elementwise operations with 2 runtime inputs
GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor,
                                const BHWC& second_shape);

absl::Status FuseElemWithElemInternal(
    const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
    const std::vector<std::pair<std::string, std::string>>& replacements,
    GPUOperation* result);

//    input       input
//      |           |
//    elem0         |
//      |    -->  elem
//    elem1         |
//      |           |
//    output      output
absl::Status FuseSimpleElemWithSimpleElem(const GpuInfo& gpu_info,
                                          GPUOperation&& elem0,
                                          GPUOperation&& elem1,
                                          GPUOperation* result);

//      input           input
//     /    \             |
//  elem0    |            |
//     \    /      -->  elem
//     elem1              |
//       |                |
//     output           output
absl::Status Fuse2InputElemWithSimpleElemAsFirstInput(const GpuInfo& gpu_info,
                                                      GPUOperation&& elem0,
                                                      GPUOperation&& elem1,
                                                      GPUOperation* result);

//      input           input
//     /    \             |
//    |    elem0          |
//     \    /      -->  elem
//     elem1              |
//       |                |
//     output           output
absl::Status Fuse2InputElemWithSimpleElemAsSecondInput(const GpuInfo& gpu_info,
                                                       GPUOperation&& elem0,
                                                       GPUOperation&& elem1,
                                                       GPUOperation* result);

//      input           input
//     /    \             |
//  elem0  elem1          |
//     \    /      -->  elem
//   elem_root            |
//       |                |
//     output           output
absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                           GPUOperation&& elem0,
                                           GPUOperation&& elem1,
                                           GPUOperation&& elem_root,
                                           GPUOperation* result);
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_
