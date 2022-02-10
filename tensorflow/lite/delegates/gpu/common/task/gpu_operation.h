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
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base_generated.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
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

// GPUOperation represents some implementation of neural network operation on
// GPU. GPUOperation can contain another GPU operations with flag elementwise_.
// When GPUOperation contains another GPU ops, this GPUoperation replaces
// some sequence of operations Op + op0 + op1 + ...
// Because of this abilities of GPUOperation, usage scenario is next:
// Create instance of GPUOperation.
// Create all instances of GPUOperations that we will(probably) attach
// to GPUOperation. Attach all GPUOperations to GPUOperation. Call
// GPUOperation.Compile(). Don't call GPUOperations.Compile() if it
// attached, it useless(and may be error)
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

  absl::Status AddOperation(GPUOperation* operation);

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

  virtual absl::Status PostCompileCheck(const GpuInfo& gpu_info,
                                        const KernelInfo& kernel_info) {
    return absl::OkStatus();
  }

  const OperationDef& GetDefinition() const { return definition_; }

  void AddSrcTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddSrcBuffer(const std::string& buffer_name,
                    const BufferDescriptor& desc);
  void AddSrcTexture2D(const std::string& texture_name,
                       const Texture2DDescriptor& desc);
  void AddDstTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);

  bool IsLinkable() const { return elementwise_ && linkable_; }

  // for linking
  void AddUniquePostfix(const std::string& unique_postfix);

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

  bool elementwise_ = false;
  // applicable only with elementwise_ = true;
  bool linkable_ = true;  // by default every elementwise is linkable
  // applicable only with elementwise_ = true;
  bool check_src_channels_size_ = false;

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
  int3 work_groups_count_ = int3(0, 0, 0);
  int linkable_count_ = 0;
  std::string elementwise_code_;  // temporary, used during op construction
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_
