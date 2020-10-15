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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/device_info.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/tuning_parameters.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization_generated.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

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

struct CreationContext {
  const CLDevice* device;
  CLContext* context;
  CLCommandQueue* queue;
  ProgramCache* cache;

  const DeviceInfo& GetDeviceInfo() const { return device->info_; }
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

  void SetSrc(Tensor* ptr, int index = 0);
  void SetDst(Tensor* ptr, int index = 0);

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  absl::Status AddToQueue(CLCommandQueue* queue) {
    RETURN_IF_ERROR(args_.Bind(kernel_.kernel()));
    return queue->Dispatch(kernel_, work_groups_count_, work_group_size_);
  }

  virtual void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info, std::vector<int3>* work_groups) const;

  absl::Status Tune(const TuningParameters& params);

  absl::Status AssembleCode(const DeviceInfo& device_info, CLContext* context);

  absl::Status Compile(const CreationContext& creation_context);

  absl::Status CompileDeserialized(const CreationContext& creation_context);

  virtual absl::Status PostCompileCheck(const DeviceInfo& device_info,
                                        const KernelInfo& kernel_info) {
    return absl::OkStatus();
  }

  const OperationDef& GetDefinition() const { return definition_; }

  void AddSrcTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddSrcBuffer(const std::string& buffer_name,
                    const BufferDescriptor& desc);
  void AddDstTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);

  bool IsLinkable() const { return elementwise_ && linkable_; }

  // for linking
  void AddUniquePostfix(const std::string& unique_postfix);

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

 protected:
  friend flatbuffers::Offset<data::GPUOperation> Encode(
      const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(CLContext* context,
                             const data::GPUOperation* fb_op, GPUOperation* op);

  virtual absl::Status BindArguments(ArgumentsBinder* args) {
    return absl::OkStatus();
  }
  virtual int3 GetGridSize() const;

  // Defines operation calculation precision and format of src/dst tensors.
  OperationDef definition_;
  std::vector<Tensor*> src_;
  std::vector<Tensor*> dst_;
  CLKernel kernel_;
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_
