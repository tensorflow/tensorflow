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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/tuning_parameters.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

struct CreationContext {
  const CLDevice* device;
  CLContext* context;
  CLCommandQueue* queue;
  ProgramCache* cache;
};

struct LinkingContext {
  // variable(FLT4) name to apply subsequent transformations
  std::string var_name;
  // x coordinate name (as it appears in kernel) for variable
  std::string x_coord;
  // y coordinate name (as it appears in kernel) for variable
  std::string y_coord;
  // s coordinate name (as it appears in kernel) for variable
  std::string s_coord;
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
  bool HasAllTensorsOfType(TensorStorageType storage_type) const;
  bool IsBatchSupported() const;
};

class ElementwiseOperation;

// GPUOperation represents some implementation of neural network operation on
// GPU. GPUOperation can contain ElementwiseOperation operations, in this case,
// ElementwiseOperation still hold necessary data and should be alive.
// When GPUOperation contains ElementwiseOperations, this GPUoperation replaces
// some sequence of operations Op + el_op0 + el_op1 + ...
// Because of this abilities of GPUOperation, usage scenario is next:
// Create instance of GPUOperation.
// Create all instances of ElementwiseOperations that we will(probably) attach
// to GPUOperation. Attach all ElementwiseOperations to GPUOperation. Call
// GPUOperation.Compile(). Don't call ElementwiseOperation.Compile() if it
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

  void AddOperation(ElementwiseOperation* operation);

  void SetSrc(Tensor* ptr, int index = 0);
  void SetDst(Tensor* ptr, int index = 0);

  virtual absl::Status AddToQueue(CLCommandQueue* queue) {
    return absl::OkStatus();
  }
  virtual absl::Status Tune(const TuningParameters& params) {
    return absl::OkStatus();
  }

  virtual absl::Status Compile(const CreationContext& creation_context) {
    return absl::OkStatus();
  }

  const OperationDef& GetDefinition() const { return definition_; }

 protected:
  // Defines operation calculation precision and format of src/dst tensors.
  OperationDef definition_;
  std::vector<Tensor*> src_;
  std::vector<Tensor*> dst_;
  Arguments args_;
  std::vector<ElementwiseOperation*> linked_operations_;
};

// ElementwiseOperation can be fused(linked) to another operation.
// field linked_ indicate about this
// link_index_ used mostly for generating of correct names for
//   linked code variables
// link_index_ is number of operation in sequence of linked operations
// and should be unique in this sequence
// link_index_ = 0 is equivalent that operation not linked.
class ElementwiseOperation : public GPUOperation {
 public:
  ElementwiseOperation() {}
  explicit ElementwiseOperation(const OperationDef& definition)
      : GPUOperation(definition) {}

  virtual ~ElementwiseOperation() {}
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;

  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  ElementwiseOperation(ElementwiseOperation&& operation);
  ElementwiseOperation& operator=(ElementwiseOperation&& operation);
  ElementwiseOperation(const ElementwiseOperation&) = delete;
  ElementwiseOperation& operator=(const ElementwiseOperation&) = delete;

  virtual absl::Status SetArgs(const std::string& unique_postfix,
                               Arguments* args) {
    return absl::OkStatus();
  }

  Arguments&& MoveArgs() { return std::move(args_); }
  std::string GetCode() const { return code_; }

  // ovveride to return false if for any reason operation can not be linked.
  virtual bool IsLinkable() const { return true; }

 protected:
  bool check_src_channels_size_ = false;
  std::string code_;
  absl::Status BindArguments();
  int3 GetGridSize() const;
  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

absl::Status MergeOperations(
    const std::vector<ElementwiseOperation*>& linked_ops,
    Arguments* merged_args, std::string* merged_code);

absl::Status SetArguments(const std::vector<ElementwiseOperation*>& linked_ops,
                          Arguments* args);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_
