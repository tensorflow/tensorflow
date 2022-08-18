/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_UTILS_EVAL_UTILS_H_
#define TENSORFLOW_CORE_IR_UTILS_EVAL_UTILS_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"

namespace Eigen {
class ThreadPoolDevice;
}  // namespace Eigen

namespace mlir {
namespace tfg {
namespace util {

// A simple CPU device for operation evaluation.
class SimpleDevice : public tensorflow::DeviceBase {
 public:
  SimpleDevice();
  ~SimpleDevice() override;

  tensorflow::Status MakeTensorFromProto(
      const tensorflow::TensorProto& tensor_proto,
      const tensorflow::AllocatorAttributes alloc_attrs,
      tensorflow::Tensor* tensor) override;

  tensorflow::Allocator* GetAllocator(
      tensorflow::AllocatorAttributes attr) override;

 private:
  std::unique_ptr<tensorflow::thread::ThreadPool> eigen_worker_;
  tensorflow::DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
};

// Attempts to evaluates an MLIR Operation with the op registered kernel. The op
// is always executed on the local host CPU irrespective of the device attribute
// of the given op. The results will be filled in the results vector.
LogicalResult EvaluateOperation(tensorflow::DeviceBase* cpu_device,
                                tensorflow::ResourceMgr* resource_mgr, TFOp op,
                                ArrayRef<ElementsAttr> operands,
                                SmallVectorImpl<TypedAttr>& results);
}  // namespace util
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_UTILS_EVAL_UTILS_H_
