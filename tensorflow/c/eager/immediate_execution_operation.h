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
#ifndef TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_OPERATION_H_
#define TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_OPERATION_H_

#include <memory>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/managed_stack_trace.h"

struct TFE_Op;

namespace tensorflow {

class ImmediateExecutionContext;
class AbstractOpAttrs;
class CancellationManager;

// Abstract interface to an operation.
class ImmediateExecutionOperation : public AbstractOperation {
 public:
  virtual void Clear() = 0;

  // Returns the inputs of this op.
  virtual absl::Span<ImmediateExecutionTensorHandle* const> GetInputs()
      const = 0;
  virtual Status SetInput(size_t index,
                          ImmediateExecutionTensorHandle* input) = 0;

  virtual ImmediateExecutionContext* GetContext() const = 0;

  // Following two methods are used to support custom device.
  // Return true if the inputs contain custom device tensor handle. It means
  // that the argument need to be handled by a custom device.
  virtual bool HasCustomDeviceInput() const = 0;

  virtual const tensorflow::OpDef* OpDef() const = 0;

  virtual Status InputLength(const char* input_name, int* length) = 0;
  virtual Status OutputLength(const char* output_name, int* length) = 0;

  // Set stack trace to be used for potential async error reporting.
  virtual void SetStackTrace(ManagedStackTrace stack_trace) = 0;

  virtual const tensorflow::AbstractOpAttrs* GetOpAttrs() const = 0;
  virtual void AddAttrs(const AbstractOpAttrs* op_attrs) = 0;

  virtual void SetCancellationManager(
      CancellationManager* cancellation_manager) = 0;

  // Returns the stack trace set by `SetStackTrace` if exists.
  virtual absl::optional<ManagedStackTrace> GetStackTrace() = 0;

  virtual void SetStepId(int64_t step_id) = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
    return ptr->getKind() == kEager || ptr->getKind() == kTfrt;
  }

 protected:
  explicit ImmediateExecutionOperation(AbstractOperationKind kind)
      : AbstractOperation(kind) {}
  ~ImmediateExecutionOperation() override {}
};

namespace internal {
struct ImmediateExecutionOperationDeleter {
  void operator()(ImmediateExecutionOperation* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using ImmediateOpPtr =
    std::unique_ptr<ImmediateExecutionOperation,
                    internal::ImmediateExecutionOperationDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_OPERATION_H_
