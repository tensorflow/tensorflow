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

#ifndef TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
#define TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_

#include <cstddef>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"

namespace mlir {
namespace detail {
// This class iterates over the control dependencies of the values.
template <typename ValueIteratorT>
class ControlRetIterator final
    : public llvm::mapped_iterator_base<ControlRetIterator<ValueIteratorT>,
                                        ValueIteratorT, Value> {
 public:
  using llvm::mapped_iterator_base<ControlRetIterator<ValueIteratorT>,
                                   ValueIteratorT, Value>::mapped_iterator_base;

  Value mapElement(Value value) const {
    return mlir::isa<tf_type::ControlType>(value.getType())
               ? value
               : tfg::LookupControlDependency(value);
  }
};
}  // namespace detail

namespace tfg {

// Wrapper class exposing convenience methods to manipulate TensorFlow graph
// nodes uniformly.
class TFOp {
 public:
  // Wrap an operation. The operation can be null. The constructor must be
  // marked as implicit to support `llvm::dyn_cast`.
  TFOp(Operation *op = nullptr);  // NOLINT

  explicit TFOp(Operation &op) : TFOp(&op) {}

  // Support LLVM-style RTTI.
  static bool classof(Operation *op) {
    return isa<TFGraphDialect>(op->getDialect());
  }

  // Get the wrapped operation.
  Operation *getOperation() { return op_; }

  // Returns a pointer to the TensorFlow Graph Dialect. It nevers returns
  // nullptr.
  TFGraphDialect *getDialect() {
    return cast<TFGraphDialect>(op_->getDialect());
  }

  // Split the operands into data and control operands.
  std::pair<OperandRange, OperandRange> splitOperands() {
    ControlType ctl_type = getDialect()->getControlType();
    return SplitDataAndControlValues(op_->getOperands(), ctl_type);
  }

  // Returns the regular operands, the control operands will be excluded.
  OperandRange getNonControlOperands() { return splitOperands().first; }

  // The control operands are always after the regular inputs.
  OperandRange getControlOperands() { return splitOperands().second; }

  // Returns the control token produced by this operation.
  Value controlRet() { return op_->getResult(op_->getNumResults() - 1); }

  // Returns the non-control results produced by this operation.
  ResultRange getNonControlResults() {
    return op_->getResults().slice(0, op_->getNumResults() - 1);
  }

  // Returns the node name for this operation.
  StringAttr nameAttr();
  StringRef name();
  // Set a new node name for this operation.
  void setName(const Twine &name);
  void setName(StringAttr name);

  // Returns the requested device, which is also the "device" field in a
  // GraphDef.
  StringAttr requestedDeviceAttr();
  StringRef requestedDevice();
  // Set a new requested device for this operation.
  void setRequestedDevice(const Twine &requested_device);
  void setRequestedDevice(StringAttr requested_device);

  // Returns the assigned device, this field is set by placer in general.
  StringAttr assignedDeviceAttr();
  StringRef assignedDevice();
  // Set a new assigned device for this operation.
  void setAssignedDevice(const Twine &assigned_device);
  void setAssignedDevice(StringAttr assigned_device);

  // Returns the assigned TPU cluster name.
  StringAttr tpuReplicate();
  // Set the assigned TPU cluster name.
  void setTpuReplicate(StringAttr tpu_replicate);

  // Returns the device, preferring the assigned device if set, and the
  // requested device otherwise.
  StringAttr deviceAttr() {
    StringAttr device = assignedDeviceAttr();
    if (device) {
      assert(!device.getValue().empty());
      return device;
    }
    return requestedDeviceAttr();
  }
  StringRef device() {
    StringAttr device_attr = deviceAttr();
    if (device_attr) return device_attr.getValue();
    return "";
  }

  // Forward `->` to the underlying operation, exposing the `Operation` methods.
  Operation *operator->() { return op_; }
  Operation &operator*() { return *op_; }

  // Converts to true if there is a wrapped operation.
  explicit operator bool() const { return op_; }

 private:
  // The wrapped operation.
  Operation *op_;
};

// A range iterator to get the control tokens associated with a value range.
// This range allows to wrap a ValueRange (or an OperandRange) and iterates on
// the control token associated to the producer of each value. For example, if
// you wrap the operands of an operation:
//     OperandControlRetRange range = op->getOperands();
// iterating this range will yield the control edges from each of the operations
// (or block arguments) producing these operands.
template <typename ValueRangeT>
class ControlRetRange final
    : public llvm::iterator_range<
          ::mlir::detail::ControlRetIterator<typename ValueRangeT::iterator>> {
 public:
  using Base = llvm::iterator_range<
      ::mlir::detail::ControlRetIterator<typename ValueRangeT::iterator>>;
  explicit ControlRetRange(ValueRangeT c) : Base(c.begin(), c.end()) {}

  /// Return the value at the given index.
  Value operator[](size_t index) const {
    assert(index < size() && "invalid index into value range");
    return *(this->begin() + index);
  }

  // Return the size of this range.
  size_t size() const { return llvm::size(*this); }

  // Return first value in the range.
  Value front() { return (*this)[0]; }

  // Compare this range with another.
  template <typename OtherT>
  bool operator==(const OtherT &other) const {
    return llvm::size(*this) == llvm::size(other) &&
           std::equal(this->begin(), this->end(), other.begin());
  }
  template <typename OtherT>
  bool operator!=(const OtherT &other) const {
    return !(*this == other);
  }
};

using OperandControlRetRange = ControlRetRange<OperandRange>;
using ValueControlRetRange = ControlRetRange<ValueRange>;

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
