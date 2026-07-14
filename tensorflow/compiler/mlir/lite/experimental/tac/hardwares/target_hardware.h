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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_TARGET_HARDWARE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_TARGET_HARDWARE_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace tac {

// Default fixed values for ops.
constexpr static float kDefaultFixedValuedCost = 1000000.0;

// This is just fake data.
constexpr static float kCrossHardwareTransferPerByteCost = 5.0f;

// This is just fake data.
constexpr static float kCrossHardwareTransferFixedCost = 10.f;

// Interface for an Operation capabilities which should be tied to
// a specific hardware.
// Users should implement the interface and use TargetHardwareOpRegistration
// for registering the operation.
class TargetHardwareOperation {
 public:
  virtual ~TargetHardwareOperation() = default;

  virtual double GetOpCost(mlir::Operation* op) const = 0;

  virtual bool IsOpSupported(mlir::Operation* op) const = 0;
};

// Abstract base class for a hardware.
// To introduce new hardware
// users should implement the interface and use TargetHardwareRegistration
// for registering the hardware.
// Subclasses must implement the pure virtual function interface and
// define static member variable that retrieves string identifying the Target
// Hardware. Example,
// class MyType : public TargetHardware {
//  public:
//   static constexpr char kId[] = "MyHardware";
// };
class TargetHardware {
 public:
  virtual ~TargetHardware() = default;

  // Initializes all TargetHardwareOperation registered for this hardware.
  // Users overriding this function, should call the base class method to
  // initialize the ops.
  virtual bool Init();

  // Returns the cost of running 'op' on this Hardware.
  virtual double GetOpCost(mlir::Operation* op) const;

  // Returns the cost of running the whole function on this hardware.
  // By default this is the sum of the cost of individual cost for each op.
  virtual double GetFuncCost(func::FuncOp* func) const;

  // Returns true if 'op' can run on this Hardware.
  virtual bool IsOpSupported(mlir::Operation* op) const;

  // Switching cost between from hardware and this hardware.
  // If both the hardwares are the same, the transfer cost is basically 0.
  virtual double GetHardwareSwitchingCost(const TargetHardware* from,
                                          size_t buffer_size) const = 0;

  // Returns a list of all patterns to apply for this hardware.
  virtual mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const = 0;

  // Returns TypeId for the provided hardware.
  // Usually should be something like mlir::TypeID::get<MyType>()
  virtual mlir::TypeID GetTypeId() const = 0;

  virtual void GetDependentDialects(mlir::DialectRegistry& registry) const {}

 protected:
  // All registered hardware ops.
  std::vector<std::unique_ptr<TargetHardwareOperation>> hardware_ops_;
};

// Returns pointer to the Hardware identified by 'hardware_name'.
// If not found nullptr is returned.
// DEPRECATED: Do not use, prefer GetTargetHardwareFactory instead.
const TargetHardware* GetTargetHardware(const std::string& hardware_name);

// Returns the factory method for the requested hardware if present.
std::function<std::unique_ptr<TargetHardware>()> GetTargetHardwareFactory(
    const std::string& hardware_name);

namespace internal {

void RegisterTargetHardwareFactory(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory);

// Registers the provided target hardware factory.
template <typename T>
void RegisterTargetHardwareFactory(
    const std::string& description,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
  RegisterTargetHardwareFactory(T::kId, description, mlir::TypeID::get<T>(),
                                target_hardware_factory);
}

// DEPRECATED: Do not use, prefer RegisterTargetHardwareOpFactory intstead.
void RegisterTargetHardwareOp(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory);

void RegisterTargetHardwareOpFactory(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory);
}  // namespace internal

// Register target hardware.
template <typename Hardware>
struct TargetHardwareRegistration {
  TargetHardwareRegistration(const std::string& description,
                             std::function<std::unique_ptr<TargetHardware>()>
                                 target_hardware_factory) {
    internal::RegisterTargetHardwareFactory<Hardware>(description,
                                                      target_hardware_factory);
  }
};

// Register Op capabilities for specific hardware.
template <typename Hardware, typename Op>
struct TargetHardwareOpRegistration {
  explicit TargetHardwareOpRegistration(
      std::function<std::unique_ptr<TargetHardwareOperation>()>
          target_hardware_op_factory) {
    // TODO(b/177376459): remove this.
    internal::RegisterTargetHardwareOp(mlir::TypeID::get<Hardware>(),
                                       mlir::TypeID::get<Op>(),
                                       target_hardware_op_factory);
    internal::RegisterTargetHardwareOpFactory(mlir::TypeID::get<Hardware>(),
                                              mlir::TypeID::get<Op>(),
                                              target_hardware_op_factory);
  }
};

//======== util functions ==========

// Process user specified device specs, will always add CPU if it's not there.
// specified_device_specs: ',' separated, like "GPU,DSP,CPU".
// device_specs: processed device specs enum.
bool ProcessTargetDevices(llvm::ArrayRef<std::string> specified_device_specs,
                          std::vector<std::string>* device_specs);

// Check whether two hardwares are the same.
inline bool IsSameHardware(const TargetHardware* lhs,
                           const TargetHardware* rhs) {
  return lhs->GetTypeId() == rhs->GetTypeId();
}

// Returns the ID identifying 'hardware'. This should match the ID defined
// in the hardware field ID.
// For example, if MyHardware is passed the value returned should match
// MyHardware::kId.
std::string GetHardwareName(const TargetHardware* hardware);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_TARGET_HARDWARE_H_
