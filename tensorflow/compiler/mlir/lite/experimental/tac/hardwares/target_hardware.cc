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
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"

#include <algorithm>
#include <cctype>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
struct RegisteredTargetHardware {
  // TODO(b/177376459): Remove this constructor.
  RegisteredTargetHardware(const std::string& name,
                           const std::string& description, mlir::TypeID type_id,
                           std::unique_ptr<TargetHardware> target_hardware)
      : unique_name(GetCanonicalHardwareName(name)),
        description(description),
        type_id(type_id),
        target_hardware(std::move(target_hardware)) {}

  RegisteredTargetHardware(
      const std::string& name, const std::string& description,
      mlir::TypeID type_id,
      std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory)
      : unique_name(GetCanonicalHardwareName(name)),
        description(description),
        target_hardware_factory(target_hardware_factory) {}

  std::string unique_name;
  std::string description;
  mlir::TypeID type_id;
  std::unique_ptr<TargetHardware> target_hardware;
  std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory;
};

struct RegisteredTargetHardwareOps {
  explicit RegisteredTargetHardwareOps(mlir::TypeID hardware_type)
      : hardware_typeid(hardware_type) {}
  // Key is the Operation TypeID
  llvm::DenseMap<mlir::TypeID, std::unique_ptr<TargetHardwareOperation>>
      target_hardware_ops;
  // Key is the Operation TypeID
  llvm::DenseMap<mlir::TypeID,
                 std::function<std::unique_ptr<TargetHardwareOperation>()>>
      target_hardware_ops_factory;
  mlir::TypeID hardware_typeid;
};

std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>*
GetRegisteredTargetHardwareOps() {
  static std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>*
      hardwares_ops =
          []() -> std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>* {
    return new std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>();
  }();
  return hardwares_ops;
}

std::vector<RegisteredTargetHardware>* GetRegisteredHardwares() {
  static std::vector<RegisteredTargetHardware>* hardwares =
      []() -> std::vector<RegisteredTargetHardware>* {
    return new std::vector<RegisteredTargetHardware>();
  }();
  return hardwares;
}

llvm::DenseMap<mlir::TypeID, std::unique_ptr<TargetHardwareOperation>>*
getRegisteredOperationsForHardware(mlir::TypeID type_id) {
  auto* hardwares = GetRegisteredTargetHardwareOps();
  for (auto& hardware : *hardwares) {
    if (hardware->hardware_typeid == type_id) {
      return &hardware->target_hardware_ops;
    }
  }
  return nullptr;
}

// A deny list for op cost computation since those ops are not arithemtic.
inline bool IsNonArithmeticOp(mlir::Operation* op) {
  if (llvm::isa<func::ReturnOp, func::FuncOp>(op)) return true;
  if (op->hasTrait<OpTrait::ConstantLike>()) return true;
  if (llvm::isa<QConstOp, SparseQConstOp>(op)) return true;
  if (!NotTFLQuantDequantizeOp(op)) return true;
  return false;
}

}  // namespace

bool TargetHardware::Init() {
  auto* hardware_ops_factory = GetRegisteredTargetHardwareOps();
  for (auto& hardware_ops : *hardware_ops_factory) {
    if (hardware_ops->hardware_typeid != this->GetTypeId()) continue;
    auto& op_factories = hardware_ops->target_hardware_ops_factory;
    for (auto& op_factory : op_factories) {
      hardware_ops_.emplace_back(op_factory.getSecond()());
    }
    break;
  }
  return true;
}

double TargetHardware::GetOpCost(mlir::Operation* op) const {
  auto* registered_ops = getRegisteredOperationsForHardware(GetTypeId());
  if (registered_ops == nullptr) {
    return kDefaultFixedValuedCost;
  }
  auto abstract_op = op->getRegisteredInfo();
  auto hardware_op = registered_ops->find(abstract_op->getTypeID());
  if (hardware_op == registered_ops->end()) return kDefaultFixedValuedCost;
  return hardware_op->second->GetOpCost(op);
}

bool TargetHardware::IsOpSupported(mlir::Operation* op) const {
  auto* registered_ops = getRegisteredOperationsForHardware(GetTypeId());
  if (registered_ops == nullptr) {
    return false;
  }
  auto abstract_op = op->getRegisteredInfo();
  auto hardware_op = registered_ops->find(abstract_op->getTypeID());
  if (hardware_op == registered_ops->end()) return false;
  return hardware_op->second->IsOpSupported(op);
}

double TargetHardware::GetFuncCost(func::FuncOp* func) const {
  double total_cost = 0.0;
  func->walk([&](Operation* op) {
    if (IsNonArithmeticOp(op)) return;
    // We will always defer to the hardware to decide the cost.
    total_cost += GetOpCost(op);
  });
  return total_cost;
}

const TargetHardware* GetTargetHardware(const std::string& hardware_name) {
  const std::string canonical_name = GetCanonicalHardwareName(hardware_name);
  // Just loop for now, we don't expect number of hardwares to be huge.
  // Revisit to have map if number of elements increased.
  auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == canonical_name) {
      return hardware.target_hardware.get();
    }
  }
  return nullptr;
}

std::function<std::unique_ptr<TargetHardware>()> GetTargetHardwareFactory(
    const std::string& hardware_name) {
  const std::string canonical_name = GetCanonicalHardwareName(hardware_name);
  // Just loop for now, we don't expect number of hardwares to be huge.
  // Revisit to have map if number of elements increased.
  auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == canonical_name) {
      return hardware.target_hardware_factory;
    }
  }
  return nullptr;
}

namespace internal {

void RegisterTargetHardware(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
  auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == unique_name) {
      llvm::errs() << "Ignoring duplicate hardware. Hardware " << unique_name
                   << " already registered\n";
      return;
    }
  }
  registered_hardwares->push_back(RegisteredTargetHardware(
      unique_name, description, type_id, target_hardware_factory()));
}

void RegisterTargetHardwareFactory(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
  auto* registered_hardwares = GetRegisteredHardwares();
  for (auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == unique_name) {
      llvm::errs() << "Ignoring duplicate hardware. Hardware " << unique_name
                   << " already registered\n";
      hardware.target_hardware_factory = target_hardware_factory;
      return;
    }
  }
  registered_hardwares->push_back(RegisteredTargetHardware(
      unique_name, description, type_id, target_hardware_factory));
}

void RegisterTargetHardwareOp(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory) {
  auto* registered_hardware_ops = GetRegisteredTargetHardwareOps();
  for (auto& hardware : *registered_hardware_ops) {
    if (hardware->hardware_typeid == hardware_type) {
      if (hardware->target_hardware_ops.count(op_type)) {
        llvm::errs() << "Trying to register duplicate Op";
        return;
      }
      hardware->target_hardware_ops[op_type] = target_hardware_op_factory();
      return;
    }
  }
  registered_hardware_ops->push_back(
      std::make_unique<RegisteredTargetHardwareOps>(
          RegisteredTargetHardwareOps(hardware_type)));
  registered_hardware_ops->back()->target_hardware_ops[op_type] =
      target_hardware_op_factory();
}

void RegisterTargetHardwareOpFactory(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory) {
  auto* registered_hardware_ops = GetRegisteredTargetHardwareOps();
  for (auto& hardware : *registered_hardware_ops) {
    if (hardware->hardware_typeid == hardware_type) {
      if (hardware->target_hardware_ops_factory.count(op_type)) {
        llvm::errs() << "Trying to register duplicate Op";
        return;
      }
      hardware->target_hardware_ops_factory[op_type] =
          target_hardware_op_factory;
      return;
    }
  }
  registered_hardware_ops->push_back(
      std::make_unique<RegisteredTargetHardwareOps>(
          RegisteredTargetHardwareOps(hardware_type)));
  registered_hardware_ops->back()->target_hardware_ops_factory[op_type] =
      target_hardware_op_factory;
}

}  // namespace internal

bool ProcessTargetDevices(llvm::ArrayRef<std::string> specified_device_specs,
                          std::vector<std::string>* device_specs) {
  bool cpu_include = false;
  for (auto& device_spec : specified_device_specs) {
    auto device = GetCanonicalHardwareName(device_spec);

    if (device == "CPU") cpu_include = true;
    device_specs->push_back(device);
  }
  if (!cpu_include) {
    device_specs->push_back("CPU");
  }

  // Make sure all the devices are registered.
  for (const std::string& device : *device_specs) {
    if (GetTargetHardware(device) == nullptr) {
      llvm::errs() << "cannot get target hardware for device: " << device;
      return false;
    }
  }

  return true;
}

std::string GetHardwareName(const TargetHardware* hardware) {
  const auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& registered_hardware : *registered_hardwares) {
    if (registered_hardware.type_id == hardware->GetTypeId())
      return registered_hardware.unique_name;
  }
  return "";
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
