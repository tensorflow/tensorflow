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

#ifndef XLA_MLIR_RUNTIME_TRANSFORMS_COMPILER_H_
#define XLA_MLIR_RUNTIME_TRANSFORMS_COMPILER_H_

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace xla {
namespace runtime {

class DialectRegistry {
 public:
  DialectRegistry() = default;
  mlir::DialectRegistry* operator->() { return &registry_; }
  mlir::DialectRegistry& operator*() { return registry_; }

 private:
  mlir::DialectRegistry registry_;
};

class PassManager {
 public:
  explicit PassManager(mlir::PassManager* pm) : pm_(pm) {}
  mlir::PassManager* operator->() { return pm_; }
  mlir::PassManager& operator*() { return *pm_; }

 private:
  mlir::PassManager* pm_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_TRANSFORMS_COMPILER_H_
