/* Copyright 2025 The OpenXLA Authors.

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

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassOptions.h"
#include "nanobind/nanobind.h"
#include "shardy/dialect/sdy/transforms/propagation/auto_partitioner_registry.h"

namespace xla {
namespace spmd {

void RegisterAutoSharding() {
  mlir::sdy::AutoPartitionerRegistry::setCallback(
      /*callback=*/[](mlir::OpPassManager&) {},
      /*dialectsDependenciesCallback=*/[](mlir::DialectRegistry&) {});
}

NB_MODULE(auto_sharding_python_extension, m) {
  m.doc() = "AutoSharding Python extension";
  m.def("register", &RegisterAutoSharding);
  m.def("clear", &mlir::sdy::AutoPartitionerRegistry::clear);
  m.def("is_registered", &mlir::sdy::AutoPartitionerRegistry::isRegistered);
}

}  // namespace spmd
}  // namespace xla
