/* Copyright 2020 The OpenXLA Authors.

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

#ifndef MLIR_HLO_DIALECT_MHLO_IR_REGISTER_H_
#define MLIR_HLO_DIALECT_MHLO_IR_REGISTER_H_

namespace mlir {
class DialectRegistry;
namespace mhlo {

// Add chlo and mhlo dialects to the provided registry.
void registerAllMhloDialects(DialectRegistry &registry);
}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_IR_REGISTER_H_
