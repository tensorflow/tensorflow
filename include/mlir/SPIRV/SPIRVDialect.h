//===- SPIRVDialect.h - MLIR SPIR-V dialect ---------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file declares the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SPIRV_SPIRVDIALECT_H_
#define MLIR_SPIRV_SPIRVDIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
class MLIRContext;

namespace spirv {

class SPIRVDialect : public Dialect {
public:
  explicit SPIRVDialect(MLIRContext *context);
};

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_SPIRV_SPIRVDIALECT_H_
