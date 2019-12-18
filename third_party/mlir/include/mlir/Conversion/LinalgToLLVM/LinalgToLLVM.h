//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
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
#ifndef MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_
#define MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class MLIRContext;

class LinalgTypeConverter : public LLVMTypeConverter {
public:
  using LLVMTypeConverter::LLVMTypeConverter;
  Type convertType(Type t) override;
};

/// Populate the given list with patterns that convert from Linalg to LLVM.
void populateLinalgToLLVMConversionPatterns(LinalgTypeConverter &converter,
                                            OwningRewritePatternList &patterns,
                                            MLIRContext *ctx);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_
