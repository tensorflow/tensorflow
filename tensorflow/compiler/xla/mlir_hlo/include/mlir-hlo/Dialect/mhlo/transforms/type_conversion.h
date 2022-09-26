/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_TYPE_CONVERSION_H
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_TYPE_CONVERSION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

// Type converter to use as part of lowerings from dialects that carry signs
// in their types to those that are signless.
class RemoveSignTypeConverter : public TypeConverter {
 public:
  RemoveSignTypeConverter();
};

// Type converter which adds additional materializations (beyond signless)
// that are needed as part of the HloToLinalg conversion patterns.
// This is the type converter used by the test pass and is the sanctioned
// way to use the underlying patterns.
class LinalgTypeConverter : public RemoveSignTypeConverter {
 public:
  LinalgTypeConverter();
};

}  // namespace mhlo

namespace stablehlo {

// Type converter that handles types which are common to HLO dialects.
// This consists of:
//   * Boolean types (i1).
//   * Signless integer types (i4/i8/i16/i32/i64).
//   * Unsigned integer types (ui4/ui8/ui16/ui32/ui64).
//   * Floating-point types (bf1/f16/f32/f64).
//   * Complex types (complex of f32/f64).
//   * Index types (index).
//   * Tensor types.
//   * Tuple types.
// Types which are specific to individual dialects like !stablehlo.token
// and !mhlo.token are handled in subclasses.
class HloTypeConverter : public TypeConverter {
 public:
  HloTypeConverter();
  virtual ~HloTypeConverter() = default;

  // Checks whether the given dialect is the source dialect of the type
  // conversion (e.g. MHLO for HloToStablehloTypeConverter).
  virtual bool isSourceDialect(Dialect& dialect) = 0;

  // Convert an encoding defined by the source dialect.
  virtual Attribute convertSourceDialectEncoding(Attribute attr) = 0;
};

// Type converter that changes all !mhlo.foo types to !stablehlo.foo types.
// Also changes MHLO-defined encodings to StableHLO equivalents.
class HloToStablehloTypeConverter : public HloTypeConverter {
 public:
  HloToStablehloTypeConverter();
  bool isSourceDialect(Dialect& dialect) override;
  Attribute convertSourceDialectEncoding(Attribute attr) override;
};

// Type converter that changes all !stablehlo.foo types to !mhlo.foo types.
// Also changes StableHLO-defined encodings to MHLO equivalents.
class StablehloToHloTypeConverter : public HloTypeConverter {
 public:
  StablehloToHloTypeConverter();
  bool isSourceDialect(Dialect& dialect) override;
  Attribute convertSourceDialectEncoding(Attribute attr) override;
};

// Complements StableHLO <=> MHLO conversion patterns with boilerplate that
// makes sure `func.func`, `func.call` and `func.return` ops which involve
// illegal types get converted to use legal types.
void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter);

}  // namespace stablehlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_TYPE_CONVERSION_H
