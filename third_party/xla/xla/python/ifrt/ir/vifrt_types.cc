/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/vifrt_types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: export
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/AssemblyFormat.h"  // IWYU pragma: export

namespace xla {
namespace ifrt {

void VifrtTypeConverterBuiltin::addBuiltinToVifrtConversions() {
  // We currently rely on the builtin types being stable, and thus we do not
  // convert builtin types to VIFRT types.
}

void VifrtTypeConverterBuiltin::addVifrtToBuiltinConversions() {
  // We currently rely on the builtin types are stable, and thus we do not
  // convert from VIFRT types to builtin types.
}

namespace {

// Verifies if a given type or attribute is from VIFRT dialect.
// Must be defined before importing the generated type interfaces and defs.
template <typename TypeOrAttr>
bool isFromVifrt(TypeOrAttr t) {
  return t.getDialect().getNamespace() == "vifrt";
}

}  // namespace

}  // namespace ifrt
}  // namespace xla

// Include order matters.
#include "xla/python/ifrt/ir/vifrt_type_interfaces.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_type_defs.cc.inc"

namespace xla {
namespace ifrt {

mlir::LogicalResult printVifrtType(mlir::Type type, mlir::AsmPrinter& printer) {
  return generatedTypePrinter(type, printer);
}

mlir::OptionalParseResult parseVifrtType(mlir::AsmParser& parser,
                                         llvm::StringRef* mnemonic,
                                         mlir::Type& type) {
  return generatedTypeParser(parser, mnemonic, type);
}

namespace {
template <typename... Types>
void registerVifrtTypes(mlir::MLIRContext* context) {
  (mlir::detail::TypeUniquer::registerType<Types>(context), ...);
}
}  // namespace

void registerVifrtTypes(mlir::MLIRContext* context) {
  registerVifrtTypes<
#define GET_TYPEDEF_LIST
#include "xla/python/ifrt/ir/vifrt_type_defs.cc.inc"
      >(context);
}

}  // namespace ifrt
}  // namespace xla
