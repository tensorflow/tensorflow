// Copyright 2021 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements MLIR operations for the xlir_ops library.

#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"

#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// XlirDialect Dialect
//===----------------------------------------------------------------------===//

XlirDialect::XlirDialect(mlir::MLIRContext *context)
    : Dialect(/*name*/ "xlir", context, TypeID::get<XlirDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/xla/service/gpu/xlir_opdefs.cpp.inc"
      >();
}

// The following Enum parsers/printers are duplicated from
// tf_runtime/backends/gpu/lib/kernels/gpu_ops.cc.
template <typename T>
static ParseResult parseEnum(OpAsmParser &parser,
                             tfrt::gpu::EnumAttr<T> &attribute,
                             llvm::Expected<T> (*parse_func)(StringRef)) {
  StringRef name;
  if (failed(parser.parseKeyword(&name))) return failure();
  auto value = parse_func(name);
  if (!value) {
    return parser.emitError(parser.getCurrentLocation(),
                            toString(value.takeError()));
  }
  attribute =
      tfrt::gpu::EnumAttr<T>::get(parser.getBuilder().getContext(), *value);
  return success();
}

template <typename T>
static ParseResult parseEnum(OpAsmParser &parser,
                             tfrt::gpu::EnumAttr<T> &attribute) {
  auto parse_func = [](StringRef name) {
    return tfrt::gpu::wrapper::Parse<T>(name);
  };
  return parseEnum(parser, attribute, +parse_func);
}

template <typename T>
static void printEnum(OpAsmPrinter &printer, Operation *,
                      tfrt::gpu::EnumAttr<T> attribute) {
  tfrt::gpu::wrapper::operator<<(printer.getStream(), attribute.getValue());
}

}  // namespace gpu
}  // namespace xla

// TableGen'd definitions
#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/service/gpu/xlir_opdefs.cpp.inc"
