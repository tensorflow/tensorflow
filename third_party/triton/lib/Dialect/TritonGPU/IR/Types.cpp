#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::gpu;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"

static constexpr llvm::StringRef kMutableMemory = "mutable";

Type MemDescType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();

  SmallVector<int64_t> dimensions; // required
  if (failed(parser.parseDimensionList(dimensions, /*allowDynamic=*/false)))
    return Type();

  Type elementType; // required
  if (failed(parser.parseType(elementType)))
    return Type();

  Attribute encoding; // required
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(encoding)))
    return Type();

  Attribute memorySpace; // required
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(memorySpace)))
    return Type();

  bool mutableMemory = false;      // optional
  SmallVector<int64_t> allocShape; // optional
  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword(kMutableMemory))) {
      mutableMemory = true;
      if (succeeded(parser.parseOptionalComma())) {
        if (failed(parser.parseDimensionList(allocShape, /*allowDynamic=*/false,
                                             /*withTrailingX=*/false))) {
          return Type();
        }
      }
    } else if (failed(parser.parseDimensionList(allocShape,
                                                /*allowDynamic=*/false,
                                                /*withTrailingX=*/false))) {
      return Type();
    }
  }

  if (parser.parseGreater())
    return Type();

  if (allocShape.size() > 0)
    return MemDescType::get(parser.getContext(), dimensions, elementType,
                            encoding, memorySpace, mutableMemory, allocShape);

  return MemDescType::get(parser.getContext(), dimensions, elementType,
                          encoding, memorySpace, mutableMemory, dimensions);
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";
  auto shape = getShape();
  for (auto dim : shape)
    printer << dim << "x";
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  if (getMemorySpace())
    printer << ", " << getMemorySpace();
  if (getMutableMemory())
    printer << ", " << kMutableMemory;
  auto allocShape = getAllocShape();
  if (allocShape != shape) {
    printer << ", " << allocShape[0];
    for (auto dim : allocShape.drop_front(1)) {
      printer << "x" << dim;
    }
  }
  printer << ">";
}

LogicalResult MemDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  Attribute encoding, Attribute memorySpace,
                                  bool mutableMemory,
                                  ArrayRef<int64_t> allocShape) {
  if (allocShape.size() < shape.size())
    emitError() << "alloc shape must have at least as many dimensions as shape";
  return success();
}

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::gpu::TritonGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"
      >();
}
