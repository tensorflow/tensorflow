#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
using llvm::StringRef;

inline std::string strJoin(llvm::ArrayRef<std::string> strs,
                           llvm::StringRef delimiter) {
  return llvm::join(strs.begin(), strs.end(), delimiter);
}

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
