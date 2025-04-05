#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::AMD {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

// Retursn true if given architecture support V_DOT instruction.
bool supportsVDot(llvm::StringRef arch);

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/8c75290/llvm/lib/Target/AMDGPU/SIDefines.h#L939
enum class DppCtrl : uint32_t {
  QUAD_PERM_FIRST = 0,
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  BCAST15 = 0x142,
  BCAST31 = 0x143
};

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_
