#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp);
} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
