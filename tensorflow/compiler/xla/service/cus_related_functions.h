#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CUS_RELATED_FUNCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CUS_RELATED_FUNCTIONS_H_

// #include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
// #include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

llvm::Value* EmitCusToF32(llvm::Value* cus_value, llvm::IRBuilder<>* b);

StatusOr<llvm::Value*> EmitF32ToCus(llvm::Value* f32_value,
                                    llvm::IRBuilder<>* b);

llvm::Value* EmitCusNeg(llvm::Value* cus_value, llvm::IRBuilder<>* b);


#define CUS_BINARY_OP_HEADER(op)                               \
  llvm::Value* EmitCus##op(llvm::Value* lhs, llvm::Value* rhs, \
                           llvm::IRBuilder<>* b)

CUS_BINARY_OP_HEADER(Add);
CUS_BINARY_OP_HEADER(Sub);
CUS_BINARY_OP_HEADER(Mul);
CUS_BINARY_OP_HEADER(Div);
CUS_BINARY_OP_HEADER(Eq);
CUS_BINARY_OP_HEADER(Ne);
CUS_BINARY_OP_HEADER(Lt);
CUS_BINARY_OP_HEADER(Gt);
CUS_BINARY_OP_HEADER(Le);
CUS_BINARY_OP_HEADER(Ge);


}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CUS_RELATED_FUNCTIONS_H_
