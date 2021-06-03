#include "tensorflow/compiler/xla/service/cus_related_functions.h"

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

namespace xla {

namespace {
#ifdef GOOGLE_CUDA
#define XLA_REGISTER_CUS_RELATED_CALL(function) \
  XLA_REGISTER_CUSTOM_CALL_TARGET(function, "CUDA")
#else
#define XLA_REGISTER_CUS_RELATED_CALL(function) \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(function);
#endif

// extern c wrapper of cus functions to avoid mangling
extern "C" {
float CastCusToF32(tensorflow::cus c) { return (float)(c); }
tensorflow::cus CastF32ToCus(const float f) { return tensorflow::cus(f); }
tensorflow::cus CusAdd(tensorflow::cus c1, tensorflow::cus c2) {
  return c1 + c2;
}
tensorflow::cus CusSub(tensorflow::cus c1, tensorflow::cus c2) {
  return c1 - c2;
}
tensorflow::cus CusMul(tensorflow::cus c1, tensorflow::cus c2) {
  return c1 * c2;
}
tensorflow::cus CusDiv(tensorflow::cus c1, tensorflow::cus c2) {
  return c1 / c2;
}

tensorflow::cus CusNeg(tensorflow::cus c) { return -c; }

uint8_t CusEq(tensorflow::cus c1, tensorflow::cus c2) { return c1 == c2; }
uint8_t CusNe(tensorflow::cus c1, tensorflow::cus c2) { return c1 != c2; }
uint8_t CusLt(tensorflow::cus c1, tensorflow::cus c2) { return c1 < c2; }
uint8_t CusGt(tensorflow::cus c1, tensorflow::cus c2) { return c1 > c2; }
uint8_t CusLe(tensorflow::cus c1, tensorflow::cus c2) { return c1 <= c2; }
uint8_t CusGe(tensorflow::cus c1, tensorflow::cus c2) { return c1 >= c2; }
}

bool registerCusFunctions() {
  XLA_REGISTER_CUS_RELATED_CALL(CastCusToF32);
  XLA_REGISTER_CUS_RELATED_CALL(CastF32ToCus);
  XLA_REGISTER_CUS_RELATED_CALL(CusAdd);
  XLA_REGISTER_CUS_RELATED_CALL(CusSub);
  XLA_REGISTER_CUS_RELATED_CALL(CusMul);
  XLA_REGISTER_CUS_RELATED_CALL(CusDiv);
  XLA_REGISTER_CUS_RELATED_CALL(CusNeg);
  XLA_REGISTER_CUS_RELATED_CALL(CusEq);
  XLA_REGISTER_CUS_RELATED_CALL(CusNe);
  XLA_REGISTER_CUS_RELATED_CALL(CusLt);
  XLA_REGISTER_CUS_RELATED_CALL(CusGt);
  XLA_REGISTER_CUS_RELATED_CALL(CusLe);
  XLA_REGISTER_CUS_RELATED_CALL(CusGe);
  return true;
}

bool unused = registerCusFunctions();
}  // namespace

llvm::Value* EmitCusToF32(llvm::Value* cus_value, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::StructType* cus = llvm_ir::getCusTy(module->getContext());
  llvm::Value* func =
      module->getOrInsertFunction("CastCusToF32", b->getFloatTy(), cus)
          .getCallee();
  return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {cus_value});
}

StatusOr<llvm::Value*> EmitF32ToCus(llvm::Value* f32_value,
                                    llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::StructType* cus = llvm_ir::getCusTy(module->getContext());
  llvm::Value* func =
      module->getOrInsertFunction("CastF32ToCus", cus, b->getFloatTy())
          .getCallee();
  return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {f32_value});
}

llvm::Value* EmitCusNeg(llvm::Value* cus_value, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::StructType* cus = llvm_ir::getCusTy(module->getContext());
  llvm::Value* func =
      module->getOrInsertFunction("CusNeg", cus, cus)
          .getCallee();
  return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {cus_value});
}

#define CUS_BINARY_OP(op)                                                   \
  llvm::Value* EmitCus##op(llvm::Value* lhs, llvm::Value* rhs,              \
                           llvm::IRBuilder<>* b) {                          \
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();   \
    llvm::StructType* cus = llvm_ir::getCusTy(module->getContext());        \
    llvm::Value* func =                                                     \
        module->getOrInsertFunction("Cus" #op, cus, cus, cus).getCallee();  \
    return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {lhs, rhs}); \
  }

CUS_BINARY_OP(Add);
CUS_BINARY_OP(Sub);
CUS_BINARY_OP(Mul);
CUS_BINARY_OP(Div);

#define CUS_COMPARE(op)                                              \
  llvm::Value* EmitCus##op(llvm::Value* lhs, llvm::Value* rhs,              \
                           llvm::IRBuilder<>* b) {                          \
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();   \
    llvm::StructType* cus = llvm_ir::getCusTy(module->getContext());        \
    llvm::Value* func =                                                     \
        module                                                              \
            ->getOrInsertFunction(                                          \
                "Cus" #op, b->getIntNTy(8), cus, cus) \
            .getCallee();                                                   \
    return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {lhs, rhs}); \
  }

CUS_COMPARE(Eq);
CUS_COMPARE(Ne);
CUS_COMPARE(Lt);
CUS_COMPARE(Gt);
CUS_COMPARE(Le);
CUS_COMPARE(Ge);

}  // namespace xla