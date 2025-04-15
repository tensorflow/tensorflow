/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
#define XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_

#include <optional>

#include "llvm/IR/IRBuilder.h"

namespace xla {

// Mixin class that injects more ergonomic versions of llvm::IRBuilder methods
// into a class.  Intended to be used as a CRTP base class, like:
//
//  class MyIrEmitter : public IrBuilderMixin<MyIrEmitter> {
//    llvm::IRBuilder<>* builder() { return builder_; }
//
//    void EmitFoo(HloInstruction* foo) {
//      Add(Mul(...), FPToUI(...));
//    }
//  };

template <typename Derived>
class IrBuilderMixin {
 protected:
  template <class... Args>
  llvm::Value* Add(Args&&... args) {
    return mixin_builder()->CreateAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::LoadInst* AlignedLoad(Args&&... args) {
    return mixin_builder()->CreateAlignedLoad(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::StoreInst* AlignedStore(Args&&... args) {
    return mixin_builder()->CreateAlignedStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::AllocaInst* Alloca(Args&&... args) {
    return mixin_builder()->CreateAlloca(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* And(Args&&... args) {
    return mixin_builder()->CreateAnd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* AtomicCmpXchg(Args&&... args) {
    return mixin_builder()->CreateAtomicCmpXchg(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* AtomicRMW(Args&&... args) {
    return mixin_builder()->CreateAtomicRMW(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* BitCast(Args&&... args) {
    return mixin_builder()->CreateBitCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Br(Args&&... args) {
    return mixin_builder()->CreateBr(std::forward<Args>(args)...);
  }

  llvm::CallInst* Call(llvm::FunctionCallee func_callee,
                       llvm::ArrayRef<llvm::Value*> args = std::nullopt,
                       const llvm::Twine& name = "",
                       llvm::MDNode* fp_math_tag = nullptr) {
    return mixin_builder()->CreateCall(func_callee, args, name, fp_math_tag);
  }

  llvm::CallInst* Call(llvm::FunctionType* func_type, llvm::Value* callee,
                       llvm::ArrayRef<llvm::Value*> args = std::nullopt,
                       const llvm::Twine& name = "",
                       llvm::MDNode* fp_math_tag = nullptr) {
    return mixin_builder()->CreateCall(func_type, callee, args, name,
                                       fp_math_tag);
  }

  template <class... Args>
  llvm::BranchInst* CondBr(Args&&... args) {
    return mixin_builder()->CreateCondBr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ConstInBoundsGEP1_32(Args&&... args) {
    return mixin_builder()->CreateConstInBoundsGEP1_32(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ConstInBoundsGEP1_64(Args&&... args) {
    return mixin_builder()->CreateConstInBoundsGEP1_64(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FAdd(Args&&... args) {
    return mixin_builder()->CreateFAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FMul(Args&&... args) {
    return mixin_builder()->CreateFMul(std::forward<Args>(args)...);
  }

  llvm::Value* GEP(llvm::Type* type, llvm::Value* ptr,
                   llvm::ArrayRef<llvm::Value*> idx_list,
                   const llvm::Twine& name = "") {
    return mixin_builder()->CreateGEP(type, ptr, idx_list, name);
  }

  template <class... Args>
  llvm::Value* ICmpEQ(Args&&... args) {
    return mixin_builder()->CreateICmpEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpNE(Args&&... args) {
    return mixin_builder()->CreateICmpNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpULE(Args&&... args) {
    return mixin_builder()->CreateICmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpULT(Args&&... args) {
    return mixin_builder()->CreateICmpULT(std::forward<Args>(args)...);
  }

  llvm::Value* InBoundsGEP(llvm::Type* type, llvm::Value* ptr,
                           llvm::ArrayRef<llvm::Value*> idx_list,
                           const llvm::Twine& name = "") {
    return mixin_builder()->CreateInBoundsGEP(type, ptr, idx_list, name);
  }

  llvm::Value* ExtractValue(llvm::Value* agg, llvm::ArrayRef<unsigned> idxs,
                            const llvm::Twine& name = "") {
    return mixin_builder()->CreateExtractValue(agg, idxs, name);
  }

  llvm::Value* InsertValue(llvm::Value* agg, llvm::Value* val,
                           llvm::ArrayRef<unsigned> idxs,
                           const llvm::Twine& name = "") {
    return mixin_builder()->CreateInsertValue(agg, val, idxs, name);
  }

  template <class... Args>
  llvm::Value* IntToPtr(Args&&... args) {
    return mixin_builder()->CreateIntToPtr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::LoadInst* Load(Args&&... args) {
    return mixin_builder()->CreateLoad(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::CallInst* MemCpy(Args&&... args) {
    return mixin_builder()->CreateMemCpy(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Mul(Args&&... args) {
    return mixin_builder()->CreateMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* NSWAdd(Args&&... args) {
    return mixin_builder()->CreateNSWAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* NSWMul(Args&&... args) {
    return mixin_builder()->CreateNSWMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* NSWSub(Args&&... args) {
    return mixin_builder()->CreateNSWSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Or(Args&&... args) {
    return mixin_builder()->CreateOr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* PointerCast(Args&&... args) {
    return mixin_builder()->CreatePointerCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* PtrToInt(Args&&... args) {
    return mixin_builder()->CreatePtrToInt(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SDiv(Args&&... args) {
    return mixin_builder()->CreateSDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Select(Args&&... args) {
    return mixin_builder()->CreateSelect(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SRem(Args&&... args) {
    return mixin_builder()->CreateSRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::StoreInst* Store(Args&&... args) {
    return mixin_builder()->CreateStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* UDiv(Args&&... args) {
    return mixin_builder()->CreateUDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* URem(Args&&... args) {
    return mixin_builder()->CreateURem(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* VectorSplat(Args&&... args) {
    return mixin_builder()->CreateVectorSplat(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ZExtOrTrunc(Args&&... args) {
    return mixin_builder()->CreateZExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* AShr(Args&&... args) {
    return mixin_builder()->CreateAShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOEQ(Args&&... args) {
    return mixin_builder()->CreateFCmpOEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOGT(Args&&... args) {
    return mixin_builder()->CreateFCmpOGT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOGE(Args&&... args) {
    return mixin_builder()->CreateFCmpOGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOLT(Args&&... args) {
    return mixin_builder()->CreateFCmpOLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpULT(Args&&... args) {
    return mixin_builder()->CreateFCmpULT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpULE(Args&&... args) {
    return mixin_builder()->CreateFCmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpOLE(Args&&... args) {
    return mixin_builder()->CreateFCmpOLE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpONE(Args&&... args) {
    return mixin_builder()->CreateFCmpONE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpUNE(Args&&... args) {
    return mixin_builder()->CreateFCmpUNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpUNO(Args&&... args) {
    return mixin_builder()->CreateFCmpUNO(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FCmpUGE(Args&&... args) {
    return mixin_builder()->CreateFCmpUGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FDiv(Args&&... args) {
    return mixin_builder()->CreateFDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FNeg(Args&&... args) {
    return mixin_builder()->CreateFNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPCast(Args&&... args) {
    return mixin_builder()->CreateFPCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPToSI(Args&&... args) {
    return mixin_builder()->CreateFPToSI(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPToUI(Args&&... args) {
    return mixin_builder()->CreateFPToUI(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FPTrunc(Args&&... args) {
    return mixin_builder()->CreateFPTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FRem(Args&&... args) {
    return mixin_builder()->CreateFRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* FSub(Args&&... args) {
    return mixin_builder()->CreateFSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpSGE(Args&&... args) {
    return mixin_builder()->CreateICmpSGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* ICmpSLT(Args&&... args) {
    return mixin_builder()->CreateICmpSLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* IntCast(Args&&... args) {
    return mixin_builder()->CreateIntCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* LShr(Args&&... args) {
    return mixin_builder()->CreateLShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* MemSet(Args&&... args) {
    return mixin_builder()->CreateMemSet(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Neg(Args&&... args) {
    return mixin_builder()->CreateNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Not(Args&&... args) {
    return mixin_builder()->CreateNot(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::PHINode* PHI(Args&&... args) {
    return mixin_builder()->CreatePHI(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* RetVoid(Args&&... args) {
    return mixin_builder()->CreateRetVoid(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SExtOrTrunc(Args&&... args) {
    return mixin_builder()->CreateSExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Shl(Args&&... args) {
    return mixin_builder()->CreateShl(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* SIToFP(Args&&... args) {
    return mixin_builder()->CreateSIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Sub(Args&&... args) {
    return mixin_builder()->CreateSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Trunc(Args&&... args) {
    return mixin_builder()->CreateTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* UIToFP(Args&&... args) {
    return mixin_builder()->CreateUIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Unreachable(Args&&... args) {
    return mixin_builder()->CreateUnreachable(std::forward<Args>(args)...);
  }

  template <class... Args>
  llvm::Value* Xor(Args&&... args) {
    return mixin_builder()->CreateXor(std::forward<Args>(args)...);
  }

 private:
  llvm::IRBuilderBase* mixin_builder() {
    return static_cast<Derived*>(this)->builder();
  }
};

}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_IR_BUILDER_MIXIN_H_
