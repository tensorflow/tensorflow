//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion ---------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a pass that converts CFG function to LLVM IR.  No ML
// functions must be presented in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Functional.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Translation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace mlir;

namespace {
class ModuleLowerer {
public:
  explicit ModuleLowerer(llvm::LLVMContext &llvmContext)
      : llvmContext(llvmContext), builder(llvmContext) {}

  bool runOnModule(Module &m, llvm::Module &llvmModule);

private:
  bool convertBasicBlock(const BasicBlock &bb, bool ignoreArguments = false);
  bool convertCFGFunction(const CFGFunction &cfgFunc, llvm::Function &llvmFunc);
  bool convertFunctions(const Module &mlirModule, llvm::Module &llvmModule);
  bool convertInstruction(const Instruction &inst);

  void connectPHINodes(const CFGFunction &cfgFunc);

  /// Type conversion functions.  If any conversion fails, report errors to the
  /// context of the MLIR type and return nullptr.
  /// \{
  llvm::FunctionType *convertFunctionType(FunctionType type);
  llvm::IntegerType *convertIndexType(IndexType type);
  llvm::IntegerType *convertIntegerType(IntegerType type);
  llvm::Type *convertFloatType(FloatType type);
  llvm::Type *convertType(Type type);
  /// \}

  llvm::DenseMap<const Function *, llvm::Function *> functionMapping;
  llvm::DenseMap<const SSAValue *, llvm::Value *> valueMapping;
  llvm::DenseMap<const BasicBlock *, llvm::BasicBlock *> blockMapping;
  llvm::LLVMContext &llvmContext;
  llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> builder;
  llvm::IntegerType *indexType;
};

llvm::IntegerType *ModuleLowerer::convertIndexType(IndexType type) {
  return indexType;
}

llvm::IntegerType *ModuleLowerer::convertIntegerType(IntegerType type) {
  return builder.getIntNTy(type.getBitWidth());
}

llvm::Type *ModuleLowerer::convertFloatType(FloatType type) {
  MLIRContext *context = type.getContext();
  switch (type.getKind()) {
  case Type::Kind::F32:
    return builder.getFloatTy();
  case Type::Kind::F64:
    return builder.getDoubleTy();
  case Type::Kind::F16:
    return builder.getHalfTy();
  case Type::Kind::BF16:
    return context->emitError(UnknownLoc::get(context),
                              "Unsupported type: BF16"),
           nullptr;
  default:
    llvm_unreachable("non-float type in convertFloatType");
  }
}

llvm::FunctionType *ModuleLowerer::convertFunctionType(FunctionType type) {
  // TODO(zinenko): convert tuple to LLVM structure types
  assert(type.getNumResults() <= 1 && "NYI: tuple returns");
  auto resultType = type.getNumResults() == 0
                        ? llvm::Type::getVoidTy(llvmContext)
                        : convertType(type.getResult(0));
  if (!resultType)
    return nullptr;

  auto argTypes =
      functional::map([this](Type inputType) { return convertType(inputType); },
                      type.getInputs());
  if (std::any_of(argTypes.begin(), argTypes.end(),
                  [](const llvm::Type *t) { return t == nullptr; }))
    return nullptr;

  return llvm::FunctionType::get(resultType, argTypes, /*isVarArg=*/false);
}

llvm::Type *ModuleLowerer::convertType(Type type) {
  if (auto funcType = type.dyn_cast<FunctionType>())
    return convertFunctionType(funcType);
  if (auto intType = type.dyn_cast<IntegerType>())
    return convertIntegerType(intType);
  if (auto floatType = type.dyn_cast<FloatType>())
    return convertFloatType(floatType);
  if (auto indexType = type.dyn_cast<IndexType>())
    return convertIndexType(indexType);

  MLIRContext *context = type.getContext();
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "unsupported type: ";
  type.print(os);
  context->emitError(UnknownLoc::get(context), os.str());
  return nullptr;
}

static llvm::CmpInst::Predicate getLLVMCmpPredicate(CmpIPredicate p) {
  switch (p) {
  case CmpIPredicate::EQ:
    return llvm::CmpInst::Predicate::ICMP_EQ;
  case CmpIPredicate::NE:
    return llvm::CmpInst::Predicate::ICMP_NE;
  case CmpIPredicate::SLT:
    return llvm::CmpInst::Predicate::ICMP_SLT;
  case CmpIPredicate::SLE:
    return llvm::CmpInst::Predicate::ICMP_SLE;
  case CmpIPredicate::SGT:
    return llvm::CmpInst::Predicate::ICMP_SGT;
  case CmpIPredicate::SGE:
    return llvm::CmpInst::Predicate::ICMP_SGE;
  case CmpIPredicate::ULT:
    return llvm::CmpInst::Predicate::ICMP_ULT;
  case CmpIPredicate::ULE:
    return llvm::CmpInst::Predicate::ICMP_ULE;
  case CmpIPredicate::UGT:
    return llvm::CmpInst::Predicate::ICMP_UGT;
  case CmpIPredicate::UGE:
    return llvm::CmpInst::Predicate::ICMP_UGE;
  default:
    llvm_unreachable("incorrect comparison predicate");
  }
}

// Convert specific operation instruction types LLVM instructions.
// FIXME(zinenko): this should eventually become a separate MLIR pass that
// converts MLIR standard operations into LLVM IR dialect; the translation in
// that case would become a simple 1:1 instruction and value remapping.
bool ModuleLowerer::convertInstruction(const Instruction &inst) {
  if (auto op = inst.dyn_cast<AddIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateAdd(valueMapping[op->getOperand(0)],
                                 valueMapping[op->getOperand(1)]),
           false;
  if (auto op = inst.dyn_cast<MulIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateMul(valueMapping[op->getOperand(0)],
                                 valueMapping[op->getOperand(1)]),
           false;
  if (auto op = inst.dyn_cast<CmpIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateICmp(getLLVMCmpPredicate(op->getPredicate()),
                                  valueMapping[op->getOperand(0)],
                                  valueMapping[op->getOperand(1)]),
           false;
  if (auto constantOp = inst.dyn_cast<ConstantOp>()) {
    llvm::Type *type = convertType(constantOp->getType());
    if (!type)
      return true;
    assert(isa<llvm::IntegerType>(type) &&
           "only integer LLVM types are supported");
    auto attr = (constantOp->getValue()).cast<IntegerAttr>();
    // Create a new APInt even if we can extract one from the attribute, because
    // attributes are currently hardcoded to be 64-bit APInts and LLVM will
    // create an i64 constant from those.
    valueMapping[constantOp->getResult()] = llvm::Constant::getIntegerValue(
        type, llvm::APInt(type->getIntegerBitWidth(), attr.getInt()));
    return false;
  }
  if (auto callOp = inst.dyn_cast<CallOp>()) {
    auto operands = functional::map(
        [this](const SSAValue *value) { return valueMapping.lookup(value); },
        callOp->getOperands());
    auto numResults = callOp->getNumResults();
    // TODO(zinenko): support tuple returns
    assert(numResults <= 1 && "NYI: tuple returns");

    llvm::Value *result =
        builder.CreateCall(functionMapping[callOp->getCallee()], operands);
    if (numResults == 1)
      valueMapping[callOp->getResult(0)] = result;
    return false;
  }

  // Terminators.
  if (auto returnInst = inst.dyn_cast<ReturnOp>()) {
    unsigned numOperands = returnInst->getNumOperands();
    // TODO(zinenko): support tuple returns
    assert(numOperands <= 1u && "NYI: tuple returns");

    if (numOperands == 0)
      builder.CreateRetVoid();
    else
      builder.CreateRet(valueMapping[returnInst->getOperand(0)]);

    return false;
  }
  if (auto branchInst = inst.dyn_cast<BranchOp>()) {
    builder.CreateBr(blockMapping[branchInst->getDest()]);
    return false;
  }
  if (auto condBranchInst = inst.dyn_cast<CondBranchOp>()) {
    builder.CreateCondBr(valueMapping[condBranchInst->getCondition()],
                         blockMapping[condBranchInst->getTrueDest()],
                         blockMapping[condBranchInst->getFalseDest()]);
    return false;
  }
  inst.emitError("unsupported operation");
  return true;
}

bool ModuleLowerer::convertBasicBlock(const BasicBlock &bb,
                                      bool ignoreArguments) {
  builder.SetInsertPoint(blockMapping[&bb]);

  // Before traversing instructions, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first basic block have been already made available through the remapping of
  // LLVM function arguments.
  if (!ignoreArguments) {
    auto predecessors = bb.getPredecessors();
    unsigned numPredecessors =
        std::distance(predecessors.begin(), predecessors.end());
    for (const auto *arg : bb.getArguments()) {
      llvm::Type *type = convertType(arg->getType());
      if (!type)
        return true;
      llvm::PHINode *phi = builder.CreatePHI(type, numPredecessors);
      valueMapping[arg] = phi;
    }
  }

  // Traverse instructions.
  for (const auto &inst : bb) {
    if (convertInstruction(inst))
      return true;
  }

  return false;
}

// Get the SSA value passed to the current block from the terminator instruction
// of its predecessor.
static const SSAValue *getPHISourceValue(const BasicBlock *current,
                                         const BasicBlock *pred,
                                         unsigned numArguments,
                                         unsigned index) {
  const Instruction &terminator = *pred->getTerminator();
  if (terminator.isa<BranchOp>()) {
    return terminator.getOperand(index);
  }

  // For conditional branches, we need to check if the current block is reached
  // through the "true" or the "false" branch and take the relevant operands.
  auto condBranchOp = terminator.dyn_cast<CondBranchOp>();
  assert(condBranchOp &&
         "only branch instructions can be terminators of a basic block that "
         "has successors");

  condBranchOp->emitError("NYI: conditional branches with arguments");
  return nullptr;
}

void ModuleLowerer::connectPHINodes(const CFGFunction &cfgFunc) {
  // Skip the first block, it cannot be branched to and its arguments correspond
  // to the arguments of the LLVM function.
  for (auto it = std::next(cfgFunc.begin()), eit = cfgFunc.end(); it != eit;
       ++it) {
    const BasicBlock *bb = &*it;
    llvm::BasicBlock *llvmBB = blockMapping[bb];
    auto phis = llvmBB->phis();
    auto numArguments = bb->getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto &numberedPhiNode : llvm::enumerate(phis)) {
      auto &phiNode = numberedPhiNode.value();
      unsigned index = numberedPhiNode.index();
      for (const auto *pred : bb->getPredecessors()) {
        phiNode.addIncoming(
            valueMapping[getPHISourceValue(bb, pred, numArguments, index)],
            blockMapping[pred]);
      }
    }
  }
}

bool ModuleLowerer::convertCFGFunction(const CFGFunction &cfgFunc,
                                       llvm::Function &llvmFunc) {
  // Clear the block mapping.  Blocks belong to a function, no need to keep
  // blocks from the previous functions around.  Furthermore, we use this
  // mapping to connect PHI nodes inside the function later.
  blockMapping.clear();
  // First, create all blocks so we can jump to them.
  for (const auto &bb : cfgFunc) {
    auto *llvmBB = llvm::BasicBlock::Create(llvmContext);
    llvmBB->insertInto(&llvmFunc);
    blockMapping[&bb] = llvmBB;
  }

  // Then, convert blocks one by one.
  for (auto indexedBB : llvm::enumerate(cfgFunc)) {
    const auto &bb = indexedBB.value();
    if (convertBasicBlock(bb, /*ignoreArguments=*/indexedBB.index() == 0))
      return true;
  }

  // Finally, after all blocks have been traversed and values mapped, connect
  // the PHI nodes to the results of preceding blocks.
  connectPHINodes(cfgFunc);
  return false;
}

bool ModuleLowerer::convertFunctions(const Module &mlirModule,
                                     llvm::Module &llvmModule) {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles.  We don't expect MLFunctions here.
  for (const Function &function : mlirModule) {
    const Function *functionPtr = &function;
    if (!isa<ExtFunction>(functionPtr) && !isa<CFGFunction>(functionPtr))
      continue;
    llvm::Constant *llvmFuncCst = llvmModule.getOrInsertFunction(
        function.getName(), convertFunctionType(function.getType()));
    assert(isa<llvm::Function>(llvmFuncCst));
    functionMapping[functionPtr] = cast<llvm::Function>(llvmFuncCst);
  }

  // Convert CFG functions.
  for (const Function &function : mlirModule) {
    const Function *functionPtr = &function;
    auto cfgFunction = dyn_cast<CFGFunction>(functionPtr);
    if (!cfgFunction)
      continue;
    llvm::Function *llvmFunc = functionMapping[cfgFunction];

    // Add function arguments to the value remapping table.  In CFGFunction,
    // arguments of the first block are those of the function.
    assert(!cfgFunction->getBlocks().empty() &&
           "expected at least one basic block in a CFGFunction");
    const BasicBlock &firstBlock = *cfgFunction->begin();
    for (auto arg : llvm::enumerate(llvmFunc->args())) {
      valueMapping[firstBlock.getArgument(arg.index())] = &arg.value();
    }

    if (convertCFGFunction(*cfgFunction, *functionMapping[cfgFunction]))
      return true;
  }
  return false;
}

bool ModuleLowerer::runOnModule(Module &m, llvm::Module &llvmModule) {
  // Create index type once for the entire module, it needs module info that is
  // not available in the convert*Type calls.
  indexType =
      builder.getIntNTy(llvmModule.getDataLayout().getPointerSizeInBits());

  return convertFunctions(m, llvmModule);
}
} // namespace

// Entry point for the lowering procedure.
std::unique_ptr<llvm::Module>
mlir::convertModuleToLLVMIR(Module &module, llvm::LLVMContext &llvmContext) {
  auto llvmModule = llvm::make_unique<llvm::Module>("FIXME_name", llvmContext);
  if (ModuleLowerer(llvmContext).runOnModule(module, *llvmModule))
    return nullptr;
  return llvmModule;
}

// MLIR to LLVM IR translation registration.
static TranslateFromMLIRRegistration MLIRToLLVMIRTranslate(
    "mlir-to-llvmir", [](Module *module, llvm::StringRef outputFilename) {
      if (!module)
        return true;

      llvm::LLVMContext llvmContext;
      auto llvmModule = convertModuleToLLVMIR(*module, llvmContext);
      if (!llvmModule)
        return true;

      auto file = openOutputFile(outputFilename);
      if (!file)
        return true;

      llvmModule->print(file->os(), nullptr);
      file->keep();
      return false;
    });
