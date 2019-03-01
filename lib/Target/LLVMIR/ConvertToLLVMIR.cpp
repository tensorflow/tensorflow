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
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Translation.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace mlir;

namespace {
// Implementation class for module translation.  Holds a reference to the module
// being translated, and the mappings between the original and the translated
// functions, basic blocks and values.  It is practically easier to hold these
// mappings in one class since the conversion of control flow instructions
// needs to look up block and function mappins.
class ModuleTranslation {
public:
  // Translate the given MLIR module expressed in MLIR LLVM IR dialect into an
  // LLVM IR module.  The MLIR LLVM IR dialect holds a pointer to an
  // LLVMContext, the LLVM IR module will be created in that context.
  static std::unique_ptr<llvm::Module> translateModule(const Module &m);

private:
  explicit ModuleTranslation(const Module &module) : mlirModule(module) {}

  bool convertFunctions();
  bool convertOneFunction(const Function &func);
  void connectPHINodes(const Function &func);
  bool convertBlock(const Block &bb, bool ignoreArguments);
  bool convertInstruction(const Instruction &inst, llvm::IRBuilder<> &builder);

  llvm::Constant *getLLVMConstant(llvm::Type *llvmType, Attribute attr,
                                  Location loc);

  // Original and translated module.
  const Module &mlirModule;
  std::unique_ptr<llvm::Module> llvmModule;

  // Mappings between original and translated values, used for lookups.
  llvm::DenseMap<const Function *, llvm::Function *> functionMapping;
  llvm::DenseMap<const Value *, llvm::Value *> valueMapping;
  llvm::DenseMap<const Block *, llvm::BasicBlock *> blockMapping;
};
} // end anonymous namespace

// Convert an MLIR function type to LLVM IR.  Arguments of the function must of
// MLIR LLVM IR dialect types.  Use `loc` as a location when reporting errors.
// Return nullptr on errors.
static llvm::FunctionType *convertFunctionType(llvm::LLVMContext &llvmContext,
                                               FunctionType type,
                                               Location loc) {
  assert(type && "expected non-null type");

  auto context = type.getContext();
  if (type.getNumResults() > 1)
    return context->emitError(loc,
                              "LLVM functions can only have 0 or 1 result"),
           nullptr;

  SmallVector<llvm::Type *, 8> argTypes;
  argTypes.reserve(type.getNumInputs());
  for (auto t : type.getInputs()) {
    auto wrappedLLVMType = t.dyn_cast<LLVM::LLVMType>();
    if (!wrappedLLVMType)
      return context->emitError(loc, "non-LLVM function argument type"),
             nullptr;
    argTypes.push_back(wrappedLLVMType.getUnderlyingType());
  }

  if (type.getNumResults() == 0)
    return llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), argTypes,
                                   /*isVarArg=*/false);

  auto wrappedResultType = type.getResult(0).dyn_cast<LLVM::LLVMType>();
  if (!wrappedResultType)
    return context->emitError(loc, "non-LLVM function result"), nullptr;

  return llvm::FunctionType::get(wrappedResultType.getUnderlyingType(),
                                 argTypes, /*isVarArg=*/false);
}

// Create an LLVM IR constant of `llvmType` from the MLIR attribute `attr`.
// This currently supports integer, floating point, splat and dense element
// attributes and combinations thereof.  In case of error, report it to `loc`
// and return nullptr.
llvm::Constant *ModuleTranslation::getLLVMConstant(llvm::Type *llvmType,
                                                   Attribute attr,
                                                   Location loc) {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return llvm::ConstantInt::get(llvmType, intAttr.getValue());
  if (auto floatAttr = attr.dyn_cast<FloatAttr>())
    return llvm::ConstantFP::get(llvmType, floatAttr.getValue());
  if (auto funcAttr = attr.dyn_cast<FunctionAttr>())
    return functionMapping.lookup(funcAttr.getValue());
  if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
    auto *vectorType = cast<llvm::VectorType>(llvmType);
    auto *child = getLLVMConstant(vectorType->getElementType(),
                                  splatAttr.getValue(), loc);
    return llvm::ConstantVector::getSplat(vectorType->getNumElements(), child);
  }
  if (auto denseAttr = attr.dyn_cast<DenseElementsAttr>()) {
    auto *vectorType = cast<llvm::VectorType>(llvmType);
    SmallVector<llvm::Constant *, 8> constants;
    uint64_t numElements = vectorType->getNumElements();
    constants.reserve(numElements);
    SmallVector<Attribute, 8> nested;
    denseAttr.getValues(nested);
    for (auto n : nested) {
      constants.push_back(
          getLLVMConstant(vectorType->getElementType(), n, loc));
      if (!constants.back())
        return nullptr;
    }
    return llvm::ConstantVector::get(constants);
  }
  mlirModule.getContext()->emitError(loc, "unsupported constant value");
  return nullptr;
}

// Convert MLIR integer comparison predicate to LLVM IR comparison predicate.
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

// Given a single MLIR instruction, create the corresponding LLVM IR instruction
// using the `builder`.  LLVM IR Builder does not have a generic interface so
// this has to be a long chain of `if`s calling different functions with a
// different number of arguments.
// TODO(zinenko): the conversion is largely mechanical and should be tablegen'ed
bool ModuleTranslation::convertInstruction(const Instruction &inst,
                                           llvm::IRBuilder<> &builder) {
#define CONV_BINARY_OP(CLASS, FUNC)                                            \
  if (auto op = inst.dyn_cast<CLASS>()) {                                      \
    valueMapping[op->getResult()] = builder.FUNC(                              \
        valueMapping.lookup(op->lhs()), valueMapping.lookup(op->rhs()));       \
    return false;                                                              \
  }

  CONV_BINARY_OP(LLVM::AddOp, CreateAdd);
  CONV_BINARY_OP(LLVM::SubOp, CreateSub);
  CONV_BINARY_OP(LLVM::MulOp, CreateMul);
  CONV_BINARY_OP(LLVM::SDivOp, CreateSDiv);
  CONV_BINARY_OP(LLVM::UDivOp, CreateUDiv);
  CONV_BINARY_OP(LLVM::SRemOp, CreateSRem);
  CONV_BINARY_OP(LLVM::URemOp, CreateURem);
  CONV_BINARY_OP(LLVM::FAddOp, CreateFAdd);
  CONV_BINARY_OP(LLVM::FSubOp, CreateFSub);
  CONV_BINARY_OP(LLVM::FMulOp, CreateFMul);
  CONV_BINARY_OP(LLVM::FDivOp, CreateFDiv);
  CONV_BINARY_OP(LLVM::FRemOp, CreateFRem);

#undef CONV_BINARY_OP

  if (auto op = inst.dyn_cast<LLVM::ICmpOp>()) {
    auto attr = op->getAttrOfType<IntegerAttr>("predicate");
    auto predicate = static_cast<CmpIPredicate>(attr.getValue().getSExtValue());

    valueMapping[op->getResult()] = builder.CreateICmp(
        getLLVMCmpPredicate(predicate), valueMapping.lookup(op->lhs()),
        valueMapping.lookup(op->rhs()));
    return false;
  }

  // Pseudo-ops.  These do not exist as LLVM operations but produce (constant)
  // values.
  if (auto op = inst.dyn_cast<LLVM::UndefOp>()) {
    auto wrappedType = op->getResult()->getType().dyn_cast<LLVM::LLVMType>();
    valueMapping[op->getResult()] =
        llvm::UndefValue::get(wrappedType.getUnderlyingType());
    return false;
  }

  if (auto op = inst.dyn_cast<LLVM::ConstantOp>()) {
    Attribute attr = op->getAttr("value");
    auto type = op->getResult()->getType().cast<LLVM::LLVMType>();
    valueMapping[op->getResult()] =
        getLLVMConstant(type.getUnderlyingType(), attr, inst.getLoc());
    return false;
  }

  // A helper to look up remapped operands in the value remapping table.
  auto lookupValues =
      [this](const llvm::iterator_range<Instruction::const_operand_iterator>
                 &values) {
        SmallVector<llvm::Value *, 8> remapped;
        remapped.reserve(llvm::size(values));
        for (const Value *v : values) {
          remapped.push_back(valueMapping.lookup(v));
        }
        return remapped;
      };

  // Emit function calls.  If the "callee" attribute is present, this is a
  // direct function call and we also need to look up the remapped function
  // itself.  Otherwise, this is an indirect call and the callee is the first
  // operand, look it up as a normal value.  Return the llvm::Value representing
  // the function result, which may be of llvm::VoidTy type.
  auto convertCall = [this, lookupValues,
                      &builder](const Instruction &inst) -> llvm::Value * {
    auto operands = lookupValues(inst.getOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    if (auto attr = inst.getAttrOfType<FunctionAttr>("callee")) {
      return builder.CreateCall(functionMapping.lookup(attr.getValue()),
                                operandsRef);
    } else {
      return builder.CreateCall(operandsRef.front(), operandsRef.drop_front());
    }
  };

  // Emit calls.  If the called function has a result, remap the corresponding
  // value.  Note that LLVM IR dialect CallOp has either 0 or 1 result.
  if (auto op = inst.dyn_cast<LLVM::CallOp>()) {
    llvm::Value *result = convertCall(inst);
    if (inst.getNumResults() != 0) {
      valueMapping[inst.getResult(0)] = result;
      return false;
    }
    // Check that LLVM call returns void for 0-result functions.
    return !result->getType()->isVoidTy();
  }

  // Emit branches.  We need to look up the remapped blocks and ignore the block
  // arguments that were transformed into PHI nodes.
  if (auto op = inst.dyn_cast<LLVM::BrOp>()) {
    builder.CreateBr(blockMapping[op->getSuccessor(0)]);
    return false;
  }
  if (auto op = inst.dyn_cast<LLVM::CondBrOp>()) {
    builder.CreateCondBr(valueMapping.lookup(op->getOperand(0)),
                         blockMapping[op->getSuccessor(0)],
                         blockMapping[op->getSuccessor(1)]);
    return false;
  }

  if (auto op = inst.dyn_cast<LLVM::ReturnOp>()) {
    if (op->getNumOperands() == 0)
      builder.CreateRetVoid();
    else
      builder.CreateRet(valueMapping.lookup(op->getOperand(0)));
    return false;
  }

  auto extractPosition = [](ArrayAttr attr) {
    SmallVector<unsigned, 4> position;
    position.reserve(attr.size());
    for (Attribute v : attr)
      position.push_back(v.cast<IntegerAttr>().getValue().getZExtValue());
    return position;
  };

  if (auto op = inst.dyn_cast<LLVM::ExtractValueOp>()) {
    auto attr = op->getAttrOfType<ArrayAttr>("position");
    valueMapping[op->getResult()] = builder.CreateExtractValue(
        valueMapping.lookup(op->getOperand()), extractPosition(attr));
    return false;
  }
  if (auto op = inst.dyn_cast<LLVM::InsertValueOp>()) {
    auto attr = op->getAttrOfType<ArrayAttr>("position");
    valueMapping[op->getResult()] = builder.CreateInsertValue(
        valueMapping.lookup(op->getOperand(0)),
        valueMapping.lookup(op->getOperand(1)), extractPosition(attr));
    return false;
  }
  if (auto op = inst.dyn_cast<LLVM::BitcastOp>()) {
    valueMapping[op->getResult()] = builder.CreateBitCast(
        valueMapping.lookup(op->getOperand()),
        op->getType().cast<LLVM::LLVMType>().getUnderlyingType());
    return false;
  }

  if (auto op = inst.dyn_cast<LLVM::GEPOp>()) {
    auto mappedOperands = lookupValues(op->getOperands());
    valueMapping[op->getResult()] =
        builder.CreateGEP(mappedOperands.front(),
                          llvm::makeArrayRef(mappedOperands).drop_front());
    return false;
  }
  if (auto op = inst.dyn_cast<LLVM::LoadOp>()) {
    valueMapping[op->getResult()] =
        builder.CreateLoad(valueMapping.lookup(op->getOperand()));
    return false;
  }
  if (auto op = inst.dyn_cast<LLVM::StoreOp>()) {
    builder.CreateStore(valueMapping.lookup(op->getOperand(0)),
                        valueMapping.lookup(op->getOperand(1)));
    return false;
  }
  if (auto op = inst.dyn_cast<LLVM::SelectOp>()) {
    valueMapping[op->getResult()] =
        builder.CreateSelect(valueMapping.lookup(op->getOperand(0)),
                             valueMapping.lookup(op->getOperand(1)),
                             valueMapping.lookup(op->getOperand(2)));
    return false;
  }

  inst.emitError("unsupported or non-LLVM operation: " +
                 inst.getName().getStringRef());
  return true;
}

// Convert block to LLVM IR.  Unless `ignoreArguments` is set, emit PHI nodes
// to define values corresponding to the MLIR block arguments.  These nodes
// are not connected to the source basic blocks, which may not exist yet.
bool ModuleTranslation::convertBlock(const Block &bb, bool ignoreArguments) {
  llvm::IRBuilder<> builder(blockMapping[&bb]);

  // Before traversing instructions, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first block have been already made available through the remapping of
  // LLVM function arguments.
  if (!ignoreArguments) {
    auto predecessors = bb.getPredecessors();
    unsigned numPredecessors =
        std::distance(predecessors.begin(), predecessors.end());
    for (const auto *arg : bb.getArguments()) {
      auto wrappedType = arg->getType().dyn_cast<LLVM::LLVMType>();
      if (!wrappedType) {
        arg->getType().getContext()->emitError(
            bb.front().getLoc(), "block argument does not have an LLVM type");
        return true;
      }
      llvm::Type *type = wrappedType.getUnderlyingType();
      llvm::PHINode *phi = builder.CreatePHI(type, numPredecessors);
      valueMapping[arg] = phi;
    }
  }

  // Traverse instructions.
  for (const auto &inst : bb) {
    if (convertInstruction(inst, builder))
      return true;
  }

  return false;
}

// Get the SSA value passed to the current block from the terminator instruction
// of its predecessor.
static const Value *getPHISourceValue(const Block *current, const Block *pred,
                                      unsigned numArguments, unsigned index) {
  auto &terminator = *pred->getTerminator();
  if (terminator.isa<LLVM::BrOp>()) {
    return terminator.getOperand(index);
  }

  // For conditional branches, we need to check if the current block is reached
  // through the "true" or the "false" branch and take the relevant operands.
  auto condBranchOp = terminator.dyn_cast<LLVM::CondBrOp>();
  assert(condBranchOp &&
         "only branch instructions can be terminators of a block that "
         "has successors");
  assert((condBranchOp->getSuccessor(0) != condBranchOp->getSuccessor(1)) &&
         "successors with arguments in LLVM conditional branches must be "
         "different blocks");

  return condBranchOp->getSuccessor(0) == current
             ? terminator.getSuccessorOperand(0, index)
             : terminator.getSuccessorOperand(1, index);
}

void ModuleTranslation::connectPHINodes(const Function &func) {
  // Skip the first block, it cannot be branched to and its arguments correspond
  // to the arguments of the LLVM function.
  for (auto it = std::next(func.begin()), eit = func.end(); it != eit; ++it) {
    const Block *bb = &*it;
    llvm::BasicBlock *llvmBB = blockMapping.lookup(bb);
    auto phis = llvmBB->phis();
    auto numArguments = bb->getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto &numberedPhiNode : llvm::enumerate(phis)) {
      auto &phiNode = numberedPhiNode.value();
      unsigned index = numberedPhiNode.index();
      for (const auto *pred : bb->getPredecessors()) {
        phiNode.addIncoming(valueMapping.lookup(getPHISourceValue(
                                bb, pred, numArguments, index)),
                            blockMapping.lookup(pred));
      }
    }
  }
}

// TODO(mlir-team): implement an iterative version
static void topologicalSortImpl(llvm::SetVector<const Block *> &blocks,
                                const Block *b) {
  blocks.insert(b);
  for (const Block *bb : b->getSuccessors()) {
    if (blocks.count(bb) == 0)
      topologicalSortImpl(blocks, bb);
  }
}

// Sort function blocks topologically.
static llvm::SetVector<const Block *> topologicalSort(const Function &f) {
  // For each blocks that has not been visited yet (i.e. that has no
  // predecessors), add it to the list and traverse its successors in DFS
  // preorder.
  llvm::SetVector<const Block *> blocks;
  for (const Block &b : f.getBlocks()) {
    if (blocks.count(&b) == 0)
      topologicalSortImpl(blocks, &b);
  }
  assert(blocks.size() == f.getBlocks().size() && "some blocks are not sorted");

  return blocks;
}

bool ModuleTranslation::convertOneFunction(const Function &func) {
  // Clear the block and value mappings, they are only relevant within one
  // function.
  blockMapping.clear();
  valueMapping.clear();
  llvm::Function *llvmFunc = functionMapping.lookup(&func);
  // Add function arguments to the value remapping table.
  for (const auto &kvp : llvm::zip(func.getArguments(), llvmFunc->args())) {
    valueMapping[std::get<0>(kvp)] = &std::get<1>(kvp);
  }

  // First, create all blocks so we can jump to them.
  llvm::LLVMContext &llvmContext = llvmFunc->getContext();
  for (const auto &bb : func) {
    auto *llvmBB = llvm::BasicBlock::Create(llvmContext);
    llvmBB->insertInto(llvmFunc);
    blockMapping[&bb] = llvmBB;
  }

  // Then, convert blocks one by one in topological order to ensure defs are
  // converted before uses.
  auto blocks = topologicalSort(func);
  for (auto indexedBB : llvm::enumerate(blocks)) {
    const auto *bb = indexedBB.value();
    if (convertBlock(*bb, /*ignoreArguments=*/indexedBB.index() == 0))
      return true;
  }

  // Finally, after all blocks have been traversed and values mapped, connect
  // the PHI nodes to the results of preceding blocks.
  connectPHINodes(func);
  return false;
}

bool ModuleTranslation::convertFunctions() {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles.
  for (const Function &function : mlirModule) {
    const Function *functionPtr = &function;
    llvm::FunctionType *functionType = convertFunctionType(
        llvmModule->getContext(), function.getType(), function.getLoc());
    if (!functionType)
      return true;
    llvm::FunctionCallee llvmFuncCst =
        llvmModule->getOrInsertFunction(function.getName(), functionType);
    assert(isa<llvm::Function>(llvmFuncCst.getCallee()));
    functionMapping[functionPtr] =
        cast<llvm::Function>(llvmFuncCst.getCallee());
  }

  // Convert functions.
  for (const Function &function : mlirModule) {
    // Ignore external functions.
    if (function.isExternal())
      continue;

    if (convertOneFunction(function))
      return true;
  }

  return false;
}

std::unique_ptr<llvm::Module>
ModuleTranslation::translateModule(const Module &m) {

  Dialect *dialect = m.getContext()->getRegisteredDialect("llvm");
  assert(dialect && "LLVM dialect must be registered");
  auto *llvmDialect = static_cast<LLVM::LLVMDialect *>(dialect);

  auto llvmModule = llvm::CloneModule(llvmDialect->getLLVMModule());
  if (!llvmModule)
    return nullptr;

  llvm::LLVMContext &llvmContext = llvmModule->getContext();
  llvm::IRBuilder<> builder(llvmContext);

  // Inject declarations for `malloc` and `free` functions that can be used in
  // memref allocation/deallocation coming from standard ops lowering.
  llvmModule->getOrInsertFunction("malloc", builder.getInt8PtrTy(),
                                  builder.getInt64Ty());
  llvmModule->getOrInsertFunction("free", builder.getVoidTy(),
                                  builder.getInt8PtrTy());

  ModuleTranslation translator(m);
  translator.llvmModule = std::move(llvmModule);
  if (translator.convertFunctions())
    return nullptr;

  return std::move(translator.llvmModule);
}

std::unique_ptr<llvm::Module> translateModuleToLLVMIR(const Module &m) {
  return ModuleTranslation::translateModule(m);
}

static TranslateFromMLIRRegistration registration(
    "mlir-to-llvmir", [](Module *module, llvm::StringRef outputFilename) {
      if (!module)
        return true;

      auto llvmModule = ModuleTranslation::translateModule(*module);
      if (!llvmModule)
        return true;

      auto file = openOutputFile(outputFilename);
      if (!file)
        return true;

      llvmModule->print(file->os(), nullptr);
      file->keep();
      return false;
    });
