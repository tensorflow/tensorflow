//===- ConvertFromLLVMIR.cpp - MLIR to LLVM IR conversion -----------------===//
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
// This file implements a translation between LLVM IR and the MLIR LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Translation.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

// Utility to print an LLVM value as a string for passing to emitError().
// FIXME: Diagnostic should be able to natively handle types that have
// operator << (raw_ostream&) defined.
static std::string diag(llvm::Value &v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << v;
  return os.str();
}

// Handles importing globals and functions from an LLVM module.
namespace {
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module)
      : b(context), context(context), module(module),
        unknownLoc(FileLineColLoc::get("imported-bitcode", 0, 0, context)) {
    b.setInsertionPointToStart(module.getBody());
    dialect = context->getRegisteredDialect<LLVMDialect>();
  }

  /// Imports `f` into the current module.
  LogicalResult processFunction(llvm::Function *f);

  /// Imports GV as a GlobalOp, creating it if it doesn't exist.
  GlobalOp processGlobal(llvm::GlobalVariable *GV);

private:
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Imports `inst` and populates instMap[inst] with the imported Value.
  LogicalResult processInstruction(llvm::Instruction *inst);
  /// Creates an LLVMType for `type`.
  LLVMType processType(llvm::Type *type);
  /// `value` is an SSA-use. Return the remapped version of `value` or a
  /// placeholder that will be remapped later if this is an instruction that
  /// has not yet been visited.
  Value *processValue(llvm::Value *value);
  /// Create the most accurate Location possible using a llvm::DebugLoc and
  /// possibly an llvm::Instruction to narrow the Location if debug information
  /// is unavailable.
  Location processDebugLoc(const llvm::DebugLoc &loc,
                           llvm::Instruction *inst = nullptr);
  /// `br` branches to `target`. Return the block arguments to attach to the
  /// generated branch op. These should be in the same order as the PHIs in
  /// `target`.
  SmallVector<Value *, 4> processBranchArgs(llvm::BranchInst *br,
                                            llvm::BasicBlock *target);
  /// Return `value` as an attribute to attach to a GlobalOp.
  Attribute getConstantAsAttr(llvm::Constant *value);
  /// Return `c` as an MLIR Value. This could either be a ConstantOp, or
  /// an expanded sequence of ops in the current function's entry block (for
  /// ConstantExprs or ConstantGEPs).
  Value *processConstant(llvm::Constant *c);

  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder b;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;
  /// The entry block of the current function being processed.
  Block *currentEntryBlock;

  /// Globals are inserted before the first function, if any.
  Block::iterator getGlobalInsertPt() {
    auto i = module.getBody()->begin();
    while (!isa<LLVMFuncOp>(i) && !isa<ModuleTerminatorOp>(i))
      ++i;
    return i;
  }

  /// Functions are always inserted before the module terminator.
  Block::iterator getFuncInsertPt() {
    return std::prev(module.getBody()->end());
  }

  /// Remapped blocks, for the current function.
  DenseMap<llvm::BasicBlock *, Block *> blocks;
  /// Remapped values. These are function-local.
  DenseMap<llvm::Value *, Value *> instMap;
  /// Instructions that had not been defined when first encountered as a use.
  /// Maps to the dummy Operation that was created in processValue().
  DenseMap<llvm::Value *, Operation *> unknownInstMap;
  /// Uniquing map of GlobalVariables.
  DenseMap<llvm::GlobalVariable *, GlobalOp> globals;
  /// Cached FileLineColLoc::get("imported-bitcode", 0, 0).
  Location unknownLoc;
  /// Cached dialect.
  LLVMDialect *dialect;
};
} // namespace

Location Importer::processDebugLoc(const llvm::DebugLoc &loc,
                                   llvm::Instruction *inst) {
  if (!loc && inst) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "llvm-imported-inst-%";
    inst->printAsOperand(os, /*PrintType=*/false);
    return FileLineColLoc::get(os.str(), 0, 0, context);
  } else if (!loc) {
    return unknownLoc;
  }
  // FIXME: Obtain the filename from DILocationInfo.
  return FileLineColLoc::get("imported-bitcode", loc.getLine(), loc.getCol(),
                             context);
}

LLVMType Importer::processType(llvm::Type *type) {
  switch (type->getTypeID()) {
  case llvm::Type::FloatTyID:
    return LLVMType::getFloatTy(dialect);
  case llvm::Type::DoubleTyID:
    return LLVMType::getDoubleTy(dialect);
  case llvm::Type::IntegerTyID:
    return LLVMType::getIntNTy(dialect, type->getIntegerBitWidth());
  case llvm::Type::PointerTyID:
    return processType(type->getPointerElementType())
        .getPointerTo(type->getPointerAddressSpace());
  case llvm::Type::ArrayTyID:
    return LLVMType::getArrayTy(processType(type->getArrayElementType()),
                                type->getArrayNumElements());
  case llvm::Type::VectorTyID: {
    if (type->getVectorIsScalable())
      emitError(unknownLoc) << "scalable vector types not supported";
    return LLVMType::getVectorTy(processType(type->getVectorElementType()),
                                 type->getVectorNumElements());
  }
  case llvm::Type::VoidTyID:
    return LLVMType::getVoidTy(dialect);
  case llvm::Type::FP128TyID:
    return LLVMType::getFP128Ty(dialect);
  case llvm::Type::X86_FP80TyID:
    return LLVMType::getX86_FP80Ty(dialect);
  case llvm::Type::StructTyID: {
    SmallVector<LLVMType, 4> elementTypes;
    for (unsigned i = 0, e = type->getStructNumElements(); i != e; ++i)
      elementTypes.push_back(processType(type->getStructElementType(i)));
    return LLVMType::getStructTy(dialect, elementTypes,
                                 cast<llvm::StructType>(type)->isPacked());
  }
  case llvm::Type::FunctionTyID: {
    llvm::FunctionType *fty = cast<llvm::FunctionType>(type);
    SmallVector<LLVMType, 4> paramTypes;
    for (unsigned i = 0, e = fty->getNumParams(); i != e; ++i)
      paramTypes.push_back(processType(fty->getParamType(i)));
    return LLVMType::getFunctionTy(processType(fty->getReturnType()),
                                   paramTypes, fty->isVarArg());
  }
  default: {
    // FIXME: Diagnostic should be able to natively handle types that have
    // operator<<(raw_ostream&) defined.
    std::string s;
    llvm::raw_string_ostream os(s);
    os << *type;
    emitError(unknownLoc) << "unhandled type: " << os.str();
    return {};
  }
  }
}

// Get the given constant as an attribute. Not all constants can be represented
// as attributes.
Attribute Importer::getConstantAsAttr(llvm::Constant *value) {
  if (auto *ci = dyn_cast<llvm::ConstantInt>(value))
    return b.getIntegerAttr(
        IntegerType::get(ci->getType()->getBitWidth(), context),
        ci->getValue());
  if (auto *c = dyn_cast<llvm::ConstantDataArray>(value))
    if (c->isString())
      return b.getStringAttr(c->getAsString());
  return Attribute();
}

GlobalOp Importer::processGlobal(llvm::GlobalVariable *GV) {
  auto it = globals.find(GV);
  if (it != globals.end())
    return it->second;

  OpBuilder b(module.getBody(), getGlobalInsertPt());
  Attribute valueAttr;
  if (GV->hasInitializer())
    valueAttr = getConstantAsAttr(GV->getInitializer());
  GlobalOp op = b.create<GlobalOp>(UnknownLoc::get(context),
                                   processType(GV->getValueType()),
                                   GV->isConstant(), GV->getName(), valueAttr);
  if (GV->hasInitializer() && !valueAttr) {
    Region &r = op.getInitializerRegion();
    currentEntryBlock = b.createBlock(&r);
    b.setInsertionPoint(currentEntryBlock, currentEntryBlock->begin());
    Value *v = processConstant(GV->getInitializer());
    b.create<ReturnOp>(op.getLoc(), ArrayRef<Value *>({v}));
  }
  return globals[GV] = op;
}

Value *Importer::processConstant(llvm::Constant *c) {
  if (Attribute attr = getConstantAsAttr(c)) {
    // These constants can be represented as attributes.
    OpBuilder b(currentEntryBlock, currentEntryBlock->begin());
    return instMap[c] = b.create<ConstantOp>(unknownLoc,
                                             processType(c->getType()), attr);
  }
  if (auto *cn = dyn_cast<llvm::ConstantPointerNull>(c)) {
    OpBuilder b(currentEntryBlock, currentEntryBlock->begin());
    return instMap[c] =
               b.create<NullOp>(unknownLoc, processType(cn->getType()));
  }
  if (auto *ce = dyn_cast<llvm::ConstantExpr>(c)) {
    llvm::Instruction *i = ce->getAsInstruction();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(currentEntryBlock, currentEntryBlock->begin());
    if (failed(processInstruction(i)))
      return nullptr;
    assert(instMap.count(i));

    // Remove this zombie LLVM instruction now, leaving us only with the MLIR
    // op.
    i->deleteValue();
    return instMap[c] = instMap[i];
  }
  emitError(unknownLoc) << "unhandled constant: " << diag(*c);
  return nullptr;
}

Value *Importer::processValue(llvm::Value *value) {
  auto it = instMap.find(value);
  if (it != instMap.end())
    return it->second;

  // We don't expect to see instructions in dominator order. If we haven't seen
  // this instruction yet, create an unknown op and remap it later.
  if (isa<llvm::Instruction>(value)) {
    OperationState state(UnknownLoc::get(context), "unknown");
    state.addTypes({processType(value->getType())});
    unknownInstMap[value] = b.createOperation(state);
    return unknownInstMap[value]->getResult(0);
  }

  if (auto *GV = dyn_cast<llvm::GlobalVariable>(value)) {
    return b.create<AddressOfOp>(UnknownLoc::get(context), processGlobal(GV),
                                 ArrayRef<NamedAttribute>());
  }

  // Note, constant global variables are both GlobalVariables and Constants,
  // so we handle GlobalVariables first above.
  if (auto *c = dyn_cast<llvm::Constant>(value))
    return processConstant(c);

  emitError(unknownLoc) << "unhandled value: " << diag(*value);
  return nullptr;
}

// Maps from LLVM opcode to MLIR OperationName. This is deliberately ordered
// as in llvm/IR/Instructions.def to aid comprehension and spot missing
// instructions.
#define INST(llvm_n, mlir_n)                                                   \
  { llvm::Instruction::llvm_n, LLVM::mlir_n##Op::getOperationName() }
static const DenseMap<unsigned, StringRef> opcMap = {
    // Ret is handled specially.
    // Br is handled specially.
    // FIXME: switch
    // FIXME: indirectbr
    // FIXME: invoke
    // FIXME: resume
    // FIXME: unreachable
    // FIXME: cleanupret
    // FIXME: catchret
    // FIXME: catchswitch
    // FIXME: callbr
    // FIXME: fneg
    INST(Add, Add), INST(FAdd, FAdd), INST(Sub, Sub), INST(FSub, FSub),
    INST(Mul, Mul), INST(FMul, FMul), INST(UDiv, UDiv), INST(SDiv, SDiv),
    INST(FDiv, FDiv), INST(URem, URem), INST(SRem, SRem), INST(FRem, FRem),
    INST(Shl, Shl), INST(LShr, LShr), INST(AShr, AShr), INST(And, And),
    INST(Or, Or), INST(Xor, XOr), INST(Alloca, Alloca), INST(Load, Load),
    INST(Store, Store),
    // Getelementptr is handled specially.
    INST(Ret, Return),
    // FIXME: fence
    // FIXME: atomiccmpxchg
    // FIXME: atomicrmw
    INST(Trunc, Trunc), INST(ZExt, ZExt), INST(SExt, SExt),
    INST(FPToUI, FPToUI), INST(FPToSI, FPToSI), INST(UIToFP, UIToFP),
    INST(SIToFP, SIToFP), INST(FPTrunc, FPTrunc), INST(FPExt, FPExt),
    INST(PtrToInt, PtrToInt), INST(IntToPtr, IntToPtr), INST(BitCast, Bitcast),
    INST(AddrSpaceCast, AddrSpaceCast),
    // FIXME: cleanuppad
    // FIXME: catchpad
    // ICmp is handled specially.
    // FIXME: fcmp
    // PHI is handled specially.
    INST(Call, Call),
    // FIXME: select
    // FIXME: vaarg
    // FIXME: extractelement
    // FIXME: insertelement
    // FIXME: shufflevector
    // FIXME: extractvalue
    // FIXME: insertvalue
    // FIXME: landingpad
};
#undef INST

static ICmpPredicate getICmpPredicate(llvm::CmpInst::Predicate p) {
  switch (p) {
  default:
    llvm_unreachable("incorrect comparison predicate");
  case llvm::CmpInst::Predicate::ICMP_EQ:
    return LLVM::ICmpPredicate::eq;
  case llvm::CmpInst::Predicate::ICMP_NE:
    return LLVM::ICmpPredicate::ne;
  case llvm::CmpInst::Predicate::ICMP_SLT:
    return LLVM::ICmpPredicate::slt;
  case llvm::CmpInst::Predicate::ICMP_SLE:
    return LLVM::ICmpPredicate::sle;
  case llvm::CmpInst::Predicate::ICMP_SGT:
    return LLVM::ICmpPredicate::sgt;
  case llvm::CmpInst::Predicate::ICMP_SGE:
    return LLVM::ICmpPredicate::sge;
  case llvm::CmpInst::Predicate::ICMP_ULT:
    return LLVM::ICmpPredicate::ult;
  case llvm::CmpInst::Predicate::ICMP_ULE:
    return LLVM::ICmpPredicate::ule;
  case llvm::CmpInst::Predicate::ICMP_UGT:
    return LLVM::ICmpPredicate::ugt;
  case llvm::CmpInst::Predicate::ICMP_UGE:
    return LLVM::ICmpPredicate::uge;
  }
  llvm_unreachable("incorrect comparison predicate");
}

// `br` branches to `target`. Return the branch arguments to `br`, in the
// same order of the PHIs in `target`.
SmallVector<Value *, 4> Importer::processBranchArgs(llvm::BranchInst *br,
                                                    llvm::BasicBlock *target) {
  SmallVector<Value *, 4> v;
  for (auto inst = target->begin(); isa<llvm::PHINode>(inst); ++inst) {
    auto *PN = cast<llvm::PHINode>(&*inst);
    v.push_back(processValue(PN->getIncomingValueForBlock(br->getParent())));
  }
  return v;
}

LogicalResult Importer::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData. Currently inbounds GEPs, fast-math
  // flags and call / operand attributes are not supported.
  Location loc = processDebugLoc(inst->getDebugLoc(), inst);
  Value *&v = instMap[inst];
  assert(!v && "processInstruction must be called only once per instruction!");
  switch (inst->getOpcode()) {
  default:
    return emitError(loc) << "unknown instruction: " << diag(*inst);
  case llvm::Instruction::Add:
  case llvm::Instruction::FAdd:
  case llvm::Instruction::Sub:
  case llvm::Instruction::FSub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
  case llvm::Instruction::FRem:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Alloca:
  case llvm::Instruction::Load:
  case llvm::Instruction::Store:
  case llvm::Instruction::Ret:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::FPToUI:
  case llvm::Instruction::FPToSI:
  case llvm::Instruction::UIToFP:
  case llvm::Instruction::SIToFP:
  case llvm::Instruction::FPTrunc:
  case llvm::Instruction::FPExt:
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::IntToPtr:
  case llvm::Instruction::AddrSpaceCast:
  case llvm::Instruction::BitCast: {
    OperationState state(loc, opcMap.lookup(inst->getOpcode()));
    SmallVector<Value *, 4> ops;
    ops.reserve(inst->getNumOperands());
    for (auto *op : inst->operand_values())
      ops.push_back(processValue(op));
    state.addOperands(ops);
    if (!inst->getType()->isVoidTy())
      state.addTypes(ArrayRef<Type>({processType(inst->getType())}));
    Operation *op = b.createOperation(state);
    if (!inst->getType()->isVoidTy())
      v = op->getResult(0);
    return success();
  }
  case llvm::Instruction::ICmp: {
    v = b.create<ICmpOp>(
        loc, getICmpPredicate(cast<llvm::ICmpInst>(inst)->getPredicate()),
        processValue(inst->getOperand(0)), processValue(inst->getOperand(1)));
    return success();
  }
  case llvm::Instruction::Br: {
    auto *brInst = cast<llvm::BranchInst>(inst);
    OperationState state(loc,
                         brInst->isConditional() ? "llvm.cond_br" : "llvm.br");
    SmallVector<Value *, 4> ops;
    if (brInst->isConditional())
      ops.push_back(processValue(brInst->getCondition()));
    state.addOperands(ops);
    SmallVector<Block *, 4> succs;
    for (auto *succ : llvm::reverse(brInst->successors()))
      state.addSuccessor(blocks[succ], processBranchArgs(brInst, succ));
    b.createOperation(state);
    return success();
  }
  case llvm::Instruction::PHI: {
    v = b.getInsertionBlock()->addArgument(processType(inst->getType()));
    return success();
  }
  case llvm::Instruction::Call: {
    llvm::CallInst *ci = cast<llvm::CallInst>(inst);
    SmallVector<Value *, 4> ops;
    ops.reserve(inst->getNumOperands());
    for (auto &op : ci->arg_operands())
      ops.push_back(processValue(op.get()));

    SmallVector<Type, 2> tys;
    if (!ci->getType()->isVoidTy())
      tys.push_back(processType(inst->getType()));
    Operation *op;
    if (llvm::Function *callee = ci->getCalledFunction()) {
      op = b.create<CallOp>(loc, tys, b.getSymbolRefAttr(callee->getName()),
                            ops);
    } else {
      ops.insert(ops.begin(), processValue(ci->getCalledValue()));
      op = b.create<CallOp>(loc, tys, ops, ArrayRef<NamedAttribute>());
    }
    if (!ci->getType()->isVoidTy())
      v = op->getResult(0);
    return success();
  }
  case llvm::Instruction::GetElementPtr: {
    // FIXME: Support inbounds GEPs.
    llvm::GetElementPtrInst *gep = cast<llvm::GetElementPtrInst>(inst);
    SmallVector<Value *, 4> ops;
    for (auto *op : gep->operand_values())
      ops.push_back(processValue(op));
    v = b.create<GEPOp>(loc, processType(inst->getType()), ops,
                        ArrayRef<NamedAttribute>());
    return success();
  }
  }
}

LogicalResult Importer::processFunction(llvm::Function *f) {
  blocks.clear();
  instMap.clear();
  unknownInstMap.clear();

  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  LLVMFuncOp fop = b.create<LLVMFuncOp>(UnknownLoc::get(context), f->getName(),
                                        processType(f->getFunctionType()));
  if (f->isDeclaration())
    return success();

  // Eagerly create all blocks.
  SmallVector<Block *, 4> blockList;
  for (llvm::BasicBlock &bb : *f) {
    blockList.push_back(b.createBlock(&fop.body(), fop.body().end()));
    blocks[&bb] = blockList.back();
  }
  currentEntryBlock = blockList[0];

  // Add function arguments to the entry block.
  for (auto &arg : f->args())
    instMap[&arg] = blockList[0]->addArgument(processType(arg.getType()));

  for (auto bbs : llvm::zip(*f, blockList)) {
    if (failed(processBasicBlock(&std::get<0>(bbs), std::get<1>(bbs))))
      return failure();
  }

  // Now that all instructions are guaranteed to have been visited, ensure
  // any unknown uses we encountered are remapped.
  for (auto &llvmAndUnknown : unknownInstMap) {
    assert(instMap.count(llvmAndUnknown.first));
    Value *newValue = instMap[llvmAndUnknown.first];
    Value *oldValue = llvmAndUnknown.second->getResult(0);
    oldValue->replaceAllUsesWith(newValue);
    llvmAndUnknown.second->erase();
  }
  return success();
}

LogicalResult Importer::processBasicBlock(llvm::BasicBlock *bb, Block *block) {
  b.setInsertionPointToStart(block);
  for (llvm::Instruction &inst : *bb) {
    if (failed(processInstruction(&inst)))
      return failure();
  }
  return success();
}

OwningModuleRef
mlir::translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                              MLIRContext *context) {
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));

  Importer deserializer(context, module.get());
  for (llvm::GlobalVariable &gv : llvmModule->globals()) {
    if (!deserializer.processGlobal(&gv))
      return {};
  }
  for (llvm::Function &f : llvmModule->functions()) {
    if (failed(deserializer.processFunction(&f)))
      return {};
  }

  return module;
}

// Deserializes the LLVM bitcode stored in `input` into an MLIR module in the
// LLVM dialect.
OwningModuleRef translateLLVMIRToModule(llvm::SourceMgr &sourceMgr,
                                        MLIRContext *context) {
  LLVMDialect *dialect = context->getRegisteredDialect<LLVMDialect>();
  assert(dialect && "Could not find LLVMDialect?");

  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> llvmModule =
      llvm::parseIR(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), err,
                    dialect->getLLVMContext(),
                    /*UpgradeDebugInfo=*/true,
                    /*DataLayoutString=*/"");
  if (!llvmModule) {
    std::string errStr;
    llvm::raw_string_ostream errStream(errStr);
    err.print(/*ProgName=*/"", errStream);
    emitError(UnknownLoc::get(context)) << errStream.str();
    return {};
  }
  return translateLLVMIRToModule(std::move(llvmModule), context);
}

static TranslateToMLIRRegistration
    fromLLVM("import-llvm",
             [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
               return translateLLVMIRToModule(sourceMgr, context);
             });
