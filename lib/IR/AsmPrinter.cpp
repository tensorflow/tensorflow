//===- AsmPrinter.cpp - MLIR Assembly Printer Implementation --------------===//
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
// This file implements the MLIR AsmPrinter class, which is used to implement
// the various print() methods on the core IR objects.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

void Identifier::print(raw_ostream &os) const { os << str(); }

void Identifier::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// Module printing
//===----------------------------------------------------------------------===//

namespace {
class ModuleState {
public:
  ModuleState(raw_ostream &os);

  void initialize(const Module *module);

  void print(const Module *module);
  void print(const Type *type) const;
  void print(const Function *fn);
  void print(const ExtFunction *fn);
  void print(const CFGFunction *fn);
  void print(const MLFunction *fn);

  void recordAffineMapReference(const AffineMap *affineMap) {
    if (affineMapIds.count(affineMap) == 0) {
      affineMapIds[affineMap] = nextAffineMapId++;
    }
  }

  int getAffineMapId(const AffineMap *affineMap) const {
    auto it = affineMapIds.find(affineMap);
    if (it == affineMapIds.end()) {
      return -1;
    }
    return it->second;
  }

private:
  // Visit functions.
  void visitFunction(const Function *fn);
  void visitExtFunction(const ExtFunction *fn);
  void visitCFGFunction(const CFGFunction *fn);
  void visitMLFunction(const MLFunction *fn);
  void visitType(const Type *type);

  raw_ostream &os;
  DenseMap<const AffineMap *, int> affineMapIds;
  int nextAffineMapId = 0;
};
}  // end anonymous namespace

ModuleState::ModuleState(raw_ostream &os) : os(os) {}

// Initializes module state, populating affine map state.
void ModuleState::initialize(const Module *module) {
  for (auto fn : module->functionList) {
    visitFunction(fn);
  }
}

// TODO Support visiting other types/instructions when implemented.
void ModuleState::visitType(const Type *type) {
  if (type->getKind() == Type::Kind::Function) {
    // Visit input and result types for functions.
    auto *funcType = cast<FunctionType>(type);
    for (auto *input : funcType->getInputs()) {
      visitType(input);
    }
    for (auto *result : funcType->getResults()) {
      visitType(result);
    }
  } else if (type->getKind() == Type::Kind::MemRef) {
    // Visit affine maps in memref type.
    auto *memref = cast<MemRefType>(type);
    for (AffineMap *map : memref->getAffineMaps()) {
      recordAffineMapReference(map);
    }
  }
}

void ModuleState::visitExtFunction(const ExtFunction *fn) {
  visitType(fn->getType());
}

void ModuleState::visitCFGFunction(const CFGFunction *fn) {
  visitType(fn->getType());
  // TODO Visit function body instructions.
}

void ModuleState::visitMLFunction(const MLFunction *fn) {
  visitType(fn->getType());
  // TODO Visit function body statements.
}

void ModuleState::visitFunction(const Function *fn) {
  switch (fn->getKind()) {
  case Function::Kind::ExtFunc:
    return visitExtFunction(cast<ExtFunction>(fn));
  case Function::Kind::CFGFunc:
    return visitCFGFunction(cast<CFGFunction>(fn));
  case Function::Kind::MLFunc:
    return visitMLFunction(cast<MLFunction>(fn));
  }
}

// Prints function with initialized module state.
void ModuleState::print(const Function *fn) {
  switch (fn->getKind()) {
  case Function::Kind::ExtFunc:
    return print(cast<ExtFunction>(fn));
  case Function::Kind::CFGFunc:
    return print(cast<CFGFunction>(fn));
  case Function::Kind::MLFunc:
    return print(cast<MLFunction>(fn));
  }
}

// Prints affine map identifier.
static void printAffineMapId(unsigned affineMapId, raw_ostream &os) {
  os << "#map" << affineMapId;
}

void ModuleState::print(const Module *module) {
  for (const auto &mapAndId : affineMapIds) {
    printAffineMapId(mapAndId.second, os);
    os << " = ";
    mapAndId.first->print(os);
    os << '\n';
  }
  for (auto *fn : module->functionList) print(fn);
}

void ModuleState::print(const Type *type) const {
  switch (type->getKind()) {
  case Type::Kind::AffineInt:
    os << "affineint";
    return;
  case Type::Kind::BF16:
    os << "bf16";
    return;
  case Type::Kind::F16:
    os << "f16";
    return;
  case Type::Kind::F32:
    os << "f32";
    return;
  case Type::Kind::F64:
    os << "f64";
    return;

  case Type::Kind::Integer: {
    auto *integer = cast<IntegerType>(type);
    os << 'i' << integer->getWidth();
    return;
  }
  case Type::Kind::Function: {
    auto *func = cast<FunctionType>(type);
    os << '(';
    interleave(func->getInputs(), [&](Type *type) { os << *type; },
               [&]() { os << ", "; });
    os << ") -> ";
    auto results = func->getResults();
    if (results.size() == 1)
      os << *results[0];
    else {
      os << '(';
      interleave(results, [&](Type *type) { os << *type; },
                 [&]() { os << ", "; });
      os << ')';
    }
    return;
  }
  case Type::Kind::Vector: {
    auto *v = cast<VectorType>(type);
    os << "vector<";
    for (auto dim : v->getShape()) os << dim << 'x';
    os << *v->getElementType() << '>';
    return;
  }
  case Type::Kind::RankedTensor: {
    auto *v = cast<RankedTensorType>(type);
    os << "tensor<";
    for (auto dim : v->getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    os << *v->getElementType() << '>';
    return;
  }
  case Type::Kind::UnrankedTensor: {
    auto *v = cast<UnrankedTensorType>(type);
    os << "tensor<??" << *v->getElementType() << '>';
    return;
  }
  case Type::Kind::MemRef: {
    auto *v = cast<MemRefType>(type);
    os << "memref<";
    for (auto dim : v->getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    os << *v->getElementType();
    for (auto map : v->getAffineMaps()) {
      os << ", ";
      const int mapId = getAffineMapId(map);
      if (mapId >= 0) {
        // Map will be printed at top of module so print reference to its id.
        printAffineMapId(mapId, os);
      } else {
        // Map not in module state so print inline.
        map->print(os);
      }
    }
    os << ", " << v->getMemorySpace();
    os << '>';
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
// Function printing
//===----------------------------------------------------------------------===//

static void printFunctionSignature(const Function *fn,
                                   const ModuleState *moduleState,
                                   raw_ostream &os) {
  auto type = fn->getType();

  os << "@" << fn->getName() << '(';
  interleave(type->getInputs(),
             [&](Type *eltType) { moduleState->print(eltType); },
             [&]() { os << ", "; });
  os << ')';

  switch (type->getResults().size()) {
  case 0:
    break;
  case 1:
    os << " -> ";
    moduleState->print(type->getResults()[0]);
    break;
  default:
    os << " -> (";
    interleave(type->getResults(),
               [&](Type *eltType) { moduleState->print(eltType); },
               [&]() { os << ", "; });
    os << ')';
    break;
  }
}

void ModuleState::print(const ExtFunction *fn) {
  os << "extfunc ";
  printFunctionSignature(fn, this, os);
  os << '\n';
}

namespace {

// FunctionState contains common functionality for printing
// CFG and ML functions.
class FunctionState {
public:
  FunctionState(MLIRContext *context, const ModuleState *moduleState,
                raw_ostream &os);

  void printOperation(const Operation *op);

protected:
  raw_ostream &os;
  const ModuleState *moduleState;
  const OperationSet &operationSet;
};
}  // end anonymous namespace

FunctionState::FunctionState(MLIRContext *context,
                             const ModuleState *moduleState, raw_ostream &os)
    : os(os),
      moduleState(moduleState),
      operationSet(OperationSet::get(context)) {}

void FunctionState::printOperation(const Operation *op) {
  // Check to see if this is a known operation.  If so, use the registered
  // custom printer hook.
  if (auto opInfo = operationSet.lookup(op->getName().str())) {
    os << "  ";
    opInfo->printAssembly(op, os);
    return;
  }

  // TODO: escape name if necessary.
  os << "  \"" << op->getName().str() << "\"()";

  auto attrs = op->getAttrs();
  if (!attrs.empty()) {
    os << '{';
    interleave(
        attrs,
        [&](NamedAttribute attr) { os << attr.first << ": " << *attr.second; },
        [&]() { os << ", "; });
    os << '}';
  }

  // TODO: Print signature type once that is plumbed through to Operation.
}

//===----------------------------------------------------------------------===//
// CFG Function printing
//===----------------------------------------------------------------------===//

namespace {
class CFGFunctionState : public FunctionState {
public:
  CFGFunctionState(const CFGFunction *function, const ModuleState *moduleState,
                   raw_ostream &os);

  const CFGFunction *getFunction() const { return function; }

  void print();
  void print(const BasicBlock *block);

  void print(const Instruction *inst);
  void print(const OperationInst *inst);
  void print(const ReturnInst *inst);
  void print(const BranchInst *inst);

  unsigned getBBID(const BasicBlock *block) {
    auto it = basicBlockIDs.find(block);
    assert(it != basicBlockIDs.end() && "Block not in this function?");
    return it->second;
  }

private:
  const CFGFunction *function;
  DenseMap<const BasicBlock *, unsigned> basicBlockIDs;
};
}  // end anonymous namespace

CFGFunctionState::CFGFunctionState(const CFGFunction *function,
                                   const ModuleState *moduleState,
                                   raw_ostream &os)
    : FunctionState(function->getContext(), moduleState, os),
      function(function) {
  // Each basic block gets a unique ID per function.
  unsigned blockID = 0;
  for (auto &block : *function) basicBlockIDs[&block] = blockID++;
}

void CFGFunctionState::print() {
  os << "cfgfunc ";
  printFunctionSignature(this->getFunction(), moduleState, os);
  os << " {\n";

  for (auto &block : *function) print(&block);
  os << "}\n\n";
}

void CFGFunctionState::print(const BasicBlock *block) {
  os << "bb" << getBBID(block) << ":\n";

  // TODO Print arguments.
  for (auto &inst : block->getOperations()) {
    print(&inst);
    os << "\n";
  }

  print(block->getTerminator());
  os << "\n";
}

void CFGFunctionState::print(const Instruction *inst) {
  switch (inst->getKind()) {
  case Instruction::Kind::Operation:
    return print(cast<OperationInst>(inst));
  case TerminatorInst::Kind::Branch:
    return print(cast<BranchInst>(inst));
  case TerminatorInst::Kind::Return:
    return print(cast<ReturnInst>(inst));
  }
}

void CFGFunctionState::print(const OperationInst *inst) {
  printOperation(inst);

  // FIXME: Move this into printOperation when Operation has operands and
  // results

  // Print the type signature of the operation.
  os << " : (";
  interleave(
      inst->getOperands(),
      [&](const InstOperand &op) { moduleState->print(op.get()->getType()); },
      [&]() { os << ", "; });
  os << ") -> ";

  auto resultList = inst->getResults();
  if (resultList.size() == 1) {
    moduleState->print(resultList[0].getType());
  } else {
    os << '(';
    interleave(
        resultList,
        [&](const InstResult &result) { moduleState->print(result.getType()); },
        [&]() { os << ", "; });
    os << ')';
  }
}
void CFGFunctionState::print(const BranchInst *inst) {
  os << "  br bb" << getBBID(inst->getDest());
}
void CFGFunctionState::print(const ReturnInst *inst) { os << "  return"; }

void ModuleState::print(const CFGFunction *fn) {
  CFGFunctionState state(fn, this, os);
  state.print();
}

//===----------------------------------------------------------------------===//
// ML Function printing
//===----------------------------------------------------------------------===//

namespace {
class MLFunctionState : public FunctionState {
public:
  MLFunctionState(const MLFunction *function, const ModuleState *moduleState,
                  raw_ostream &os);

  const MLFunction *getFunction() const { return function; }

  // Prints ML function
  void print();

  // Methods to print ML function statements
  void print(const Statement *stmt);
  void print(const OperationStmt *stmt);
  void print(const ForStmt *stmt);
  void print(const IfStmt *stmt);
  void print(const StmtBlock *block);

  // Number of spaces used for indenting nested statements
  const static unsigned indentWidth = 2;

private:
  const MLFunction *function;
  int numSpaces;
};
}  // end anonymous namespace

MLFunctionState::MLFunctionState(const MLFunction *function,
                                 const ModuleState *moduleState,
                                 raw_ostream &os)
    : FunctionState(function->getContext(), moduleState, os),
      function(function),
      numSpaces(0) {}

void MLFunctionState::print() {
  os << "mlfunc ";
  // FIXME: should print argument names rather than just signature
  printFunctionSignature(function, moduleState, os);
  os << " {\n";
  print(function);
  os << "  return\n";
  os << "}\n\n";
}

void MLFunctionState::print(const StmtBlock *block) {
  numSpaces += indentWidth;
  for (auto &stmt : block->getStatements()) {
    print(&stmt);
    os << "\n";
  }
  numSpaces -= indentWidth;
}

void MLFunctionState::print(const Statement *stmt) {
  switch (stmt->getKind()) {
  case Statement::Kind::Operation:
    return print(cast<OperationStmt>(stmt));
  case Statement::Kind::For:
    return print(cast<ForStmt>(stmt));
  case Statement::Kind::If:
    return print(cast<IfStmt>(stmt));
  }
}

void MLFunctionState::print(const OperationStmt *stmt) { printOperation(stmt); }

void MLFunctionState::print(const ForStmt *stmt) {
  os.indent(numSpaces) << "for {\n";
  print(static_cast<const StmtBlock *>(stmt));
  os.indent(numSpaces) << "}";
}

void MLFunctionState::print(const IfStmt *stmt) {
  os.indent(numSpaces) << "if () {\n";
  print(stmt->getThenClause());
  os.indent(numSpaces) << "}";
  if (stmt->hasElseClause()) {
    os << " else {\n";
    print(stmt->getElseClause());
    os.indent(numSpaces) << "}";
  }
}

void ModuleState::print(const MLFunction *fn) {
  MLFunctionState state(fn, this, os);
  state.print();
}

//===----------------------------------------------------------------------===//
// print and dump methods
//===----------------------------------------------------------------------===//

void Type::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  moduleState.print(this);
}

void Type::dump() const { print(llvm::errs()); }

void Instruction::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  CFGFunctionState state(getFunction(), &moduleState, os);
  state.print(this);
}

void Instruction::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineAddExpr::print(raw_ostream &os) const {
  os << "(" << *getLHS() << " + " << *getRHS() << ")";
}

void AffineSubExpr::print(raw_ostream &os) const {
  os << "(" << *getLHS() << " - " << *getRHS() << ")";
}

void AffineMulExpr::print(raw_ostream &os) const {
  os << "(" << *getLHS() << " * " << *getRHS() << ")";
}

void AffineModExpr::print(raw_ostream &os) const {
  os << "(" << *getLHS() << " mod " << *getRHS() << ")";
}

void AffineFloorDivExpr::print(raw_ostream &os) const {
  os << "(" << *getLHS() << " floordiv " << *getRHS() << ")";
}

void AffineCeilDivExpr::print(raw_ostream &os) const {
  os << "(" << *getLHS() << " ceildiv " << *getRHS() << ")";
}

void AffineSymbolExpr::print(raw_ostream &os) const {
  os << "s" << getPosition();
}

void AffineDimExpr::print(raw_ostream &os) const { os << "d" << getPosition(); }

void AffineConstantExpr::print(raw_ostream &os) const { os << getValue(); }

void AffineExpr::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::SymbolId:
    return cast<AffineSymbolExpr>(this)->print(os);
  case Kind::DimId:
    return cast<AffineDimExpr>(this)->print(os);
  case Kind::Constant:
    return cast<AffineConstantExpr>(this)->print(os);
  case Kind::Add:
    return cast<AffineAddExpr>(this)->print(os);
  case Kind::Sub:
    return cast<AffineSubExpr>(this)->print(os);
  case Kind::Mul:
    return cast<AffineMulExpr>(this)->print(os);
  case Kind::FloorDiv:
    return cast<AffineFloorDivExpr>(this)->print(os);
  case Kind::CeilDiv:
    return cast<AffineCeilDivExpr>(this)->print(os);
  case Kind::Mod:
    return cast<AffineModExpr>(this)->print(os);
  }
}

void AffineMap::print(raw_ostream &os) const {
  // Dimension identifiers.
  os << "(";
  for (int i = 0; i < (int)getNumDims() - 1; i++) os << "d" << i << ", ";
  if (getNumDims() >= 1) os << "d" << getNumDims() - 1;
  os << ")";

  // Symbolic identifiers.
  if (getNumSymbols() >= 1) {
    os << " [";
    for (int i = 0; i < (int)getNumSymbols() - 1; i++) os << "s" << i << ", ";
    if (getNumSymbols() >= 1) os << "s" << getNumSymbols() - 1;
    os << "]";
  }

  // AffineMap should have at least one result.
  assert(!getResults().empty());
  // Result affine expressions.
  os << " -> (";
  interleave(getResults(), [&](AffineExpr *expr) { os << *expr; },
             [&]() { os << ", "; });
  os << ")";

  if (!isBounded()) {
    return;
  }

  // Print range sizes for bounded affine maps.
  os << " size (";
  interleave(getRangeSizes(), [&](AffineExpr *expr) { os << *expr; },
             [&]() { os << ", "; });
  os << ")";
}

void BasicBlock::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  CFGFunctionState state(getFunction(), &moduleState, os);
  state.print();
}

void BasicBlock::dump() const { print(llvm::errs()); }

void Statement::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  MLFunctionState state(getFunction(), &moduleState, os);
  state.print(this);
}

void Statement::dump() const { print(llvm::errs()); }

void Function::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::ExtFunc:
    return cast<ExtFunction>(this)->print(os);
  case Kind::CFGFunc:
    return cast<CFGFunction>(this)->print(os);
  case Kind::MLFunc:
    return cast<MLFunction>(this)->print(os);
  }
}

void Function::dump() const { print(llvm::errs()); }

void ExtFunction::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  os << "extfunc ";
  printFunctionSignature(this, &moduleState, os);
  os << "\n";
}

void CFGFunction::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  CFGFunctionState state(this, &moduleState, os);
  state.print();
}

void MLFunction::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  MLFunctionState state(this, &moduleState, os);
  state.print();
}

void Module::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  moduleState.initialize(this);
  moduleState.print(this);
}

void Module::dump() const { print(llvm::errs()); }
