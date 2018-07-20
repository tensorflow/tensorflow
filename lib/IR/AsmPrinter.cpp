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

template <typename Container, typename UnaryFunctor>
inline void interleaveComma(raw_ostream &os, const Container &c,
                            UnaryFunctor each_fn) {
  interleave(c.begin(), c.end(), each_fn, [&]() { os << ", "; });
}

//===----------------------------------------------------------------------===//
// Module printing
//===----------------------------------------------------------------------===//

namespace {
class ModuleState {
public:
  ModuleState(raw_ostream &os);

  void initialize(const Module *module);

  void print(const Module *module);
  void print(const Attribute *attr) const;
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
  void visitAttribute(const Attribute *attr);
  void visitOperation(const Operation *op);

  void printAffineMapId(int affineMapId) const;
  void printAffineMapReference(const AffineMap* affineMap) const;

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

void ModuleState::visitAttribute(const Attribute *attr) {
  if (isa<AffineMapAttr>(attr)) {
    recordAffineMapReference(cast<AffineMapAttr>(attr)->getValue());
  } else if (isa<ArrayAttr>(attr)) {
    for (auto elt : cast<ArrayAttr>(attr)->getValue()) {
      visitAttribute(elt);
    }
  }
}

void ModuleState::visitOperation(const Operation *op) {
  for (auto elt : op->getAttrs()) {
    visitAttribute(elt.second);
  }
}

void ModuleState::visitExtFunction(const ExtFunction *fn) {
  visitType(fn->getType());
}

void ModuleState::visitCFGFunction(const CFGFunction *fn) {
  visitType(fn->getType());
  // TODO Visit function body instructions.
  for (auto &block : *fn) {
    for (auto &op : block.getOperations()) {
      visitOperation(&op);
    }
  }
}

void ModuleState::visitMLFunction(const MLFunction *fn) {
  visitType(fn->getType());
  // TODO Visit function body statements (and attributes if required).
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
void ModuleState::printAffineMapId(int affineMapId) const {
  os << "#map" << affineMapId;
}

void ModuleState::printAffineMapReference(const AffineMap* affineMap) const {
  const int mapId = getAffineMapId(affineMap);
  if (mapId >= 0) {
    // Map will be printed at top of module so print reference to its id.
    printAffineMapId(mapId);
  } else {
    // Map not in module state so print inline.
    affineMap->print(os);
  }
}

void ModuleState::print(const Module *module) {
  for (const auto &mapAndId : affineMapIds) {
    printAffineMapId(mapAndId.second);
    os << " = ";
    mapAndId.first->print(os);
    os << '\n';
  }
  for (auto *fn : module->functionList) print(fn);
}

void ModuleState::print(const Attribute *attr) const {
  switch (attr->getKind()) {
  case Attribute::Kind::Bool:
    os << (cast<BoolAttr>(attr)->getValue() ? "true" : "false");
    break;
  case Attribute::Kind::Integer:
    os << cast<IntegerAttr>(attr)->getValue();
    break;
  case Attribute::Kind::Float:
    // FIXME: this isn't precise, we should print with a hex format.
    os << cast<FloatAttr>(attr)->getValue();
    break;
  case Attribute::Kind::String:
    // FIXME: should escape the string.
    os << '"' << cast<StringAttr>(attr)->getValue() << '"';
    break;
  case Attribute::Kind::Array: {
    auto elts = cast<ArrayAttr>(attr)->getValue();
    os << '[';
    interleaveComma(os, elts, [&](Attribute *attr) { print(attr); });
    os << ']';
    break;
  }
  case Attribute::Kind::AffineMap:
    printAffineMapReference(cast<AffineMapAttr>(attr)->getValue());
    break;
  }
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
    interleaveComma(os, func->getInputs(), [&](Type *type) { os << *type; });
    os << ") -> ";
    auto results = func->getResults();
    if (results.size() == 1)
      os << *results[0];
    else {
      os << '(';
      interleaveComma(os, results, [&](Type *type) { os << *type; });
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
      printAffineMapReference(map);
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
  interleaveComma(os, type->getInputs(),
                  [&](Type *eltType) { moduleState->print(eltType); });
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
    interleaveComma(os, type->getResults(),
                    [&](Type *eltType) { moduleState->print(eltType); });
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

  void numberValueID(const SSAValue *value) {
    assert(!valueIDs.count(value) && "Value numbered multiple times");
    valueIDs[value] = nextValueID++;
  }

  void printValueID(const SSAValue *value) const {
    // TODO: If this is the result of an operation with multiple results, look
    // up the first result, and print the #32 syntax.
    auto it = valueIDs.find(value);
    if (it != valueIDs.end())
      os << '%' << it->getSecond();
    else
      os << "<<INVALID SSA VALUE>>";
  }

private:
  /// This is the value ID for each SSA value in the current function.
  DenseMap<const SSAValue *, unsigned> valueIDs;
  unsigned nextValueID = 0;
};
}  // end anonymous namespace

FunctionState::FunctionState(MLIRContext *context,
                             const ModuleState *moduleState, raw_ostream &os)
    : os(os),
      moduleState(moduleState),
      operationSet(OperationSet::get(context)) {}

void FunctionState::printOperation(const Operation *op) {
  os << "  ";

  // TODO: When we have SSAValue version of operands & results wired into
  // Operation this check can go away.
  if (auto *inst = dyn_cast<OperationInst>(op)) {
    if (inst->getNumResults()) {
      printValueID(inst->getResult(0));
      os << " = ";
    }
  }

  // Check to see if this is a known operation.  If so, use the registered
  // custom printer hook.
  if (auto opInfo = operationSet.lookup(op->getName().str())) {
    opInfo->printAssembly(op, os);
    return;
  }

  // Otherwise use the standard verbose printing approach.

  // TODO: escape name if necessary.
  os << "\"" << op->getName().str() << "\"(";

  // TODO: When we have SSAValue version of operands & results wired into
  // Operation this check can go away.
  if (auto *inst = dyn_cast<OperationInst>(op)) {
    // TODO: Use getOperands() when we have it.
    interleaveComma(
        os, inst->getInstOperands(),
        [&](const InstOperand &operand) { printValueID(operand.get()); });
  }

  os << ')';
  auto attrs = op->getAttrs();
  if (!attrs.empty()) {
    os << '{';
    interleaveComma(os, attrs, [&](NamedAttribute attr) {
      os << attr.first << ": ";
      moduleState->print(attr.second);
    });
    os << '}';
  }

  // TODO: When we have SSAValue version of operands & results wired into
  // Operation this check can go away.
  if (auto *inst = dyn_cast<OperationInst>(op)) {
    // Print the type signature of the operation.
    os << " : (";
    // TODO: Switch to getOperands() when we have it.
    interleaveComma(os, inst->getInstOperands(), [&](const InstOperand &op) {
      moduleState->print(op.get()->getType());
    });
    os << ") -> ";

    // TODO: Switch to getResults() when we have it.
    if (inst->getNumResults() == 1) {
      moduleState->print(inst->getInstResult(0).getType());
    } else {
      os << '(';
      interleaveComma(os, inst->getInstResults(),
                      [&](const InstResult &result) {
                        moduleState->print(result.getType());
                      });
      os << ')';
    }
  }
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

  void numberBlock(const BasicBlock *block);
};
}  // end anonymous namespace

CFGFunctionState::CFGFunctionState(const CFGFunction *function,
                                   const ModuleState *moduleState,
                                   raw_ostream &os)
    : FunctionState(function->getContext(), moduleState, os),
      function(function) {
  // Each basic block gets a unique ID per function.
  unsigned blockID = 0;
  for (auto &block : *function) {
    basicBlockIDs[&block] = blockID++;
    numberBlock(&block);
  }
}

/// Number all of the SSA values in the specified basic block.
void CFGFunctionState::numberBlock(const BasicBlock *block) {
  // TODO: basic block arguments.
  for (auto &op : *block) {
    // We number instruction that have results, and we only number the first
    // result.
    if (op.getNumResults() != 0)
      numberValueID(op.getResult(0));
  }

  // Terminators do not define values.
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
  os.indent(numSpaces) << "for x = " << *stmt->getLowerBound();
  os << " to " << *stmt->getUpperBound();
  if (stmt->getStep()->getValue() != 1)
    os << " step " << *stmt->getStep();

  os << " {\n";
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

void Attribute::print(raw_ostream &os) const {
  ModuleState moduleState(os);
  moduleState.print(this);
}

void Attribute::dump() const {
  print(llvm::errs());
}

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

void AffineSymbolExpr::print(raw_ostream &os) const {
  os << 's' << getPosition();
}

void AffineDimExpr::print(raw_ostream &os) const { os << 'd' << getPosition(); }

void AffineConstantExpr::print(raw_ostream &os) const { os << getValue(); }

static void printAdd(const AffineBinaryOpExpr *addExpr, raw_ostream &os) {
  os << '(' << *addExpr->getLHS();

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto *rhs = dyn_cast<AffineBinaryOpExpr>(addExpr->getRHS())) {
    if (rhs->getKind() == AffineExpr::Kind::Mul) {
      if (auto *rrhs = dyn_cast<AffineConstantExpr>(rhs->getRHS())) {
        if (rrhs->getValue() < 0) {
          os << " - (" << *rhs->getLHS() << " * " << -rrhs->getValue() << "))";
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto *rhs = dyn_cast<AffineConstantExpr>(addExpr->getRHS())) {
    if (rhs->getValue() < 0) {
      os << " - " << -rhs->getValue() << ")";
      return;
    }
  }

  os << " + " << *addExpr->getRHS() << ")";
}

void AffineBinaryOpExpr::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::Add:
    return printAdd(this, os);
  case Kind::Mul:
    os << "(" << *getLHS() << " * " << *getRHS() << ")";
    return;
  case Kind::FloorDiv:
    os << "(" << *getLHS() << " floordiv " << *getRHS() << ")";
    return;
  case Kind::CeilDiv:
    os << "(" << *getLHS() << " ceildiv " << *getRHS() << ")";
    return;
  case Kind::Mod:
    os << "(" << *getLHS() << " mod " << *getRHS() << ")";
    return;
  default:
    llvm_unreachable("unexpected affine binary op expression");
  }
}

void AffineExpr::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::SymbolId:
    return cast<AffineSymbolExpr>(this)->print(os);
  case Kind::DimId:
    return cast<AffineDimExpr>(this)->print(os);
  case Kind::Constant:
    return cast<AffineConstantExpr>(this)->print(os);
  case Kind::Add:
  case Kind::Mul:
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod:
    return cast<AffineBinaryOpExpr>(this)->print(os);
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
  interleaveComma(os, getResults(), [&](AffineExpr *expr) { os << *expr; });
  os << ")";

  if (!isBounded()) {
    return;
  }

  // Print range sizes for bounded affine maps.
  os << " size (";
  interleaveComma(os, getRangeSizes(), [&](AffineExpr *expr) { os << *expr; });
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
