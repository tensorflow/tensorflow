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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
using namespace mlir;

void Identifier::print(raw_ostream &os) const { os << str(); }

void Identifier::dump() const { print(llvm::errs()); }

void OperationName::print(raw_ostream &os) const { os << getStringRef(); }

void OperationName::dump() const { print(llvm::errs()); }

OpAsmPrinter::~OpAsmPrinter() {}

//===----------------------------------------------------------------------===//
// ModuleState
//===----------------------------------------------------------------------===//

namespace {
class ModuleState {
public:
  /// This is the current context if it is knowable, otherwise this is null.
  MLIRContext *const context;

  explicit ModuleState(MLIRContext *context) : context(context) {}

  // Initializes module state, populating affine map state.
  void initialize(const Module *module);

  int getAffineMapId(AffineMap affineMap) const {
    auto it = affineMapIds.find(affineMap);
    if (it == affineMapIds.end()) {
      return -1;
    }
    return it->second;
  }

  ArrayRef<AffineMap> getAffineMapIds() const { return affineMapsById; }

  int getIntegerSetId(IntegerSet integerSet) const {
    auto it = integerSetIds.find(integerSet);
    if (it == integerSetIds.end()) {
      return -1;
    }
    return it->second;
  }

  ArrayRef<IntegerSet> getIntegerSetIds() const { return integerSetsById; }

private:
  void recordAffineMapReference(AffineMap affineMap) {
    if (affineMapIds.count(affineMap) == 0) {
      affineMapIds[affineMap] = affineMapsById.size();
      affineMapsById.push_back(affineMap);
    }
  }

  void recordIntegerSetReference(IntegerSet integerSet) {
    if (integerSetIds.count(integerSet) == 0) {
      integerSetIds[integerSet] = integerSetsById.size();
      integerSetsById.push_back(integerSet);
    }
  }

  // Return true if this map could be printed using the shorthand form.
  static bool hasShorthandForm(AffineMap boundMap) {
    if (boundMap.isSingleConstant())
      return true;

    // Check if the affine map is single dim id or single symbol identity -
    // (i)->(i) or ()[s]->(i)
    return boundMap.getNumInputs() == 1 && boundMap.getNumResults() == 1 &&
           (boundMap.getResult(0).isa<AffineDimExpr>() ||
            boundMap.getResult(0).isa<AffineSymbolExpr>());
  }

  // Visit functions.
  void visitFunction(const Function *fn);
  void visitExtFunction(const Function *fn);
  void visitCFGFunction(const CFGFunction *fn);
  void visitMLFunction(const MLFunction *fn);
  void visitStatement(const Statement *stmt);
  void visitForStmt(const ForStmt *forStmt);
  void visitIfStmt(const IfStmt *ifStmt);
  void visitOperationInst(const OperationInst *opStmt);
  void visitType(Type type);
  void visitAttribute(Attribute attr);
  void visitOperation(const OperationInst *op);

  DenseMap<AffineMap, int> affineMapIds;
  std::vector<AffineMap> affineMapsById;

  DenseMap<IntegerSet, int> integerSetIds;
  std::vector<IntegerSet> integerSetsById;
};
} // end anonymous namespace

// TODO Support visiting other types/instructions when implemented.
void ModuleState::visitType(Type type) {
  if (auto funcType = type.dyn_cast<FunctionType>()) {
    // Visit input and result types for functions.
    for (auto input : funcType.getInputs())
      visitType(input);
    for (auto result : funcType.getResults())
      visitType(result);
  } else if (auto memref = type.dyn_cast<MemRefType>()) {
    // Visit affine maps in memref type.
    for (auto map : memref.getAffineMaps()) {
      recordAffineMapReference(map);
    }
  }
}

void ModuleState::visitAttribute(Attribute attr) {
  if (auto mapAttr = attr.dyn_cast<AffineMapAttr>()) {
    recordAffineMapReference(mapAttr.getValue());
  } else if (auto setAttr = attr.dyn_cast<IntegerSetAttr>()) {
    recordIntegerSetReference(setAttr.getValue());
  } else if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
    for (auto elt : arrayAttr.getValue()) {
      visitAttribute(elt);
    }
  }
}

void ModuleState::visitOperation(const OperationInst *op) {
  // Visit all the types used in the operation.
  for (auto *operand : op->getOperands())
    visitType(operand->getType());
  for (auto *result : op->getResults())
    visitType(result->getType());

  // Visit each of the attributes.
  for (auto elt : op->getAttrs())
    visitAttribute(elt.second);
}

void ModuleState::visitExtFunction(const Function *fn) {
  visitType(fn->getType());
}

void ModuleState::visitCFGFunction(const CFGFunction *fn) {
  visitType(fn->getType());
  for (auto &block : *fn) {
    for (auto &op : block.getStatements()) {
      if (auto *opInst = dyn_cast<OperationInst>(&op))
        visitOperation(opInst);
      else {
        llvm_unreachable("IfStmt/ForStmt in a CFGFunction isn't supported");
      }
    }
  }
}

void ModuleState::visitIfStmt(const IfStmt *ifStmt) {
  recordIntegerSetReference(ifStmt->getIntegerSet());
  for (auto &childStmt : *ifStmt->getThen())
    visitStatement(&childStmt);
  if (ifStmt->hasElse())
    for (auto &childStmt : *ifStmt->getElse())
      visitStatement(&childStmt);
}

void ModuleState::visitForStmt(const ForStmt *forStmt) {
  AffineMap lbMap = forStmt->getLowerBoundMap();
  if (!hasShorthandForm(lbMap))
    recordAffineMapReference(lbMap);

  AffineMap ubMap = forStmt->getUpperBoundMap();
  if (!hasShorthandForm(ubMap))
    recordAffineMapReference(ubMap);

  for (auto &childStmt : *forStmt->getBody())
    visitStatement(&childStmt);
}

void ModuleState::visitOperationInst(const OperationInst *opStmt) {
  for (auto attr : opStmt->getAttrs())
    visitAttribute(attr.second);
}

void ModuleState::visitStatement(const Statement *stmt) {
  switch (stmt->getKind()) {
  case Statement::Kind::If:
    return visitIfStmt(cast<IfStmt>(stmt));
  case Statement::Kind::For:
    return visitForStmt(cast<ForStmt>(stmt));
  case Statement::Kind::OperationInst:
    return visitOperationInst(cast<OperationInst>(stmt));
  default:
    return;
  }
}

void ModuleState::visitMLFunction(const MLFunction *fn) {
  visitType(fn->getType());
  for (auto &stmt : *fn->getBody()) {
    ModuleState::visitStatement(&stmt);
  }
}

void ModuleState::visitFunction(const Function *fn) {
  switch (fn->getKind()) {
  case Function::Kind::ExtFunc:
    return visitExtFunction(fn);
  case Function::Kind::CFGFunc:
    return visitCFGFunction(fn);
  case Function::Kind::MLFunc:
    return visitMLFunction(fn);
  }
}

// Initializes module state, populating affine map and integer set state.
void ModuleState::initialize(const Module *module) {
  for (auto &fn : *module) {
    visitFunction(&fn);
  }
}

//===----------------------------------------------------------------------===//
// ModulePrinter
//===----------------------------------------------------------------------===//

namespace {
class ModulePrinter {
public:
  ModulePrinter(raw_ostream &os, ModuleState &state) : os(os), state(state) {}
  explicit ModulePrinter(const ModulePrinter &printer)
      : os(printer.os), state(printer.state) {}

  template <typename Container, typename UnaryFunctor>
  inline void interleaveComma(const Container &c, UnaryFunctor each_fn) const {
    interleave(c.begin(), c.end(), each_fn, [&]() { os << ", "; });
  }

  void print(const Module *module);
  void printFunctionReference(const Function *func);
  void printAttribute(Attribute attr);
  void printType(Type type);
  void print(const Function *fn);
  void printExt(const Function *fn);
  void printCFG(const Function *fn);
  void printML(const Function *fn);

  void printAffineMap(AffineMap map);
  void printAffineExpr(AffineExpr expr);
  void printAffineConstraint(AffineExpr expr, bool isEq);
  void printIntegerSet(IntegerSet set);

protected:
  raw_ostream &os;
  ModuleState &state;

  void printFunctionSignature(const Function *fn);
  void printFunctionAttributes(const Function *fn);
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<const char *> elidedAttrs = {});
  void printFunctionResultType(FunctionType type);
  void printAffineMapId(int affineMapId) const;
  void printAffineMapReference(AffineMap affineMap);
  void printIntegerSetId(int integerSetId) const;
  void printIntegerSetReference(IntegerSet integerSet);
  void printDenseElementsAttr(DenseElementsAttr attr);

  /// This enum is used to represent the binding stength of the enclosing
  /// context that an AffineExprStorage is being printed in, so we can
  /// intelligently produce parens.
  enum class BindingStrength {
    Weak,   // + and -
    Strong, // All other binary operators.
  };
  void printAffineExprInternal(AffineExpr expr,
                               BindingStrength enclosingTightness);
};
} // end anonymous namespace

// Prints function with initialized module state.
void ModulePrinter::print(const Function *fn) {
  switch (fn->getKind()) {
  case Function::Kind::ExtFunc:
    return printExt(fn);
  case Function::Kind::CFGFunc:
    return printCFG(fn);
  case Function::Kind::MLFunc:
    return printML(fn);
  }
}

// Prints affine map identifier.
void ModulePrinter::printAffineMapId(int affineMapId) const {
  os << "#map" << affineMapId;
}

void ModulePrinter::printAffineMapReference(AffineMap affineMap) {
  int mapId = state.getAffineMapId(affineMap);
  if (mapId >= 0) {
    // Map will be printed at top of module so print reference to its id.
    printAffineMapId(mapId);
  } else {
    // Map not in module state so print inline.
    affineMap.print(os);
  }
}

// Prints integer set identifier.
void ModulePrinter::printIntegerSetId(int integerSetId) const {
  os << "#set" << integerSetId;
}

void ModulePrinter::printIntegerSetReference(IntegerSet integerSet) {
  int setId;
  if ((setId = state.getIntegerSetId(integerSet)) >= 0) {
    // The set will be printed at top of module; so print reference to its id.
    printIntegerSetId(setId);
  } else {
    // Set not in module state so print inline.
    integerSet.print(os);
  }
}

void ModulePrinter::print(const Module *module) {
  for (const auto &map : state.getAffineMapIds()) {
    printAffineMapId(state.getAffineMapId(map));
    os << " = ";
    map.print(os);
    os << '\n';
  }
  for (const auto &set : state.getIntegerSetIds()) {
    printIntegerSetId(state.getIntegerSetId(set));
    os << " = ";
    set.print(os);
    os << '\n';
  }
  for (auto const &fn : *module)
    print(&fn);
}

/// Print a floating point value in a way that the parser will be able to
/// round-trip losslessly.
static void printFloatValue(const APFloat &apValue, raw_ostream &os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, 6, 0, false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");
    // Reparse stringized version!
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return;
    }
  }

  SmallVector<char, 16> str;
  apValue.toString(str);
  os << str;
}

void ModulePrinter::printFunctionReference(const Function *func) {
  os << '@' << func->getName();
}

void ModulePrinter::printAttribute(Attribute attr) {
  if (!attr) {
    os << "<<NULL ATTRIBUTE>>";
    return;
  }

  switch (attr.getKind()) {
  case Attribute::Kind::Bool:
    os << (attr.cast<BoolAttr>().getValue() ? "true" : "false");
    break;
  case Attribute::Kind::Integer: {
    auto intAttr = attr.cast<IntegerAttr>();
    // Print all integer attributes as signed unless i1.
    bool isSigned = intAttr.getType().isIndex() ||
                    intAttr.getType().getIntOrFloatBitWidth() != 1;
    intAttr.getValue().print(os, isSigned);
    break;
  }
  case Attribute::Kind::Float:
    printFloatValue(attr.cast<FloatAttr>().getValue(), os);
    break;
  case Attribute::Kind::String:
    os << '"';
    printEscapedString(attr.cast<StringAttr>().getValue(), os);
    os << '"';
    break;
  case Attribute::Kind::Array:
    os << '[';
    interleaveComma(attr.cast<ArrayAttr>().getValue(),
                    [&](Attribute attr) { printAttribute(attr); });
    os << ']';
    break;
  case Attribute::Kind::AffineMap:
    printAffineMapReference(attr.cast<AffineMapAttr>().getValue());
    break;
  case Attribute::Kind::IntegerSet:
    printIntegerSetReference(attr.cast<IntegerSetAttr>().getValue());
    break;
  case Attribute::Kind::Type:
    printType(attr.cast<TypeAttr>().getValue());
    break;
  case Attribute::Kind::Function: {
    auto *function = attr.cast<FunctionAttr>().getValue();
    if (!function) {
      os << "<<FUNCTION ATTR FOR DELETED FUNCTION>>";
    } else {
      printFunctionReference(function);
      os << " : ";
      printType(function->getType());
    }
    break;
  }
  case Attribute::Kind::OpaqueElements: {
    auto eltsAttr = attr.cast<OpaqueElementsAttr>();
    os << "opaque<";
    printType(eltsAttr.getType());
    os << ", " << '"' << "0x" << llvm::toHex(eltsAttr.getValue()) << '"' << '>';
    break;
  }
  case Attribute::Kind::DenseIntElements:
  case Attribute::Kind::DenseFPElements: {
    auto eltsAttr = attr.cast<DenseElementsAttr>();
    os << "dense<";
    printType(eltsAttr.getType());
    os << ", ";
    printDenseElementsAttr(eltsAttr);
    os << '>';
    break;
  }
  case Attribute::Kind::SplatElements: {
    auto elementsAttr = attr.cast<SplatElementsAttr>();
    os << "splat<";
    printType(elementsAttr.getType());
    os << ", ";
    printAttribute(elementsAttr.getValue());
    os << '>';
    break;
  }
  case Attribute::Kind::SparseElements: {
    auto elementsAttr = attr.cast<SparseElementsAttr>();
    os << "sparse<";
    printType(elementsAttr.getType());
    os << ", ";
    printDenseElementsAttr(elementsAttr.getIndices());
    os << ", ";
    printDenseElementsAttr(elementsAttr.getValues());
    os << '>';
    break;
  }
  }
}

void ModulePrinter::printDenseElementsAttr(DenseElementsAttr attr) {
  auto type = attr.getType();
  auto shape = type.getShape();
  auto rank = type.getRank();

  SmallVector<Attribute, 16> elements;
  attr.getValues(elements);

  // Special case for degenerate tensors.
  if (elements.empty()) {
    for (int i = 0; i < rank; ++i)
      os << '[';
    for (int i = 0; i < rank; ++i)
      os << ']';
    return;
  }

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned openBrackets = 0;

  auto bumpCounter = [&]() {
    // Bump the least significant digit.
    ++counter[rank - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank - 1; i > 0; --i)
      if (counter[i] >= shape[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter[i] = 0;
        ++counter[i - 1];
        --openBrackets;
        os << ']';
      }
  };

  for (unsigned idx = 0, e = elements.size(); idx != e; ++idx) {
    if (idx != 0)
      os << ", ";
    while (openBrackets++ < rank)
      os << '[';
    openBrackets = rank;
    printAttribute(elements[idx]);
    bumpCounter();
  }
  while (openBrackets-- > 0)
    os << ']';
}

void ModulePrinter::printType(Type type) {
  switch (type.getKind()) {
  case Type::Kind::Index:
    os << "index";
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
  case Type::Kind::TFControl:
    os << "tf_control";
    return;
  case Type::Kind::TFResource:
    os << "tf_resource";
    return;
  case Type::Kind::TFVariant:
    os << "tf_variant";
    return;
  case Type::Kind::TFComplex64:
    os << "tf_complex64";
    return;
  case Type::Kind::TFComplex128:
    os << "tf_complex128";
    return;
  case Type::Kind::TFF32REF:
    os << "tf_f32ref";
    return;
  case Type::Kind::TFString:
    os << "tf_string";
    return;

  case Type::Kind::Integer: {
    auto integer = type.cast<IntegerType>();
    os << 'i' << integer.getWidth();
    return;
  }
  case Type::Kind::Function: {
    auto func = type.cast<FunctionType>();
    os << '(';
    interleaveComma(func.getInputs(), [&](Type type) { printType(type); });
    os << ") -> ";
    auto results = func.getResults();
    if (results.size() == 1)
      os << results[0];
    else {
      os << '(';
      interleaveComma(results, [&](Type type) { printType(type); });
      os << ')';
    }
    return;
  }
  case Type::Kind::Vector: {
    auto v = type.cast<VectorType>();
    os << "vector<";
    for (auto dim : v.getShape())
      os << dim << 'x';
    os << v.getElementType() << '>';
    return;
  }
  case Type::Kind::RankedTensor: {
    auto v = type.cast<RankedTensorType>();
    os << "tensor<";
    for (auto dim : v.getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    os << v.getElementType() << '>';
    return;
  }
  case Type::Kind::UnrankedTensor: {
    auto v = type.cast<UnrankedTensorType>();
    os << "tensor<*x";
    printType(v.getElementType());
    os << '>';
    return;
  }
  case Type::Kind::MemRef: {
    auto v = type.cast<MemRefType>();
    os << "memref<";
    for (auto dim : v.getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    printType(v.getElementType());
    for (auto map : v.getAffineMaps()) {
      os << ", ";
      printAffineMapReference(map);
    }
    // Only print the memory space if it is the non-default one.
    if (v.getMemorySpace())
      os << ", " << v.getMemorySpace();
    os << '>';
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
// Affine expressions and maps
//===----------------------------------------------------------------------===//

void ModulePrinter::printAffineExpr(AffineExpr expr) {
  printAffineExprInternal(expr, BindingStrength::Weak);
}

void ModulePrinter::printAffineExprInternal(
    AffineExpr expr, BindingStrength enclosingTightness) {
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId:
    os << 's' << expr.cast<AffineSymbolExpr>().getPosition();
    return;
  case AffineExprKind::DimId:
    os << 'd' << expr.cast<AffineDimExpr>().getPosition();
    return;
  case AffineExprKind::Constant:
    os << expr.cast<AffineConstantExpr>().getValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  auto binOp = expr.cast<AffineBinaryOpExpr>();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      os << '(';

    printAffineExprInternal(binOp.getLHS(), BindingStrength::Strong);
    os << binopSpelling;
    printAffineExprInternal(binOp.getRHS(), BindingStrength::Strong);

    if (enclosingTightness == BindingStrength::Strong)
      os << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  AffineExpr rhsExpr = binOp.getRHS();
  if (auto rhs = rhsExpr.dyn_cast<AffineBinaryOpExpr>()) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = rrhsExpr.dyn_cast<AffineConstantExpr>()) {
        if (rrhs.getValue() == -1) {
          printAffineExprInternal(binOp.getLHS(), BindingStrength::Weak);
          os << " - ";
          if (rhs.getLHS().getKind() == AffineExprKind::Add) {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong);
          } else {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Weak);
          }

          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }

        if (rrhs.getValue() < -1) {
          printAffineExprInternal(binOp.getLHS(), BindingStrength::Weak);
          os << " - ";
          printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong);
          os << " * " << -rrhs.getValue();
          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto rhs = rhsExpr.dyn_cast<AffineConstantExpr>()) {
    if (rhs.getValue() < 0) {
      printAffineExprInternal(binOp.getLHS(), BindingStrength::Weak);
      os << " - " << -rhs.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }
  }

  printAffineExprInternal(binOp.getLHS(), BindingStrength::Weak);
  os << " + ";
  printAffineExprInternal(binOp.getRHS(), BindingStrength::Weak);

  if (enclosingTightness == BindingStrength::Strong)
    os << ')';
}

void ModulePrinter::printAffineConstraint(AffineExpr expr, bool isEq) {
  printAffineExprInternal(expr, BindingStrength::Weak);
  isEq ? os << " == 0" : os << " >= 0";
}

void ModulePrinter::printAffineMap(AffineMap map) {
  // Dimension identifiers.
  os << '(';
  for (int i = 0; i < (int)map.getNumDims() - 1; ++i)
    os << 'd' << i << ", ";
  if (map.getNumDims() >= 1)
    os << 'd' << map.getNumDims() - 1;
  os << ')';

  // Symbolic identifiers.
  if (map.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < map.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (map.getNumSymbols() >= 1)
      os << 's' << map.getNumSymbols() - 1;
    os << ']';
  }

  // AffineMap should have at least one result.
  assert(!map.getResults().empty());
  // Result affine expressions.
  os << " -> (";
  interleaveComma(map.getResults(),
                  [&](AffineExpr expr) { printAffineExpr(expr); });
  os << ')';

  if (!map.isBounded()) {
    return;
  }

  // Print range sizes for bounded affine maps.
  os << " size (";
  interleaveComma(map.getRangeSizes(),
                  [&](AffineExpr expr) { printAffineExpr(expr); });
  os << ')';
}

void ModulePrinter::printIntegerSet(IntegerSet set) {
  // Dimension identifiers.
  os << '(';
  for (unsigned i = 1; i < set.getNumDims(); ++i)
    os << 'd' << i - 1 << ", ";
  if (set.getNumDims() >= 1)
    os << 'd' << set.getNumDims() - 1;
  os << ')';

  // Symbolic identifiers.
  if (set.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < set.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (set.getNumSymbols() >= 1)
      os << 's' << set.getNumSymbols() - 1;
    os << ']';
  }

  // Print constraints.
  os << " : (";
  auto numConstraints = set.getNumConstraints();
  for (int i = 1; i < numConstraints; ++i) {
    printAffineConstraint(set.getConstraint(i - 1), set.isEq(i - 1));
    os << ", ";
  }
  if (numConstraints >= 1)
    printAffineConstraint(set.getConstraint(numConstraints - 1),
                          set.isEq(numConstraints - 1));
  os << ')';
}

//===----------------------------------------------------------------------===//
// Function printing
//===----------------------------------------------------------------------===//

void ModulePrinter::printFunctionResultType(FunctionType type) {
  switch (type.getResults().size()) {
  case 0:
    break;
  case 1:
    os << " -> ";
    printType(type.getResults()[0]);
    break;
  default:
    os << " -> (";
    interleaveComma(type.getResults(),
                    [&](Type eltType) { printType(eltType); });
    os << ')';
    break;
  }
}

void ModulePrinter::printFunctionAttributes(const Function *fn) {
  auto attrs = fn->getAttrs();
  if (attrs.empty())
    return;
  os << "\n  attributes ";
  printOptionalAttrDict(attrs);
}

void ModulePrinter::printFunctionSignature(const Function *fn) {
  auto type = fn->getType();

  os << "@" << fn->getName() << '(';
  interleaveComma(type.getInputs(), [&](Type eltType) { printType(eltType); });
  os << ')';

  printFunctionResultType(type);
}

void ModulePrinter::printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                                          ArrayRef<const char *> elidedAttrs) {
  // If there are no attributes, then there is nothing to be done.
  if (attrs.empty())
    return;

  // Filter out any attributes that shouldn't be included.
  SmallVector<NamedAttribute, 8> filteredAttrs;
  for (auto attr : attrs) {
    auto attrName = attr.first.strref();
    // Never print attributes that start with a colon.  These are internal
    // attributes that represent location or other internal metadata.
    if (attrName.startswith(":"))
      return;

    // If the caller has requested that this attribute be ignored, then drop it.
    bool ignore = false;
    for (const char *elide : elidedAttrs)
      ignore |= attrName == StringRef(elide);

    // Otherwise add it to our filteredAttrs list.
    if (!ignore) {
      filteredAttrs.push_back(attr);
    }
  }

  // If there are no attributes left to print after filtering, then we're done.
  if (filteredAttrs.empty())
    return;

  // Otherwise, print them all out in braces.
  os << " {";
  interleaveComma(filteredAttrs, [&](NamedAttribute attr) {
    os << attr.first << ": ";
    printAttribute(attr.second);
  });
  os << '}';
}

void ModulePrinter::printExt(const Function *fn) {
  os << "extfunc ";
  printFunctionSignature(fn);
  printFunctionAttributes(fn);
  os << '\n';
}

namespace {

// FunctionPrinter contains common functionality for printing
// CFG and ML functions.
class FunctionPrinter : public ModulePrinter, private OpAsmPrinter {
public:
  FunctionPrinter(const ModulePrinter &other) : ModulePrinter(other) {}

  void printOperation(const OperationInst *op);
  void printDefaultOp(const OperationInst *op);

  // Implement OpAsmPrinter.
  raw_ostream &getStream() const { return os; }
  void printType(Type type) { ModulePrinter::printType(type); }
  void printAttribute(Attribute attr) { ModulePrinter::printAttribute(attr); }
  void printAffineMap(AffineMap map) {
    return ModulePrinter::printAffineMapReference(map);
  }
  void printIntegerSet(IntegerSet set) {
    return ModulePrinter::printIntegerSetReference(set);
  }
  void printAffineExpr(AffineExpr expr) {
    return ModulePrinter::printAffineExpr(expr);
  }
  void printFunctionReference(const Function *func) {
    return ModulePrinter::printFunctionReference(func);
  }
  void printFunctionAttributes(const Function *func) {
    return ModulePrinter::printFunctionAttributes(func);
  }
  void printOperand(const Value *value) { printValueID(value); }

  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<const char *> elidedAttrs = {}) {
    return ModulePrinter::printOptionalAttrDict(attrs, elidedAttrs);
  };

  enum { nameSentinel = ~0U };

protected:
  void numberValueID(const Value *value) {
    assert(!valueIDs.count(value) && "Value numbered multiple times");

    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);

    // Give constant integers special names.
    if (auto *op = value->getDefiningInst()) {
      if (auto intOp = op->dyn_cast<ConstantIntOp>()) {
        // i1 constants get special names.
        if (intOp->getType().isInteger(1)) {
          specialName << (intOp->getValue() ? "true" : "false");
        } else {
          specialName << 'c' << intOp->getValue() << '_' << intOp->getType();
        }
      } else if (auto intOp = op->dyn_cast<ConstantIndexOp>()) {
        specialName << 'c' << intOp->getValue();
      } else if (auto constant = op->dyn_cast<ConstantOp>()) {
        if (constant->getValue().isa<FunctionAttr>())
          specialName << 'f';
        else
          specialName << "cst";
      }
    }

    if (specialNameBuffer.empty()) {
      switch (value->getKind()) {
      case Value::Kind::BlockArgument:
        // If this is an argument to the function, give it an 'arg' name.
        if (auto *block = cast<BlockArgument>(value)->getOwner())
          if (auto *fn = block->getFunction())
            if (&fn->getBlockList().front() == block) {
              specialName << "arg" << nextArgumentID++;
              break;
            }
        // Otherwise number it normally.
        valueIDs[value] = nextValueID++;
        return;
      case Value::Kind::StmtResult:
        // This is an uninteresting result, give it a boring number and be
        // done with it.
        valueIDs[value] = nextValueID++;
        return;
      case Value::Kind::ForStmt:
        specialName << 'i' << nextLoopID++;
        break;
      }
    }

    // Ok, this value had an interesting name.  Remember it with a sentinel.
    valueIDs[value] = nameSentinel;

    // Remember that we've used this name, checking to see if we had a conflict.
    auto insertRes = usedNames.insert(specialName.str());
    if (insertRes.second) {
      // If this is the first use of the name, then we're successful!
      valueNames[value] = insertRes.first->first();
      return;
    }

    // Otherwise, we had a conflict - probe until we find a unique name.  This
    // is guaranteed to terminate (and usually in a single iteration) because it
    // generates new names by incrementing nextConflictID.
    while (1) {
      std::string probeName =
          specialName.str().str() + "_" + llvm::utostr(nextConflictID++);
      insertRes = usedNames.insert(probeName);
      if (insertRes.second) {
        // If this is the first use of the name, then we're successful!
        valueNames[value] = insertRes.first->first();
        return;
      }
    }
  }

  void printValueID(const Value *value, bool printResultNo = true) const {
    int resultNo = -1;
    auto lookupValue = value;

    // If this is a reference to the result of a multi-result instruction or
    // statement, print out the # identifier and make sure to map our lookup
    // to the first result of the instruction.
    if (auto *result = dyn_cast<InstResult>(value)) {
      if (result->getOwner()->getNumResults() != 1) {
        resultNo = result->getResultNumber();
        lookupValue = result->getOwner()->getResult(0);
      }
    } else if (auto *result = dyn_cast<StmtResult>(value)) {
      if (result->getOwner()->getNumResults() != 1) {
        resultNo = result->getResultNumber();
        lookupValue = result->getOwner()->getResult(0);
      }
    }

    auto it = valueIDs.find(lookupValue);
    if (it == valueIDs.end()) {
      os << "<<INVALID SSA VALUE>>";
      return;
    }

    os << '%';
    if (it->second != nameSentinel) {
      os << it->second;
    } else {
      auto nameIt = valueNames.find(lookupValue);
      assert(nameIt != valueNames.end() && "Didn't have a name entry?");
      os << nameIt->second;
    }

    if (resultNo != -1 && printResultNo)
      os << '#' << resultNo;
  }

private:
  /// This is the value ID for each SSA value in the current function.  If this
  /// returns ~0, then the valueID has an entry in valueNames.
  DenseMap<const Value *, unsigned> valueIDs;
  DenseMap<const Value *, StringRef> valueNames;

  /// This keeps track of all of the non-numeric names that are in flight,
  /// allowing us to check for duplicates.
  llvm::StringSet<> usedNames;

  /// This is the next value ID to assign in numbering.
  unsigned nextValueID = 0;
  /// This is the ID to assign to the next induction variable.
  unsigned nextLoopID = 0;
  /// This is the next ID to assign to an MLFunction argument.
  unsigned nextArgumentID = 0;

  /// This is the next ID to assign when a name conflict is detected.
  unsigned nextConflictID = 0;
};
} // end anonymous namespace

void FunctionPrinter::printOperation(const OperationInst *op) {
  if (op->getNumResults()) {
    printValueID(op->getResult(0), /*printResultNo=*/false);
    os << " = ";
  }

  // Check to see if this is a known operation.  If so, use the registered
  // custom printer hook.
  if (auto *opInfo = op->getAbstractOperation()) {
    opInfo->printAssembly(op, this);
    return;
  }

  // Otherwise use the standard verbose printing approach.
  printDefaultOp(op);
}

void FunctionPrinter::printDefaultOp(const OperationInst *op) {
  os << '"';
  printEscapedString(op->getName().getStringRef(), os);
  os << "\"(";

  interleaveComma(op->getOperands(),
                  [&](const Value *value) { printValueID(value); });

  os << ')';
  auto attrs = op->getAttrs();
  printOptionalAttrDict(attrs);

  // Print the type signature of the operation.
  os << " : (";
  interleaveComma(op->getOperands(),
                  [&](const Value *value) { printType(value->getType()); });
  os << ") -> ";

  if (op->getNumResults() == 1) {
    printType(op->getResult(0)->getType());
  } else {
    os << '(';
    interleaveComma(op->getResults(),
                    [&](const Value *result) { printType(result->getType()); });
    os << ')';
  }
}

//===----------------------------------------------------------------------===//
// CFG Function printing
//===----------------------------------------------------------------------===//

namespace {
class CFGFunctionPrinter : public FunctionPrinter {
public:
  CFGFunctionPrinter(const CFGFunction *function, const ModulePrinter &other);

  const CFGFunction *getFunction() const { return function; }

  void print();
  void print(const BasicBlock *block);

  void print(const Instruction *inst);

  void printSuccessorAndUseList(const OperationInst *term, unsigned index);

  void printBBName(const BasicBlock *block) { os << "bb" << getBBID(block); }

  unsigned getBBID(const BasicBlock *block) {
    auto it = basicBlockIDs.find(block);
    assert(it != basicBlockIDs.end() && "Block not in this function?");
    return it->second;
  }

private:
  const CFGFunction *function;
  DenseMap<const BasicBlock *, unsigned> basicBlockIDs;

  void numberValuesInBlock(const BasicBlock *block);

  template <typename Range> void printBranchOperands(const Range &range);
};
} // end anonymous namespace

CFGFunctionPrinter::CFGFunctionPrinter(const CFGFunction *function,
                                       const ModulePrinter &other)
    : FunctionPrinter(other), function(function) {
  // Each basic block gets a unique ID per function.
  unsigned blockID = 0;
  for (auto &block : *function) {
    basicBlockIDs[&block] = blockID++;
    numberValuesInBlock(&block);
  }
}

/// Number all of the SSA values in the specified basic block.
void CFGFunctionPrinter::numberValuesInBlock(const BasicBlock *block) {
  for (auto *arg : block->getArguments()) {
    numberValueID(arg);
  }
  for (auto &op : *block) {
    // We number instruction that have results, and we only number the first
    // result.
    if (auto *opInst = dyn_cast<OperationInst>(&op))
      if (opInst->getNumResults() != 0)
        numberValueID(opInst->getResult(0));
  }

  // Terminators do not define values.
}

void CFGFunctionPrinter::print() {
  os << "cfgfunc ";
  printFunctionSignature(getFunction());
  printFunctionAttributes(getFunction());
  os << " {\n";

  for (auto &block : *function)
    print(&block);
  os << "}\n\n";
}

void CFGFunctionPrinter::print(const BasicBlock *block) {
  printBBName(block);

  if (!block->args_empty()) {
    os << '(';
    interleaveComma(block->getArguments(), [&](const BBArgument *arg) {
      printValueID(arg);
      os << ": ";
      printType(arg->getType());
    });
    os << ')';
  }
  os << ':';

  // Print out some context information about the predecessors of this block.
  if (!block->getFunction()) {
    os << "\t// block is not in a function!";
  } else if (block->hasNoPredecessors()) {
    // Don't print "no predecessors" for the entry block.
    if (block != &block->getFunction()->front())
      os << "\t// no predecessors";
  } else if (auto *pred = block->getSinglePredecessor()) {
    os << "\t// pred: ";
    printBBName(pred);
  } else {
    // We want to print the predecessors in increasing numeric order, not in
    // whatever order the use-list is in, so gather and sort them.
    SmallVector<unsigned, 4> predIDs;
    for (auto *pred : block->getPredecessors())
      predIDs.push_back(getBBID(pred));
    llvm::array_pod_sort(predIDs.begin(), predIDs.end());

    os << "\t// " << predIDs.size() << " preds: ";

    interleaveComma(predIDs, [&](unsigned predID) { os << "bb" << predID; });
  }
  os << '\n';

  for (auto &inst : block->getStatements()) {
    os << "  ";
    print(&inst);
    os << '\n';
  }
}

void CFGFunctionPrinter::print(const Instruction *inst) {
  if (!inst) {
    os << "<<null instruction>>\n";
    return;
  }
  auto *opInst = dyn_cast<OperationInst>(inst);
  assert(opInst && "IfStmt/ForStmt aren't supported in CFG functions yet");
  printOperation(opInst);
}

// Print the operands from "container" to "os", followed by a colon and their
// respective types, everything in parentheses.  Do nothing if the container is
// empty.
template <typename Range>
void CFGFunctionPrinter::printBranchOperands(const Range &range) {
  if (llvm::empty(range))
    return;

  os << '(';
  interleaveComma(range,
                  [this](const Value *operand) { printValueID(operand); });
  os << " : ";
  interleaveComma(
      range, [this](const Value *operand) { printType(operand->getType()); });
  os << ')';
}

void CFGFunctionPrinter::printSuccessorAndUseList(const OperationInst *term,
                                                  unsigned index) {
  printBBName(term->getSuccessor(index));
  printBranchOperands(term->getSuccessorOperands(index));
}

void ModulePrinter::printCFG(const Function *fn) {
  CFGFunctionPrinter(fn, *this).print();
}

//===----------------------------------------------------------------------===//
// ML Function printing
//===----------------------------------------------------------------------===//

namespace {
class MLFunctionPrinter : public FunctionPrinter {
public:
  MLFunctionPrinter(const MLFunction *function, const ModulePrinter &other);

  const MLFunction *getFunction() const { return function; }

  // Prints ML function.
  void print();

  // Prints ML function signature.
  void printFunctionSignature();

  // Methods to print ML function statements.
  void print(const Statement *stmt);
  void print(const OperationInst *stmt);
  void print(const ForStmt *stmt);
  void print(const IfStmt *stmt);
  void print(const StmtBlock *block);
  void printSuccessorAndUseList(const OperationInst *term, unsigned index) {
    assert(false && "MLFunctions do not have terminators with successors.");
  }

  // Print loop bounds.
  void printDimAndSymbolList(ArrayRef<StmtOperand> ops, unsigned numDims);
  void printBound(AffineBound bound, const char *prefix);

  // Number of spaces used for indenting nested statements.
  const static unsigned indentWidth = 2;

private:
  void numberValues();

  const MLFunction *function;
  int numSpaces;
};
} // end anonymous namespace

MLFunctionPrinter::MLFunctionPrinter(const MLFunction *function,
                                     const ModulePrinter &other)
    : FunctionPrinter(other), function(function), numSpaces(0) {
  assert(function && "Cannot print nullptr function");
  numberValues();
}

/// Number all of the SSA values in this ML function.
void MLFunctionPrinter::numberValues() {
  // Numbers ML function arguments.
  for (auto *arg : function->getArguments())
    numberValueID(arg);

  // Walks ML function statements and numbers for statements and
  // the first result of the operation statements.
  struct NumberValuesPass : public StmtWalker<NumberValuesPass> {
    NumberValuesPass(MLFunctionPrinter *printer) : printer(printer) {}
    void visitOperationInst(OperationInst *stmt) {
      if (stmt->getNumResults() != 0)
        printer->numberValueID(stmt->getResult(0));
    }
    void visitForStmt(ForStmt *stmt) { printer->numberValueID(stmt); }
    MLFunctionPrinter *printer;
  };

  NumberValuesPass pass(this);
  // TODO: it'd be cleaner to have constant visitor instead of using const_cast.
  pass.walk(const_cast<MLFunction *>(function));
}

void MLFunctionPrinter::print() {
  os << "mlfunc ";
  printFunctionSignature();
  printFunctionAttributes(getFunction());
  os << " {\n";
  print(function->getBody());
  os << "}\n\n";
}

void MLFunctionPrinter::printFunctionSignature() {
  auto type = function->getType();

  os << "@" << function->getName() << '(';

  for (unsigned i = 0, e = function->getNumArguments(); i != e; ++i) {
    if (i > 0)
      os << ", ";
    auto *arg = function->getArgument(i);
    printOperand(arg);
    os << " : ";
    printType(arg->getType());
  }
  os << ")";
  printFunctionResultType(type);
}

void MLFunctionPrinter::print(const StmtBlock *block) {
  numSpaces += indentWidth;
  for (auto &stmt : block->getStatements()) {
    print(&stmt);
    os << "\n";
  }
  numSpaces -= indentWidth;
}

void MLFunctionPrinter::print(const Statement *stmt) {
  switch (stmt->getKind()) {
  case Statement::Kind::OperationInst:
    return print(cast<OperationInst>(stmt));
  case Statement::Kind::For:
    return print(cast<ForStmt>(stmt));
  case Statement::Kind::If:
    return print(cast<IfStmt>(stmt));
  }
}

void MLFunctionPrinter::print(const OperationInst *stmt) {
  os.indent(numSpaces);
  printOperation(stmt);
}

void MLFunctionPrinter::print(const ForStmt *stmt) {
  os.indent(numSpaces) << "for ";
  printOperand(stmt);
  os << " = ";
  printBound(stmt->getLowerBound(), "max");
  os << " to ";
  printBound(stmt->getUpperBound(), "min");

  if (stmt->getStep() != 1)
    os << " step " << stmt->getStep();

  os << " {\n";
  print(stmt->getBody());
  os.indent(numSpaces) << "}";
}

void MLFunctionPrinter::printDimAndSymbolList(ArrayRef<StmtOperand> ops,
                                              unsigned numDims) {
  auto printComma = [&]() { os << ", "; };
  os << '(';
  interleave(ops.begin(), ops.begin() + numDims,
             [&](const StmtOperand &v) { printOperand(v.get()); }, printComma);
  os << ')';

  if (numDims < ops.size()) {
    os << '[';
    interleave(ops.begin() + numDims, ops.end(),
               [&](const StmtOperand &v) { printOperand(v.get()); },
               printComma);
    os << ']';
  }
}

void MLFunctionPrinter::printBound(AffineBound bound, const char *prefix) {
  AffineMap map = bound.getMap();

  // Check if this bound should be printed using short-hand notation.
  // The decision to restrict printing short-hand notation to trivial cases
  // comes from the will to roundtrip MLIR binary -> text -> binary in a
  // lossless way.
  // Therefore, short-hand parsing and printing is only supported for
  // zero-operand constant maps and single symbol operand identity maps.
  if (map.getNumResults() == 1) {
    AffineExpr expr = map.getResult(0);

    // Print constant bound.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
      if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        os << constExpr.getValue();
        return;
      }
    }

    // Print bound that consists of a single SSA symbol if the map is over a
    // single symbol.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 1) {
      if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
        printOperand(bound.getOperand(0));
        return;
      }
    }
  } else {
    // Map has multiple results. Print 'min' or 'max' prefix.
    os << prefix << ' ';
  }

  // Print the map and its operands.
  printAffineMapReference(map);
  printDimAndSymbolList(bound.getStmtOperands(), map.getNumDims());
}

void MLFunctionPrinter::print(const IfStmt *stmt) {
  os.indent(numSpaces) << "if ";
  IntegerSet set = stmt->getIntegerSet();
  printIntegerSetReference(set);
  printDimAndSymbolList(stmt->getStmtOperands(), set.getNumDims());
  os << " {\n";
  print(stmt->getThen());
  os.indent(numSpaces) << "}";
  if (stmt->hasElse()) {
    os << " else {\n";
    print(stmt->getElse());
    os.indent(numSpaces) << "}";
  }
}

void ModulePrinter::printML(const Function *fn) {
  MLFunctionPrinter(fn, *this).print();
}

//===----------------------------------------------------------------------===//
// print and dump methods
//===----------------------------------------------------------------------===//

void Attribute::print(raw_ostream &os) const {
  ModuleState state(/*no context is known*/ nullptr);
  ModulePrinter(os, state).printAttribute(*this);
}

void Attribute::dump() const { print(llvm::errs()); }

void Type::print(raw_ostream &os) const {
  ModuleState state(getContext());
  ModulePrinter(os, state).printType(*this);
}

void Type::dump() const { print(llvm::errs()); }

void AffineMap::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void IntegerSet::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineExpr::print(raw_ostream &os) const {
  ModuleState state(getContext());
  ModulePrinter(os, state).printAffineExpr(*this);
}

void AffineExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::print(raw_ostream &os) const {
  ModuleState state(getContext());
  ModulePrinter(os, state).printAffineMap(*this);
}

void IntegerSet::print(raw_ostream &os) const {
  ModuleState state(/*no context is known*/ nullptr);
  ModulePrinter(os, state).printIntegerSet(*this);
}

void Value::print(raw_ostream &os) const {
  switch (getKind()) {
  case Value::Kind::BlockArgument:
    // TODO: Improve this.
    os << "<block argument>\n";
    return;
  case Value::Kind::StmtResult:
    return getDefiningInst()->print(os);
  case Value::Kind::ForStmt:
    return cast<ForStmt>(this)->print(os);
  }
}

void Value::dump() const { print(llvm::errs()); }

void Instruction::print(raw_ostream &os) const {
  auto *function = getFunction();
  if (!function) {
    os << "<<UNLINKED INSTRUCTION>>\n";
    return;
  }
  if (function->isCFG()) {
    ModuleState state(function->getContext());
    ModulePrinter modulePrinter(os, state);
    CFGFunctionPrinter(function, modulePrinter).print(this);
  } else {
    ModuleState state(function->getContext());
    ModulePrinter modulePrinter(os, state);
    MLFunctionPrinter(function, modulePrinter).print(this);
  }
}

void Instruction::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void BasicBlock::print(raw_ostream &os) const {
  auto *function = getFunction();
  if (!function) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }

  if (function->isCFG()) {
    ModuleState state(function->getContext());
    ModulePrinter modulePrinter(os, state);
    CFGFunctionPrinter(function, modulePrinter).print(this);
  } else {
    ModuleState state(function->getContext());
    ModulePrinter modulePrinter(os, state);
    MLFunctionPrinter(function, modulePrinter).print(this);
  }
}

void BasicBlock::dump() const { print(llvm::errs()); }

/// Print out the name of the basic block without printing its body.
void StmtBlock::printAsOperand(raw_ostream &os, bool printType) {
  if (!getFunction()) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  ModuleState state(getFunction()->getContext());
  ModulePrinter modulePrinter(os, state);
  CFGFunctionPrinter(getFunction(), modulePrinter).printBBName(this);
}

void Function::print(raw_ostream &os) const {
  ModuleState state(getContext());
  ModulePrinter(os, state).print(this);
}

void Function::dump() const { print(llvm::errs()); }

void Module::print(raw_ostream &os) const {
  ModuleState state(getContext());
  state.initialize(this);
  ModulePrinter(os, state).print(this);
}

void Module::dump() const { print(llvm::errs()); }

void Location::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::Unknown:
    os << "[unknown-location]";
    break;
  case Kind::FileLineCol: {
    auto fileLoc = cast<FileLineColLoc>();
    os << fileLoc.getFilename() << ':' << fileLoc.getLine() << ':'
       << fileLoc.getColumn();
    break;
  }
  case Kind::Name: {
    auto nameLoc = cast<NameLoc>();
    os << nameLoc.getName();
    break;
  }
  case Kind::CallSite: {
    auto callLocation = cast<CallSiteLoc>();
    auto callee = callLocation.getCallee();
    auto caller = callLocation.getCaller();
    callee.print(os);
    if (caller.isa<CallSiteLoc>()) {
      os << "\n at ";
    }
    caller.print(os);
    break;
  }
  case Kind::FusedLocation: {
    auto fusedLoc = cast<FusedLoc>();
    if (auto metadata = fusedLoc.getMetadata())
      os << '<' << metadata << '>';
    os << '[';
    interleave(
        fusedLoc.getLocations(), [&](Location loc) { loc.print(os); },
        [&]() { os << ", "; });
    os << ']';
    break;
  }
  }
}

void Location::dump() const { print(llvm::errs()); }
