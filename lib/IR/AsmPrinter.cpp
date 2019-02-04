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
#include "mlir/IR/InstVisitor.h"
#include "mlir/IR/Instruction.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
using namespace mlir;

void Identifier::print(raw_ostream &os) const { os << str(); }

void Identifier::dump() const { print(llvm::errs()); }

void OperationName::print(raw_ostream &os) const { os << getStringRef(); }

void OperationName::dump() const { print(llvm::errs()); }

OpAsmPrinter::~OpAsmPrinter() {}

//===----------------------------------------------------------------------===//
// ModuleState
//===----------------------------------------------------------------------===//

// TODO(riverriddle) Rethink this flag when we have a pass that can remove debug
// info or when we have a system for printer flags.
static llvm::cl::opt<bool>
    shouldPrintDebugInfoOpt("mlir-print-debuginfo",
                            llvm::cl::desc("Print debug info in MLIR output"),
                            llvm::cl::init(false));

static llvm::cl::opt<bool> printPrettyDebugInfo(
    "mlir-pretty-debuginfo",
    llvm::cl::desc("Print pretty debug info in MLIR output"),
    llvm::cl::init(false));

namespace {
class ModuleState {
public:
  /// This is the current context if it is knowable, otherwise this is null.
  MLIRContext *const context;

  explicit ModuleState(MLIRContext *context) : context(context) {}

  // Initializes module state, populating affine map state.
  void initialize(const Module *module);

  StringRef getAffineMapAlias(AffineMap affineMap) const {
    return affineMapToAlias.lookup(affineMap);
  }

  int getAffineMapId(AffineMap affineMap) const {
    auto it = affineMapIds.find(affineMap);
    if (it == affineMapIds.end()) {
      return -1;
    }
    return it->second;
  }

  ArrayRef<AffineMap> getAffineMapIds() const { return affineMapsById; }

  StringRef getIntegerSetAlias(IntegerSet integerSet) const {
    return integerSetToAlias.lookup(integerSet);
  }

  int getIntegerSetId(IntegerSet integerSet) const {
    auto it = integerSetIds.find(integerSet);
    if (it == integerSetIds.end()) {
      return -1;
    }
    return it->second;
  }

  ArrayRef<IntegerSet> getIntegerSetIds() const { return integerSetsById; }

  StringRef getTypeAlias(Type ty) const { return typeToAlias.lookup(ty); }

  ArrayRef<Type> getTypeIds() const { return usedTypes.getArrayRef(); }

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

  void recordTypeReference(Type ty) { usedTypes.insert(ty); }

  // Visit functions.
  void visitInstruction(const Instruction *inst);
  void visitType(Type type);
  void visitAttribute(Attribute attr);

  // Initialize symbol aliases.
  void initializeSymbolAliases();

  DenseMap<AffineMap, int> affineMapIds;
  std::vector<AffineMap> affineMapsById;
  DenseMap<AffineMap, StringRef> affineMapToAlias;

  DenseMap<IntegerSet, int> integerSetIds;
  std::vector<IntegerSet> integerSetsById;
  DenseMap<IntegerSet, StringRef> integerSetToAlias;

  llvm::SetVector<Type> usedTypes;
  DenseMap<Type, StringRef> typeToAlias;
};
} // end anonymous namespace

// TODO Support visiting other types/instructions when implemented.
void ModuleState::visitType(Type type) {
  recordTypeReference(type);
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
  } else if (auto vecOrTensor = type.dyn_cast<VectorOrTensorType>()) {
    visitType(vecOrTensor.getElementType());
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

void ModuleState::visitInstruction(const Instruction *inst) {
  // Visit all the types used in the operation.
  for (auto *operand : inst->getOperands())
    visitType(operand->getType());
  for (auto *result : inst->getResults())
    visitType(result->getType());

  // Visit each of the attributes.
  for (auto elt : inst->getAttrs())
    visitAttribute(elt.second);
}

// Utility to generate a function to register a symbol alias.
template <typename SymbolsInModuleSetTy, typename SymbolTy>
static void registerSymbolAlias(StringRef name, SymbolTy sym,
                                SymbolsInModuleSetTy &symbolsInModuleSet,
                                llvm::StringSet<> &usedAliases,
                                DenseMap<SymbolTy, StringRef> &symToAlias) {
  assert(!name.empty() && "expected alias name to be non-empty");
  assert(sym && "expected alias symbol to be non-null");
  // TODO(riverriddle) Assert that the provided alias name can be lexed as
  // an identifier.

  // Check if the symbol is not referenced by the module or the name is
  // already used by another alias.
  if (!symbolsInModuleSet.count(sym) || !usedAliases.insert(name).second)
    return;
  symToAlias.try_emplace(sym, name);
}

void ModuleState::initializeSymbolAliases() {
  // Track the identifiers in use for each symbol so that the same identifier
  // isn't used twice.
  llvm::StringSet<> usedAliases;

  // Get the currently registered dialects.
  auto dialects = context->getRegisteredDialects();

  // Collect the set of aliases from each dialect.
  SmallVector<std::pair<StringRef, AffineMap>, 8> affineMapAliases;
  SmallVector<std::pair<StringRef, IntegerSet>, 8> integerSetAliases;
  SmallVector<std::pair<StringRef, Type>, 16> typeAliases;
  for (auto *dialect : dialects) {
    dialect->getAffineMapAliases(affineMapAliases);
    dialect->getIntegerSetAliases(integerSetAliases);
    dialect->getTypeAliases(typeAliases);
  }

  // Register the affine aliases.
  // Create a regex for the non-alias names of sets and maps, so that an alias
  // is not registered with a conflicting name.
  llvm::Regex reservedAffineNames("(set|map)[0-9]+");

  // AffineMap aliases
  for (auto &affineAliasPair : affineMapAliases) {
    if (!reservedAffineNames.match(affineAliasPair.first))
      registerSymbolAlias(affineAliasPair.first, affineAliasPair.second,
                          affineMapIds, usedAliases, affineMapToAlias);
  }

  // IntegerSet aliases
  for (auto &integerSetAliasPair : integerSetAliases) {
    if (!reservedAffineNames.match(integerSetAliasPair.first))
      registerSymbolAlias(integerSetAliasPair.first, integerSetAliasPair.second,
                          integerSetIds, usedAliases, integerSetToAlias);
  }

  // Clear the set of used identifiers as types can have the same identifiers as
  // affine structures.
  usedAliases.clear();

  for (auto &typeAliasPair : typeAliases)
    registerSymbolAlias(typeAliasPair.first, typeAliasPair.second, usedTypes,
                        usedAliases, typeToAlias);
}

// Initializes module state, populating affine map and integer set state.
void ModuleState::initialize(const Module *module) {
  for (auto &fn : *module) {
    visitType(fn.getType());

    const_cast<Function &>(fn).walk(
        [&](Instruction *op) { ModuleState::visitInstruction(op); });
  }

  // Initialize the symbol aliases.
  initializeSymbolAliases();
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
  void printLocation(Location loc);

  void printAffineMap(AffineMap map);
  void printAffineExpr(AffineExpr expr);
  void printAffineConstraint(AffineExpr expr, bool isEq);
  void printIntegerSet(IntegerSet set);

protected:
  raw_ostream &os;
  ModuleState &state;

  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<const char *> elidedAttrs = {});
  void printAffineMapId(int affineMapId) const;
  void printAffineMapReference(AffineMap affineMap);
  void printAffineMapAlias(StringRef alias) const;
  void printIntegerSetId(int integerSetId) const;
  void printIntegerSetReference(IntegerSet integerSet);
  void printIntegerSetAlias(StringRef alias) const;
  void printTrailingLocation(Location loc);
  void printLocationInternal(Location loc, bool pretty = false);
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

// Prints affine map identifier.
void ModulePrinter::printAffineMapId(int affineMapId) const {
  os << "#map" << affineMapId;
}

void ModulePrinter::printAffineMapAlias(StringRef alias) const {
  os << '#' << alias;
}

void ModulePrinter::printAffineMapReference(AffineMap affineMap) {
  // Check for an affine map alias.
  auto alias = state.getAffineMapAlias(affineMap);
  if (!alias.empty())
    return printAffineMapAlias(alias);

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

void ModulePrinter::printIntegerSetAlias(StringRef alias) const {
  os << '#' << alias;
}

void ModulePrinter::printIntegerSetReference(IntegerSet integerSet) {
  // Check for an integer set alias.
  auto alias = state.getIntegerSetAlias(integerSet);
  if (!alias.empty()) {
    printIntegerSetAlias(alias);
    return;
  }

  int setId;
  if ((setId = state.getIntegerSetId(integerSet)) >= 0) {
    // The set will be printed at top of module; so print reference to its id.
    printIntegerSetId(setId);
  } else {
    // Set not in module state so print inline.
    integerSet.print(os);
  }
}

void ModulePrinter::printTrailingLocation(Location loc) {
  // Check to see if we are printing debug information.
  if (!shouldPrintDebugInfoOpt)
    return;

  os << " ";
  printLocation(loc);
}

void ModulePrinter::printLocationInternal(Location loc, bool pretty) {
  switch (loc.getKind()) {
  case Location::Kind::Unknown:
    if (pretty)
      os << "[unknown]";
    else
      os << "unknown";
    break;
  case Location::Kind::FileLineCol: {
    auto fileLoc = loc.cast<FileLineColLoc>();
    auto mayQuote = pretty ? "" : "\"";
    os << mayQuote << fileLoc.getFilename() << mayQuote << ':'
       << fileLoc.getLine() << ':' << fileLoc.getColumn();
    break;
  }
  case Location::Kind::Name: {
    os << '\"' << loc.cast<NameLoc>().getName() << '\"';
    break;
  }
  case Location::Kind::CallSite: {
    auto callLocation = loc.cast<CallSiteLoc>();
    auto caller = callLocation.getCaller();
    auto callee = callLocation.getCallee();
    if (!pretty)
      os << "callsite(";
    printLocationInternal(callee, pretty);
    if (pretty) {
      if (callee.isa<NameLoc>()) {
        if (caller.isa<FileLineColLoc>()) {
          os << " at ";
        } else {
          os << "\n at ";
        }
      } else {
        os << "\n at ";
      }
    } else {
      os << " at ";
    }
    printLocationInternal(caller, pretty);
    if (!pretty)
      os << ")";
    break;
  }
  case Location::Kind::FusedLocation: {
    auto fusedLoc = loc.cast<FusedLoc>();
    if (!pretty)
      os << "fused";
    if (auto metadata = fusedLoc.getMetadata())
      os << '<' << metadata << '>';
    os << '[';
    interleave(
        fusedLoc.getLocations(),
        [&](Location loc) { printLocationInternal(loc, pretty); },
        [&]() { os << ", "; });
    os << ']';
    break;
  }
  }
}

void ModulePrinter::print(const Module *module) {
  for (const auto &map : state.getAffineMapIds()) {
    StringRef alias = state.getAffineMapAlias(map);
    if (!alias.empty())
      printAffineMapAlias(alias);
    else
      printAffineMapId(state.getAffineMapId(map));
    os << " = ";
    map.print(os);
    os << '\n';
  }
  for (const auto &set : state.getIntegerSetIds()) {
    StringRef alias = state.getIntegerSetAlias(set);
    if (!alias.empty())
      printIntegerSetAlias(alias);
    else
      printIntegerSetId(state.getIntegerSetId(set));
    os << " = ";
    set.print(os);
    os << '\n';
  }
  for (const auto &type : state.getTypeIds()) {
    StringRef alias = state.getTypeAlias(type);
    if (!alias.empty())
      os << '!' << alias << " = type " << type << '\n';
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

void ModulePrinter::printLocation(Location loc) {
  if (printPrettyDebugInfo) {
    printLocationInternal(loc, /*pretty=*/true);
  } else {
    os << "loc(";
    printLocationInternal(loc);
    os << ')';
  }
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
  // Check for an alias for this type.
  StringRef alias = state.getTypeAlias(type);
  if (!alias.empty()) {
    os << '!' << alias;
    return;
  }

  switch (type.getKind()) {
  default: {
    auto &dialect = type.getDialect();
    os << '!' << dialect.getNamespace() << "<\"";
    assert(dialect.typePrintHook && "Expected dialect type printing hook.");
    dialect.typePrintHook(type, os);
    os << "\">";
    return;
  }
  case Type::Kind::Unknown: {
    auto unknownTy = type.cast<UnknownType>();
    os << '!' << unknownTy.getDialectNamespace() << "<\""
       << unknownTy.getTypeData() << "\">";
    return;
  }
  case Type::Kind::Index:
    os << "index";
    return;
  case StandardTypes::BF16:
    os << "bf16";
    return;
  case StandardTypes::F16:
    os << "f16";
    return;
  case StandardTypes::F32:
    os << "f32";
    return;
  case StandardTypes::F64:
    os << "f64";
    return;

  case StandardTypes::Integer: {
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
  case StandardTypes::Vector: {
    auto v = type.cast<VectorType>();
    os << "vector<";
    for (auto dim : v.getShape())
      os << dim << 'x';
    os << v.getElementType() << '>';
    return;
  }
  case StandardTypes::RankedTensor: {
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
  case StandardTypes::UnrankedTensor: {
    auto v = type.cast<UnrankedTensorType>();
    os << "tensor<*x";
    printType(v.getElementType());
    os << '>';
    return;
  }
  case StandardTypes::MemRef: {
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
  AffineExpr lhsExpr = binOp.getLHS();
  AffineExpr rhsExpr = binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      os << '(';

    // Pretty print multiplication with -1.
    auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>();
    if (rhsConst && rhsConst.getValue() == -1) {
      os << "-";
      printAffineExprInternal(lhsExpr, BindingStrength::Strong);
      return;
    }

    printAffineExprInternal(lhsExpr, BindingStrength::Strong);
    os << binopSpelling;
    printAffineExprInternal(rhsExpr, BindingStrength::Strong);

    if (enclosingTightness == BindingStrength::Strong)
      os << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto rhs = rhsExpr.dyn_cast<AffineBinaryOpExpr>()) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = rrhsExpr.dyn_cast<AffineConstantExpr>()) {
        if (rrhs.getValue() == -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak);
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
          printAffineExprInternal(lhsExpr, BindingStrength::Weak);
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
  if (auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>()) {
    if (rhsConst.getValue() < 0) {
      printAffineExprInternal(lhsExpr, BindingStrength::Weak);
      os << " - " << -rhsConst.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }
  }

  printAffineExprInternal(lhsExpr, BindingStrength::Weak);
  os << " + ";
  printAffineExprInternal(rhsExpr, BindingStrength::Weak);

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

namespace {

// FunctionPrinter contains common functionality for printing
// CFG and ML functions.
class FunctionPrinter : public ModulePrinter, private OpAsmPrinter {
public:
  FunctionPrinter(const Function *function, const ModulePrinter &other);

  // Prints the function as a whole.
  void print();

  // Print the function signature.
  void printFunctionSignature();

  // Methods to print instructions.
  void print(const Instruction *inst);
  void print(const Block *block, bool printBlockArgs = true);

  void printOperation(const Instruction *op);
  void printGenericOp(const Instruction *op);

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
  void printOperand(const Value *value) { printValueID(value); }

  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<const char *> elidedAttrs = {}) {
    return ModulePrinter::printOptionalAttrDict(attrs, elidedAttrs);
  };

  enum { nameSentinel = ~0U };

  void printBlockName(const Block *block) {
    auto id = getBlockID(block);
    if (id != ~0U)
      os << "^bb" << id;
    else
      os << "^INVALIDBLOCK";
  }

  unsigned getBlockID(const Block *block) {
    auto it = blockIDs.find(block);
    return it != blockIDs.end() ? it->second : ~0U;
  }

  void printSuccessorAndUseList(const Instruction *term,
                                unsigned index) override;

  /// Print a block list.
  void printBlockList(const BlockList &blocks,
                      bool printEntryBlockArgs) override {
    os << " {\n";
    if (!blocks.empty()) {
      auto *entryBlock = &blocks.front();
      print(entryBlock,
            printEntryBlockArgs && entryBlock->getNumArguments() != 0);
      for (auto &b : llvm::drop_begin(blocks.getBlocks(), 1))
        print(&b);
    }
    os.indent(currentIndent) << "}";
  }

  // Number of spaces used for indenting nested instructions.
  const static unsigned indentWidth = 2;

protected:
  void numberValueID(const Value *value);
  void numberValuesInBlock(const Block &block);
  void printValueID(const Value *value, bool printResultNo = true) const;

private:
  const Function *function;

  /// This is the value ID for each SSA value in the current function.  If this
  /// returns ~0, then the valueID has an entry in valueNames.
  DenseMap<const Value *, unsigned> valueIDs;
  DenseMap<const Value *, StringRef> valueNames;

  /// This is the block ID for each  block in the current function.
  DenseMap<const Block *, unsigned> blockIDs;

  /// This keeps track of all of the non-numeric names that are in flight,
  /// allowing us to check for duplicates.
  llvm::StringSet<> usedNames;

  // This is the current indentation level for nested structures.
  unsigned currentIndent = 0;

  /// This is the next value ID to assign in numbering.
  unsigned nextValueID = 0;
  /// This is the ID to assign to the next region entry block argument.
  unsigned nextRegionArgumentID = 0;
  /// This is the next ID to assign to a Function argument.
  unsigned nextArgumentID = 0;
  /// This is the next ID to assign when a name conflict is detected.
  unsigned nextConflictID = 0;
  /// This is the next block ID to assign in numbering.
  unsigned nextBlockID = 0;
};
} // end anonymous namespace

FunctionPrinter::FunctionPrinter(const Function *function,
                                 const ModulePrinter &other)
    : ModulePrinter(other), function(function) {

  for (auto &block : *function)
    numberValuesInBlock(block);
}

/// Number all of the SSA values in the specified block list.
void FunctionPrinter::numberValuesInBlock(const Block &block) {
  // Each block gets a unique ID, and all of the instructions within it get
  // numbered as well.
  blockIDs[&block] = nextBlockID++;

  for (auto *arg : block.getArguments())
    numberValueID(arg);

  for (auto &inst : block) {
    // We number instruction that have results, and we only number the first
    // result.
    if (inst.getNumResults() != 0)
      numberValueID(inst.getResult(0));
    for (auto &blockList : inst.getBlockLists())
      for (const auto &block : blockList)
        numberValuesInBlock(block);
  }
}

void FunctionPrinter::numberValueID(const Value *value) {
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
      // If this is an argument to the function, give it an 'arg' name. If the
      // argument is to an entry block of an operation region, give it an 'i'
      // name.
      if (auto *block = cast<BlockArgument>(value)->getOwner()) {
        auto *parentBlockList = block->getParent();
        if (parentBlockList && block == &parentBlockList->front()) {
          if (parentBlockList->getContainingFunction())
            specialName << "arg" << nextArgumentID++;
          else
            specialName << "i" << nextRegionArgumentID++;
          break;
        }
      }
      // Otherwise number it normally.
      valueIDs[value] = nextValueID++;
      return;
    case Value::Kind::InstResult:
      // This is an uninteresting result, give it a boring number and be
      // done with it.
      valueIDs[value] = nextValueID++;
      return;
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

void FunctionPrinter::print() {
  printFunctionSignature();

  // Print out function attributes, if present.
  auto attrs = function->getAttrs();
  if (!attrs.empty()) {
    os << "\n  attributes ";
    printOptionalAttrDict(attrs);
  }

  // Print the trailing location.
  printTrailingLocation(function->getLoc());

  if (!function->empty()) {
    printBlockList(function->getBlockList(), /*printEntryBlockArgs=*/false);
    os << "\n";
  }
  os << '\n';
}

void FunctionPrinter::printFunctionSignature() {
  os << "func @" << function->getName() << '(';

  auto fnType = function->getType();

  // If this is an external function, don't print argument labels.
  if (function->empty()) {
    interleaveComma(fnType.getInputs(),
                    [&](Type eltType) { printType(eltType); });
  } else {
    for (unsigned i = 0, e = function->getNumArguments(); i != e; ++i) {
      if (i > 0)
        os << ", ";
      auto *arg = function->getArgument(i);
      printOperand(arg);
      os << ": ";
      printType(arg->getType());
    }
  }
  os << ')';

  switch (fnType.getResults().size()) {
  case 0:
    break;
  case 1:
    os << " -> ";
    printType(fnType.getResults()[0]);
    break;
  default:
    os << " -> (";
    interleaveComma(fnType.getResults(),
                    [&](Type eltType) { printType(eltType); });
    os << ')';
    break;
  }
}

void FunctionPrinter::print(const Block *block, bool printBlockArgs) {
  // Print the block label and argument list if requested.
  if (printBlockArgs) {
    os.indent(currentIndent);
    printBlockName(block);

    // Print the argument list if non-empty.
    if (!block->args_empty()) {
      os << '(';
      interleaveComma(block->getArguments(), [&](const BlockArgument *arg) {
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
      os << "\t// no predecessors";
    } else if (auto *pred = block->getSinglePredecessor()) {
      os << "\t// pred: ";
      printBlockName(pred);
    } else {
      // We want to print the predecessors in increasing numeric order, not in
      // whatever order the use-list is in, so gather and sort them.
      SmallVector<std::pair<unsigned, const Block *>, 4> predIDs;
      for (auto *pred : block->getPredecessors())
        predIDs.push_back({getBlockID(pred), pred});
      llvm::array_pod_sort(predIDs.begin(), predIDs.end());

      os << "\t// " << predIDs.size() << " preds: ";

      interleaveComma(predIDs, [&](std::pair<unsigned, const Block *> pred) {
        printBlockName(pred.second);
      });
    }
    os << '\n';
  }

  currentIndent += indentWidth;

  for (auto &inst : block->getInstructions()) {
    print(&inst);
    os << '\n';
  }
  currentIndent -= indentWidth;
}

void FunctionPrinter::print(const Instruction *inst) {
  os.indent(currentIndent);
  printOperation(inst);
  printTrailingLocation(inst->getLoc());
}

void FunctionPrinter::printValueID(const Value *value,
                                   bool printResultNo) const {
  int resultNo = -1;
  auto lookupValue = value;

  // If this is a reference to the result of a multi-result instruction or
  // instruction, print out the # identifier and make sure to map our lookup
  // to the first result of the instruction.
  if (auto *result = dyn_cast<InstResult>(value)) {
    if (result->getOwner()->getNumResults() != 1) {
      resultNo = result->getResultNumber();
      lookupValue = result->getOwner()->getResult(0);
    }
  } else if (auto *result = dyn_cast<InstResult>(value)) {
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

void FunctionPrinter::printOperation(const Instruction *op) {
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

  // Otherwise print with the generic assembly form.
  printGenericOp(op);
}

void FunctionPrinter::printGenericOp(const Instruction *op) {
  os << '"';
  printEscapedString(op->getName().getStringRef(), os);
  os << "\"(";

  // Get the list of operands that are not successor operands.
  unsigned totalNumSuccessorOperands = 0;
  unsigned numSuccessors = op->getNumSuccessors();
  for (unsigned i = 0; i < numSuccessors; ++i)
    totalNumSuccessorOperands += op->getNumSuccessorOperands(i);
  unsigned numProperOperands = op->getNumOperands() - totalNumSuccessorOperands;
  SmallVector<const Value *, 8> properOperands(
      op->operand_begin(), std::next(op->operand_begin(), numProperOperands));

  interleaveComma(properOperands,
                  [&](const Value *value) { printValueID(value); });

  os << ')';

  // For terminators, print the list of successors and their operands.
  if (op->isTerminator() && numSuccessors > 0) {
    os << '[';
    for (unsigned i = 0; i < numSuccessors; ++i) {
      if (i != 0)
        os << ", ";
      printSuccessorAndUseList(op, i);
    }
    os << ']';
  }

  auto attrs = op->getAttrs();
  printOptionalAttrDict(attrs);

  // Print the type signature of the operation.
  os << " : (";
  interleaveComma(properOperands,
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

  // Print any trailing block lists.
  for (auto &blockList : op->getBlockLists())
    printBlockList(blockList, /*printEntryBlockArgs=*/true);
}

void FunctionPrinter::printSuccessorAndUseList(const Instruction *term,
                                               unsigned index) {
  printBlockName(term->getSuccessor(index));

  auto succOperands = term->getSuccessorOperands(index);
  if (succOperands.begin() == succOperands.end())
    return;

  os << '(';
  interleaveComma(succOperands,
                  [this](const Value *operand) { printValueID(operand); });
  os << " : ";
  interleaveComma(succOperands, [this](const Value *operand) {
    printType(operand->getType());
  });
  os << ')';
}

// Prints function with initialized module state.
void ModulePrinter::print(const Function *fn) {
  FunctionPrinter(fn, *this).print();
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
  if (expr == nullptr) {
    os << "null affine expr";
    return;
  }
  ModuleState state(getContext());
  ModulePrinter(os, state).printAffineExpr(*this);
}

void AffineExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::print(raw_ostream &os) const {
  if (map == nullptr) {
    os << "null affine map";
    return;
  }
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
  case Value::Kind::InstResult:
    return getDefiningInst()->print(os);
  }
}

void Value::dump() const { print(llvm::errs()); }

void Instruction::print(raw_ostream &os) const {
  auto *function = getFunction();
  if (!function) {
    os << "<<UNLINKED INSTRUCTION>>\n";
    return;
  }

  ModuleState state(function->getContext());
  ModulePrinter modulePrinter(os, state);
  FunctionPrinter(function, modulePrinter).print(this);
}

void Instruction::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void Block::print(raw_ostream &os) const {
  auto *function = getFunction();
  if (!function) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }

  ModuleState state(function->getContext());
  ModulePrinter modulePrinter(os, state);
  FunctionPrinter(function, modulePrinter).print(this);
}

void Block::dump() const { print(llvm::errs()); }

/// Print out the name of the block without printing its body.
void Block::printAsOperand(raw_ostream &os, bool printType) {
  if (!getFunction()) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  ModuleState state(getFunction()->getContext());
  ModulePrinter modulePrinter(os, state);
  FunctionPrinter(getFunction(), modulePrinter).printBlockName(this);
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
  ModuleState state(nullptr);
  ModulePrinter(os, state).printLocation(*this);
}

void Location::dump() const { print(llvm::errs()); }
