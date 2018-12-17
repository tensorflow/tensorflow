//===- MLIRContext.cpp - MLIR Type Classes --------------------------------===//
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

#include "mlir/IR/MLIRContext.h"
#include "AffineExprDetail.h"
#include "AffineMapDetail.h"
#include "AttributeDetail.h"
#include "AttributeListStorage.h"
#include "IntegerSetDetail.h"
#include "LocationDetail.h"
#include "TypeDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::detail;
using namespace llvm;

namespace {
struct FunctionTypeKeyInfo : DenseMapInfo<FunctionTypeStorage *> {
  // Functions are uniqued based on their inputs and results.
  using KeyTy = std::pair<ArrayRef<Type>, ArrayRef<Type>>;
  using DenseMapInfo<FunctionTypeStorage *>::isEqual;

  static unsigned getHashValue(FunctionTypeStorage *key) {
    return getHashValue(KeyTy(key->getInputs(), key->getResults()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        hash_combine_range(key.first.begin(), key.first.end()),
        hash_combine_range(key.second.begin(), key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const FunctionTypeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(rhs->getInputs(), rhs->getResults());
  }
};

struct AffineMapKeyInfo : DenseMapInfo<AffineMap> {
  // Affine maps are uniqued based on their dim/symbol counts and affine
  // expressions.
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>,
                           ArrayRef<AffineExpr>>;
  using DenseMapInfo<AffineMap>::isEqual;

  static unsigned getHashValue(const AffineMap &key) {
    return getHashValue(KeyTy(key.getNumDims(), key.getNumSymbols(),
                              key.getResults(), key.getRangeSizes()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        std::get<0>(key), std::get<1>(key),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()),
        hash_combine_range(std::get<3>(key).begin(), std::get<3>(key).end()));
  }

  static bool isEqual(const KeyTy &lhs, AffineMap rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs.getNumDims(), rhs.getNumSymbols(),
                                  rhs.getResults(), rhs.getRangeSizes());
  }
};

struct IntegerSetKeyInfo : DenseMapInfo<IntegerSet> {
  // Integer sets are uniqued based on their dim/symbol counts, affine
  // expressions appearing in the LHS of constraints, and eqFlags.
  using KeyTy =
      std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>, ArrayRef<bool>>;
  using DenseMapInfo<IntegerSet>::isEqual;

  static unsigned getHashValue(const IntegerSet &key) {
    return getHashValue(KeyTy(key.getNumDims(), key.getNumSymbols(),
                              key.getConstraints(), key.getEqFlags()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        std::get<0>(key), std::get<1>(key),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()),
        hash_combine_range(std::get<3>(key).begin(), std::get<3>(key).end()));
  }

  static bool isEqual(const KeyTy &lhs, IntegerSet rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs.getNumDims(), rhs.getNumSymbols(),
                                  rhs.getConstraints(), rhs.getEqFlags());
  }
};

struct VectorTypeKeyInfo : DenseMapInfo<VectorTypeStorage *> {
  // Vectors are uniqued based on their element type and shape.
  using KeyTy = std::pair<Type, ArrayRef<int>>;
  using DenseMapInfo<VectorTypeStorage *>::isEqual;

  static unsigned getHashValue(VectorTypeStorage *key) {
    return getHashValue(KeyTy(key->elementType, key->getShape()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        DenseMapInfo<Type>::getHashValue(key.first),
        hash_combine_range(key.second.begin(), key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const VectorTypeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(rhs->elementType, rhs->getShape());
  }
};

struct RankedTensorTypeKeyInfo : DenseMapInfo<RankedTensorTypeStorage *> {
  // Ranked tensors are uniqued based on their element type and shape.
  using KeyTy = std::pair<Type, ArrayRef<int>>;
  using DenseMapInfo<RankedTensorTypeStorage *>::isEqual;

  static unsigned getHashValue(RankedTensorTypeStorage *key) {
    return getHashValue(KeyTy(key->elementType, key->getShape()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        DenseMapInfo<Type>::getHashValue(key.first),
        hash_combine_range(key.second.begin(), key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const RankedTensorTypeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(rhs->elementType, rhs->getShape());
  }
};

struct MemRefTypeKeyInfo : DenseMapInfo<MemRefTypeStorage *> {
  // MemRefs are uniqued based on their element type, shape, affine map
  // composition, and memory space.
  using KeyTy = std::tuple<Type, ArrayRef<int>, ArrayRef<AffineMap>, unsigned>;
  using DenseMapInfo<MemRefTypeStorage *>::isEqual;

  static unsigned getHashValue(MemRefTypeStorage *key) {
    return getHashValue(KeyTy(key->elementType, key->getShape(),
                              key->getAffineMaps(), key->memorySpace));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        DenseMapInfo<Type>::getHashValue(std::get<0>(key)),
        hash_combine_range(std::get<1>(key).begin(), std::get<1>(key).end()),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()),
        std::get<3>(key));
  }

  static bool isEqual(const KeyTy &lhs, const MemRefTypeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs->elementType, rhs->getShape(),
                                  rhs->getAffineMaps(), rhs->memorySpace);
  }
};

struct FloatAttrKeyInfo : DenseMapInfo<FloatAttributeStorage *> {
  // Float attributes are uniqued based on wrapped APFloat.
  using KeyTy = std::pair<Type, APFloat>;
  using DenseMapInfo<FloatAttributeStorage *>::isEqual;

  static unsigned getHashValue(FloatAttributeStorage *key) {
    return getHashValue(KeyTy(key->type, key->getValue()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(key.first, llvm::hash_value(key.second));
  }

  static bool isEqual(const KeyTy &lhs, const FloatAttributeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs.first == rhs->type && lhs.second.bitwiseIsEqual(rhs->getValue());
  }
};

struct IntegerAttrKeyInfo : DenseMapInfo<IntegerAttributeStorage *> {
  // Integer attributes are uniqued based on wrapped APInt.
  using KeyTy = std::pair<Type, APInt>;
  using DenseMapInfo<IntegerAttributeStorage *>::isEqual;

  static unsigned getHashValue(IntegerAttributeStorage *key) {
    return getHashValue(KeyTy(key->type, key->getValue()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(key.first, llvm::hash_value(key.second));
  }

  static bool isEqual(const KeyTy &lhs, const IntegerAttributeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    assert(lhs.first.isIndex() ||
           (lhs.first.getBitWidth() == lhs.second.getBitWidth()));
    return lhs.first == rhs->type && lhs.second == rhs->getValue();
  }
};

struct ArrayAttrKeyInfo : DenseMapInfo<ArrayAttributeStorage *> {
  // Array attributes are uniqued based on their elements.
  using KeyTy = ArrayRef<Attribute>;
  using DenseMapInfo<ArrayAttributeStorage *>::isEqual;

  static unsigned getHashValue(ArrayAttributeStorage *key) {
    return getHashValue(KeyTy(key->value));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine_range(key.begin(), key.end());
  }

  static bool isEqual(const KeyTy &lhs, const ArrayAttributeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == rhs->value;
  }
};

struct AttributeListKeyInfo : DenseMapInfo<AttributeListStorage *> {
  // Array attributes are uniqued based on their elements.
  using KeyTy = ArrayRef<NamedAttribute>;
  using DenseMapInfo<AttributeListStorage *>::isEqual;

  static unsigned getHashValue(AttributeListStorage *key) {
    return getHashValue(KeyTy(key->getElements()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine_range(key.begin(), key.end());
  }

  static bool isEqual(const KeyTy &lhs, const AttributeListStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == rhs->getElements();
  }
};

struct DenseElementsAttrInfo : DenseMapInfo<DenseElementsAttributeStorage *> {
  using KeyTy = std::pair<VectorOrTensorType, ArrayRef<char>>;
  using DenseMapInfo<DenseElementsAttributeStorage *>::isEqual;

  static unsigned getHashValue(DenseElementsAttributeStorage *key) {
    return getHashValue(KeyTy(key->type, key->data));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        key.first, hash_combine_range(key.second.begin(), key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs,
                      const DenseElementsAttributeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_pair(rhs->type, rhs->data);
  }
};

struct OpaqueElementsAttrInfo : DenseMapInfo<OpaqueElementsAttributeStorage *> {
  using KeyTy = std::pair<VectorOrTensorType, StringRef>;
  using DenseMapInfo<OpaqueElementsAttributeStorage *>::isEqual;

  static unsigned getHashValue(OpaqueElementsAttributeStorage *key) {
    return getHashValue(KeyTy(key->type, key->bytes));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        key.first, hash_combine_range(key.second.begin(), key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs,
                      const OpaqueElementsAttributeStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_pair(rhs->type, rhs->bytes);
  }
};

struct FusedLocKeyInfo : DenseMapInfo<FusedLocationStorage *> {
  // Fused locations are uniqued based on their held locations and an optional
  // metadata attribute.
  using KeyTy = std::pair<ArrayRef<Location>, Attribute>;
  using DenseMapInfo<FusedLocationStorage *>::isEqual;

  static unsigned getHashValue(FusedLocationStorage *key) {
    return getHashValue(KeyTy(key->getLocations(), key->metadata));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(hash_combine_range(key.first.begin(), key.first.end()),
                        key.second);
  }

  static bool isEqual(const KeyTy &lhs, const FusedLocationStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_pair(rhs->getLocations(), rhs->metadata);
  }
};
} // end anonymous namespace.

namespace mlir {
/// This is the implementation of the MLIRContext class, using the pImpl idiom.
/// This class is completely private to this file, so everything is public.
class MLIRContextImpl {
public:
  /// We put location info into this allocator, since it is generally not
  /// touched by compiler passes.
  llvm::BumpPtrAllocator locationAllocator;

  /// The singleton for UnknownLoc.
  UnknownLocationStorage *theUnknownLoc = nullptr;

  /// These are filename locations uniqued into this MLIRContext.
  llvm::StringMap<char, llvm::BumpPtrAllocator &> filenames;

  /// FileLineColLoc uniquing.
  DenseMap<std::tuple<const char *, unsigned, unsigned>,
           FileLineColLocationStorage *>
      fileLineColLocs;

  /// FusedLoc uniquing.
  using FusedLocations = DenseSet<FusedLocationStorage *, FusedLocKeyInfo>;
  FusedLocations fusedLocs;

  /// We put immortal objects into this allocator.
  llvm::BumpPtrAllocator allocator;

  /// This is the handler to use to report diagnostics, or null if not
  /// registered.
  MLIRContext::DiagnosticHandlerTy diagnosticHandler;

  /// This is a list of dialects that are created referring to this context.
  /// The MLIRContext owns the objects.
  std::vector<std::unique_ptr<Dialect>> dialects;

  /// This is a mapping from operation name to AbstractOperation for registered
  /// operations.
  StringMap<AbstractOperation> registeredOperations;

  /// These are identifiers uniqued into this MLIRContext.
  llvm::StringMap<char, llvm::BumpPtrAllocator &> identifiers;

  // Uniquing table for 'other' types.
  OtherTypeStorage *otherTypes[int(Type::Kind::LAST_OTHER_TYPE) -
                               int(Type::Kind::FIRST_OTHER_TYPE) + 1] = {
      nullptr};

  // Uniquing table for 'float' types.
  FloatTypeStorage *floatTypes[int(Type::Kind::LAST_FLOATING_POINT_TYPE) -
                               int(Type::Kind::FIRST_FLOATING_POINT_TYPE) + 1] =
      {nullptr};

  // Affine map uniquing.
  using AffineMapSet = DenseSet<AffineMap, AffineMapKeyInfo>;
  AffineMapSet affineMaps;

  // Integer set uniquing.
  using IntegerSets = DenseSet<IntegerSet, IntegerSetKeyInfo>;
  IntegerSets integerSets;

  // Affine binary op expression uniquing. Figure out uniquing of dimensional
  // or symbolic identifiers.
  DenseMap<std::tuple<unsigned, AffineExpr, AffineExpr>, AffineExpr>
      affineExprs;

  // Uniqui'ing of AffineDimExpr, AffineSymbolExpr's by their position.
  std::vector<AffineDimExprStorage *> dimExprs;
  std::vector<AffineSymbolExprStorage *> symbolExprs;

  // Uniqui'ing of AffineConstantExprStorage using constant value as key.
  DenseMap<int64_t, AffineConstantExprStorage *> constExprs;

  /// Unique index type (lazily constructed).
  IndexTypeStorage *indexType = nullptr;

  /// Integer type uniquing.
  DenseMap<unsigned, IntegerTypeStorage *> integers;

  /// Function type uniquing.
  using FunctionTypeSet = DenseSet<FunctionTypeStorage *, FunctionTypeKeyInfo>;
  FunctionTypeSet functions;

  /// Vector type uniquing.
  using VectorTypeSet = DenseSet<VectorTypeStorage *, VectorTypeKeyInfo>;
  VectorTypeSet vectors;

  /// Ranked tensor type uniquing.
  using RankedTensorTypeSet =
      DenseSet<RankedTensorTypeStorage *, RankedTensorTypeKeyInfo>;
  RankedTensorTypeSet rankedTensors;

  /// Unranked tensor type uniquing.
  DenseMap<Type, UnrankedTensorTypeStorage *> unrankedTensors;

  /// MemRef type uniquing.
  using MemRefTypeSet = DenseSet<MemRefTypeStorage *, MemRefTypeKeyInfo>;
  MemRefTypeSet memrefs;

  // Attribute uniquing.
  BoolAttributeStorage *boolAttrs[2] = {nullptr};
  DenseSet<IntegerAttributeStorage *, IntegerAttrKeyInfo> integerAttrs;
  DenseSet<FloatAttributeStorage *, FloatAttrKeyInfo> floatAttrs;
  StringMap<StringAttributeStorage *> stringAttrs;
  using ArrayAttrSet = DenseSet<ArrayAttributeStorage *, ArrayAttrKeyInfo>;
  ArrayAttrSet arrayAttrs;
  DenseMap<AffineMap, AffineMapAttributeStorage *> affineMapAttrs;
  DenseMap<IntegerSet, IntegerSetAttributeStorage *> integerSetAttrs;
  DenseMap<Type, TypeAttributeStorage *> typeAttrs;
  using AttributeListSet =
      DenseSet<AttributeListStorage *, AttributeListKeyInfo>;
  AttributeListSet attributeLists;
  DenseMap<const Function *, FunctionAttributeStorage *> functionAttrs;
  DenseMap<std::pair<Type, Attribute>, SplatElementsAttributeStorage *>
      splatElementsAttrs;
  using DenseElementsAttrSet =
      DenseSet<DenseElementsAttributeStorage *, DenseElementsAttrInfo>;
  DenseElementsAttrSet denseElementsAttrs;
  using OpaqueElementsAttrSet =
      DenseSet<OpaqueElementsAttributeStorage *, OpaqueElementsAttrInfo>;
  OpaqueElementsAttrSet opaqueElementsAttrs;
  DenseMap<std::tuple<Type, Attribute, Attribute>,
           SparseElementsAttributeStorage *>
      sparseElementsAttrs;

public:
  MLIRContextImpl() : filenames(locationAllocator), identifiers(allocator) {}

  /// Copy the specified array of elements into memory managed by our bump
  /// pointer allocator.  This assumes the elements are all PODs.
  template <typename T> ArrayRef<T> copyInto(ArrayRef<T> elements) {
    auto result = allocator.Allocate<T>(elements.size());
    std::uninitialized_copy(elements.begin(), elements.end(), result);
    return ArrayRef<T>(result, elements.size());
  }
};
} // end namespace mlir

MLIRContext::MLIRContext() : impl(new MLIRContextImpl()) {
  new BuiltinDialect(this);
  registerAllDialects(this);
}

MLIRContext::~MLIRContext() {}

//===----------------------------------------------------------------------===//
// Diagnostic Handlers
//===----------------------------------------------------------------------===//

/// Register an issue handler with this MLIR context.  The issue handler is
/// passed location information along with a message and a DiagnosticKind enum
/// value that indicates the type of the diagnostic (e.g., Warning, Error).
void MLIRContext::registerDiagnosticHandler(
    const DiagnosticHandlerTy &handler) {
  getImpl().diagnosticHandler = handler;
}

/// Return the current diagnostic handler, or null if none is present.
auto MLIRContext::getDiagnosticHandler() const -> DiagnosticHandlerTy {
  return getImpl().diagnosticHandler;
}

/// This emits a diagnostic using the registered issue handle if present, or
/// with the default behavior if not.  The MLIR compiler should not generally
/// interact with this, it should use methods on Operation instead.
void MLIRContext::emitDiagnostic(Location location, const llvm::Twine &message,
                                 DiagnosticKind kind) const {
  // Check to see if we are emitting a diagnostic on a fused location.
  if (auto fusedLoc = location.dyn_cast<FusedLoc>()) {
    auto fusedLocs = fusedLoc->getLocations();

    // Emit the original diagnostic with the first location in the fused list.
    emitDiagnostic(fusedLocs.front(), message, kind);

    // Emit the rest of the locations as notes.
    for (unsigned i = 1, e = fusedLocs.size(); i != e; ++i)
      emitDiagnostic(fusedLocs[i], "fused from here", DiagnosticKind::Note);
    return;
  }

  // If we had a handler registered, emit the diagnostic using it.
  auto handler = getImpl().diagnosticHandler;
  if (handler)
    return handler(location, message.str(), kind);

  // The default behavior for notes and warnings is to ignore them.
  if (kind != DiagnosticKind::Error)
    return;

  auto &os = llvm::errs();

  if (!location.isa<UnknownLoc>())
    os << location << ": ";

  os << "error: ";

  // The default behavior for errors is to emit them to stderr.
  os << message.str() << '\n';
  os.flush();
}

bool MLIRContext::emitError(Location location,
                            const llvm::Twine &message) const {
  emitDiagnostic(location, message, DiagnosticKind::Error);
  return true;
}

//===----------------------------------------------------------------------===//
// Dialect and Operation Registration
//===----------------------------------------------------------------------===//

/// Return information about all registered IR dialects.
std::vector<Dialect *> MLIRContext::getRegisteredDialects() const {
  std::vector<Dialect *> result;
  result.reserve(getImpl().dialects.size());
  for (auto &dialect : getImpl().dialects)
    result.push_back(dialect.get());
  return result;
}

/// Get registered IR dialect which has the longest matching with the given
/// prefix. If none is found, returns nullptr.
Dialect *MLIRContext::getRegisteredDialect(StringRef prefix) const {
  Dialect *result = nullptr;
  for (auto &dialect : getImpl().dialects) {
    if (prefix.startswith(dialect->getOperationPrefix()))
      if (!result || result->getOperationPrefix().size() <
                         dialect->getOperationPrefix().size())
        result = dialect.get();
  }
  return result;
}

/// Register this dialect object with the specified context.  The context
/// takes ownership of the heap allocated dialect.
void Dialect::registerDialect(MLIRContext *context) {
  context->getImpl().dialects.push_back(std::unique_ptr<Dialect>(this));
}

/// Return information about all registered operations.  This isn't very
/// efficient, typically you should ask the operations about their properties
/// directly.
std::vector<AbstractOperation *> MLIRContext::getRegisteredOperations() const {
  // We just have the operations in a non-deterministic hash table order.  Dump
  // into a temporary array, then sort it by operation name to get a stable
  // ordering.
  StringMap<AbstractOperation> &registeredOps = getImpl().registeredOperations;

  std::vector<std::pair<StringRef, AbstractOperation *>> opsToSort;
  opsToSort.reserve(registeredOps.size());
  for (auto &elt : registeredOps)
    opsToSort.push_back({elt.first(), &elt.second});

  llvm::array_pod_sort(opsToSort.begin(), opsToSort.end());

  std::vector<AbstractOperation *> result;
  result.reserve(opsToSort.size());
  for (auto &elt : opsToSort)
    result.push_back(elt.second);
  return result;
}

void Dialect::addOperation(AbstractOperation opInfo) {
  assert(opInfo.name.startswith(opPrefix) &&
         "op name doesn't start with prefix");
  assert(&opInfo.dialect == this && "Dialect object mismatch");

  auto &impl = context->getImpl();
  if (!impl.registeredOperations.insert({opInfo.name, opInfo}).second) {
    llvm::errs() << "error: ops named '" << opInfo.name
                 << "' is already registered.\n";
    abort();
  }
}

/// Look up the specified operation in the operation set and return a pointer
/// to it if present.  Otherwise, return a null pointer.
const AbstractOperation *AbstractOperation::lookup(StringRef opName,
                                                   MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.registeredOperations.find(opName);
  if (it != impl.registeredOperations.end())
    return &it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Identifier uniquing
//===----------------------------------------------------------------------===//

/// Return an identifier for the specified string.
Identifier Identifier::get(StringRef str, const MLIRContext *context) {
  assert(!str.empty() && "Cannot create an empty identifier");
  assert(str.find('\0') == StringRef::npos &&
         "Cannot create an identifier with a nul character");

  auto &impl = context->getImpl();
  auto it = impl.identifiers.insert({str, char()}).first;
  return Identifier(it->getKeyData());
}

//===----------------------------------------------------------------------===//
// Location uniquing
//===----------------------------------------------------------------------===//

UnknownLoc UnknownLoc::get(MLIRContext *context) {
  auto &impl = context->getImpl();
  if (auto *result = impl.theUnknownLoc)
    return result;

  impl.theUnknownLoc = impl.allocator.Allocate<UnknownLocationStorage>();
  new (impl.theUnknownLoc) UnknownLocationStorage{Location::Kind::Unknown};
  return impl.theUnknownLoc;
}

UniquedFilename UniquedFilename::get(StringRef filename, MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.filenames.insert({filename, char()}).first;
  return UniquedFilename(it->getKeyData());
}

FileLineColLoc FileLineColLoc::get(UniquedFilename filename, unsigned line,
                                   unsigned column, MLIRContext *context) {
  auto &impl = context->getImpl();
  auto &entry =
      impl.fileLineColLocs[std::make_tuple(filename.data(), line, column)];
  if (!entry) {
    entry = impl.allocator.Allocate<FileLineColLocationStorage>();
    new (entry) FileLineColLocationStorage{
        {Location::Kind::FileLineCol}, filename, line, column};
  }

  return entry;
}

Location FusedLoc::get(ArrayRef<Location> locs, MLIRContext *context) {
  return get(locs, Attribute(), context);
}

Location FusedLoc::get(ArrayRef<Location> locs, Attribute metadata,
                       MLIRContext *context) {
  // Unique the set of locations to be fused.
  SmallSetVector<Location, 4> decomposedLocs;
  for (auto loc : locs) {
    // If the location is a fused location we decompose it if it has no
    // metadata or the metadata is the same as the top level metadata.
    if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      if (fusedLoc->getMetadata() == metadata) {
        // UnknownLoc's have already been removed from FusedLocs so we can
        // simply add all of the internal locations.
        decomposedLocs.insert(fusedLoc->getLocations().begin(),
                              fusedLoc->getLocations().end());
        continue;
      }
    }
    // Otherwise, only add known locations to the set.
    if (!loc.isa<UnknownLoc>())
      decomposedLocs.insert(loc);
  }
  locs = decomposedLocs.getArrayRef();

  // Handle the simple cases of less than two locations.
  if (locs.empty())
    return UnknownLoc::get(context);
  if (locs.size() == 1)
    return locs.front();

  auto &impl = context->getImpl();

  // Look to see if the fused location has been created already.
  auto existing =
      impl.fusedLocs.insert_as(nullptr, std::make_pair(locs, metadata));

  // If it has been created, return it.
  if (!existing.second)
    return *existing.first;

  auto byteSize = FusedLocationStorage::totalSizeToAlloc<Location>(locs.size());
  auto rawMem =
      impl.allocator.Allocate(byteSize, alignof(FusedLocationStorage));
  auto result =
      new (rawMem) FusedLocationStorage{{Location::Kind::FusedLocation},
                                        {},
                                        static_cast<unsigned>(locs.size()),
                                        metadata};

  std::uninitialized_copy(locs.begin(), locs.end(),
                          result->getTrailingObjects<Location>());
  return *existing.first = result;
}

//===----------------------------------------------------------------------===//
// Type uniquing
//===----------------------------------------------------------------------===//

IndexType IndexType::get(MLIRContext *context) {
  auto &impl = context->getImpl();

  if (impl.indexType)
    return impl.indexType;

  impl.indexType = impl.allocator.Allocate<IndexTypeStorage>();
  new (impl.indexType) IndexTypeStorage{{Kind::Index, context}};
  return impl.indexType;
}

static IntegerType getIntegerType(unsigned width, MLIRContext *context,
                                  llvm::Optional<Location> location) {
  if (width > IntegerType::kMaxWidth) {
    if (location)
      context->emitError(*location, "integer bitwidth is limited to " +
                                        Twine(IntegerType::kMaxWidth) +
                                        " bits");
    return {};
  }

  auto &impl = context->getImpl();

  auto *&result = impl.integers[width];
  if (!result) {
    result = impl.allocator.Allocate<IntegerTypeStorage>();
    new (result) IntegerTypeStorage{{Type::Kind::Integer, context}, width};
  }

  return result;
}

IntegerType IntegerType::getChecked(unsigned width, MLIRContext *context,
                                    Location location) {
  return getIntegerType(width, context, location);
}

IntegerType IntegerType::get(unsigned width, MLIRContext *context) {
  auto type = getIntegerType(width, context, None);
  assert(type && "failed to construct IntegerType");
  return type;
}

FloatType FloatType::get(Kind kind, MLIRContext *context) {
  assert(kind >= Kind::FIRST_FLOATING_POINT_TYPE &&
         kind <= Kind::LAST_FLOATING_POINT_TYPE && "Not an FP type kind");
  auto &impl = context->getImpl();

  // We normally have these types.
  auto *&entry =
      impl.floatTypes[(int)kind - int(Kind::FIRST_FLOATING_POINT_TYPE)];
  if (entry)
    return entry;

  // On the first use, we allocate them into the bump pointer.
  auto *ptr = impl.allocator.Allocate<FloatTypeStorage>();

  // Initialize the memory using placement new.
  new (ptr) FloatTypeStorage{{kind, context}};

  // Cache and return it.
  return entry = ptr;
}

OtherType OtherType::get(Kind kind, MLIRContext *context) {
  assert(kind >= Kind::FIRST_OTHER_TYPE && kind <= Kind::LAST_OTHER_TYPE &&
         "Not an 'other' type kind");
  auto &impl = context->getImpl();

  // We normally have these types.
  auto *&entry = impl.otherTypes[(int)kind - int(Kind::FIRST_OTHER_TYPE)];
  if (entry)
    return entry;

  // On the first use, we allocate them into the bump pointer.
  auto *ptr = impl.allocator.Allocate<OtherTypeStorage>();

  // Initialize the memory using placement new.
  new (ptr) OtherTypeStorage{{kind, context}};

  // Cache and return it.
  return entry = ptr;
}

FunctionType FunctionType::get(ArrayRef<Type> inputs, ArrayRef<Type> results,
                               MLIRContext *context) {
  auto &impl = context->getImpl();

  // Look to see if we already have this function type.
  FunctionTypeKeyInfo::KeyTy key(inputs, results);
  auto existing = impl.functions.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<FunctionTypeStorage>();

  // Copy the inputs and results into the bump pointer.
  SmallVector<Type, 16> types;
  types.reserve(inputs.size() + results.size());
  types.append(inputs.begin(), inputs.end());
  types.append(results.begin(), results.end());
  auto typesList = impl.copyInto(ArrayRef<Type>(types));

  // Initialize the memory using placement new.
  new (result) FunctionTypeStorage{
      {Kind::Function, context, static_cast<unsigned int>(inputs.size())},
      static_cast<unsigned int>(results.size()),
      typesList.data()};

  // Cache and return it.
  return *existing.first = result;
}

/// Get or create a new VectorType defined by the arguments.  If the resulting
/// type would be ill-formed, return nullptr.  If the location is provided,
/// i.e. is not nullptr, emit detailed error messages.  To emit errors when
/// the location is unknown, pass in an instance of UnknownLoc.
static VectorType getVectorType(ArrayRef<int> shape, Type elementType,
                                Optional<Location> location) {
  auto *context = elementType.getContext();

  if (shape.empty()) {
    if (location)
      context->emitError(*location,
                         "vector types must have at least one dimension");
    return {};
  }

  if (!VectorType::isValidElementType(elementType)) {
    if (location)
      context->emitError(*location,
                         "vector elements must be int or float type");
    return {};
  }

  if (std::any_of(shape.begin(), shape.end(), [](int i) { return i < 0; })) {
    if (location)
      context->emitError(*location, "vector types must have static shape");
    return {};
  }

  auto &impl = context->getImpl();

  // Look to see if we already have this vector type.
  VectorTypeKeyInfo::KeyTy key(elementType, shape);
  auto existing = impl.vectors.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<VectorTypeStorage>();

  // Copy the shape into the bump pointer.
  shape = impl.copyInto(shape);

  // Initialize the memory using placement new.
  new (result) VectorTypeStorage{
      {{Type::Kind::Vector, context, static_cast<unsigned int>(shape.size())},
       elementType},
      shape.data()};

  // Cache and return it.
  return *existing.first = result;
}

// Try constructing a VectorType, report errors and return a nullptr on failure.
VectorType VectorType::getChecked(ArrayRef<int> shape, Type elementType,
                                  Location location) {
  return getVectorType(shape, elementType, location);
}

// Try constructing a VectorType, supressing error messages, abort on failure.
VectorType VectorType::get(ArrayRef<int> shape, Type elementType) {
  auto type = getVectorType(shape, elementType, None);
  assert(type && "failed to construct a VectorType");
  return type;
}

// Check if "elementType" can be an element type of a tensor. Emit errors if
// location is not nullptr.  Returns true of check failed.
static inline bool checkTensorElementType(Type elementType,
                                          Optional<Location> location) {
  auto *context = elementType.getContext();
  if (!TensorType::isValidElementType(elementType)) {
    if (location)
      context->emitError(*location, "invalid tensor element type");
    return true;
  }
  return false;
}

/// Get or create a new RankedTensorType defined by the arguments.  If the
/// resulting type would be ill-formed, return nullptr.  If the location is
/// provided, i.e. is not nullptr, emit detailed error messages.  To emit errors
/// when the location is unknown, pass in an instance of UnknownLoc.
static RankedTensorType getRankedTensorType(ArrayRef<int> shape,
                                            Type elementType,
                                            Optional<Location> location) {
  if (checkTensorElementType(elementType, location))
    return nullptr;

  auto *context = elementType.getContext();
  auto &impl = context->getImpl();

  // Look to see if we already have this ranked tensor type.
  RankedTensorTypeKeyInfo::KeyTy key(elementType, shape);
  auto existing = impl.rankedTensors.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<RankedTensorTypeStorage>();

  // Copy the shape into the bump pointer.
  shape = impl.copyInto(shape);

  // Initialize the memory using placement new.
  new (result)
      RankedTensorTypeStorage{{{{Type::Kind::RankedTensor, context,
                                 static_cast<unsigned int>(shape.size())},
                                elementType}},
                              shape.data()};

  // Cache and return it.
  return *existing.first = result;
}

RankedTensorType RankedTensorType::get(ArrayRef<int> shape, Type elementType) {
  auto type = getRankedTensorType(shape, elementType, None);
  assert(type && "failed to construct RankedTensorType");
  return type;
}

RankedTensorType RankedTensorType::getChecked(ArrayRef<int> shape,
                                              Type elementType,
                                              Location location) {
  return getRankedTensorType(shape, elementType, location);
}

/// Get or create a new UnrankedTensorType defined by the arguments.  If the
/// resulting type would be ill-formed, return nullptr.  If the location is
/// provided, i.e. is not nullptr, emit detailed error messages.  To emit errors
/// when the location is unknown, pass in an instance of UnknownLoc.
static UnrankedTensorType getUnrankedTensorType(Type elementType,
                                                Optional<Location> location) {
  if (checkTensorElementType(elementType, location))
    return nullptr;

  auto *context = elementType.getContext();
  auto &impl = context->getImpl();

  // Look to see if we already have this unranked tensor type.
  auto *&result = impl.unrankedTensors[elementType];

  // If we already have it, return that value.
  if (result)
    return result;

  // On the first use, we allocate them into the bump pointer.
  result = impl.allocator.Allocate<UnrankedTensorTypeStorage>();

  // Initialize the memory using placement new.
  new (result) UnrankedTensorTypeStorage{
      {{{Type::Kind::UnrankedTensor, context}, elementType}}};
  return result;
}

UnrankedTensorType UnrankedTensorType::get(Type elementType) {
  auto type = getUnrankedTensorType(elementType, None);
  assert(type && "failed to construct UnrankedTensorType");
  return type;
}

UnrankedTensorType UnrankedTensorType::getChecked(Type elementType,
                                                  Location location) {
  return getUnrankedTensorType(elementType, location);
}

/// Get or create a new MemRefType defined by the arguments.  If the resulting
/// type would be ill-formed, return nullptr.  If the location is provided,
/// emit detailed error messages.  To emit errors when the location is unknown,
/// pass in an instance of UnknownLoc.
static MemRefType getMemRefType(ArrayRef<int> shape, Type elementType,
                                ArrayRef<AffineMap> affineMapComposition,
                                unsigned memorySpace,
                                Optional<Location> location) {
  auto *context = elementType.getContext();
  auto &impl = context->getImpl();

  // Check that memref is formed from allowed types.
  if (!elementType.isa<IntegerType>() && !elementType.isa<FloatType>() &&
      !elementType.isa<VectorType>() && !elementType.isa<IntegerType>()) {
    if (location.hasValue())
      context->emitDiagnostic(*location, "invalid memref element type",
                              MLIRContext::DiagnosticKind::Error);
    return nullptr;
  }

  // Check that the structure of the composition is valid, i.e. that each
  // subsequent affine map has as many inputs as the previous map has results.
  // Take the dimensionality of the MemRef for the first map.
  auto dim = shape.size();
  unsigned i = 0;
  for (const auto &affineMap : affineMapComposition) {
    if (affineMap.getNumDims() != dim) {
      if (location.hasValue())
        context->emitDiagnostic(
            *location,
            "memref affine map dimension mismatch between " +
                (i == 0 ? Twine("memref rank") : "affine map " + Twine(i)) +
                " and affine map" + Twine(i + 1) + ": " + Twine(dim) +
                " != " + Twine(affineMap.getNumDims()),
            MLIRContext::DiagnosticKind::Error);
      return nullptr;
    }

    dim = affineMap.getNumResults();
    ++i;
  }

  // Drop the unbounded identity maps from the composition.
  // This may lead to the composition becoming empty, which is interpreted as an
  // implicit identity.
  llvm::SmallVector<AffineMap, 2> cleanedAffineMapComposition;
  for (const auto &map : affineMapComposition) {
    if (map.isIdentity() && !map.isBounded())
      continue;
    cleanedAffineMapComposition.push_back(map);
  }
  affineMapComposition = cleanedAffineMapComposition;

  // Look to see if we already have this memref type.
  auto key =
      std::make_tuple(elementType, shape, affineMapComposition, memorySpace);
  auto existing = impl.memrefs.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<MemRefTypeStorage>();

  // Copy the shape into the bump pointer.
  shape = impl.copyInto(shape);

  // Copy the affine map composition into the bump pointer.
  affineMapComposition =
      impl.copyInto(ArrayRef<AffineMap>(affineMapComposition));

  // Initialize the memory using placement new.
  new (result) MemRefTypeStorage{
      {Type::Kind::MemRef, context, static_cast<unsigned int>(shape.size())},
      elementType,
      shape.data(),
      static_cast<unsigned int>(affineMapComposition.size()),
      affineMapComposition.data(),
      memorySpace};
  // Cache and return it.
  return *existing.first = result;
}

// Try constructing a MemRefType, report errors and return a nullptr on failure.
MemRefType MemRefType::getChecked(ArrayRef<int> shape, Type elementType,
                                  ArrayRef<AffineMap> affineMapComposition,
                                  unsigned memorySpace, Location location) {
  return getMemRefType(shape, elementType, affineMapComposition, memorySpace,
                       location);
}

// Try constructing a MemRefType, supressing error messages, abort on failure.
MemRefType MemRefType::get(ArrayRef<int> shape, Type elementType,
                           ArrayRef<AffineMap> affineMapComposition,
                           unsigned memorySpace) {
  auto type =
      getMemRefType(shape, elementType, affineMapComposition, memorySpace, {});
  assert(type && "failed to construct a MemRef type");
  return type;
}

//===----------------------------------------------------------------------===//
// Attribute uniquing
//===----------------------------------------------------------------------===//

BoolAttr BoolAttr::get(bool value, MLIRContext *context) {
  auto *&result = context->getImpl().boolAttrs[value];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<BoolAttributeStorage>();
  new (result) BoolAttributeStorage{{Attribute::Kind::Bool,
                                     /*isOrContainsFunction=*/false},
                                    value};
  return result;
}

IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  auto &impl = type.getContext()->getImpl();

  // Look to see if the integer attribute has been created already.
  IntegerAttrKeyInfo::KeyTy key({type, value});
  auto existing = impl.integerAttrs.insert_as(nullptr, key);

  // If it has been created, return it.
  if (!existing.second)
    return *existing.first;

  // If it doesn't, create one and return it.
  auto elements = ArrayRef<uint64_t>(value.getRawData(), value.getNumWords());

  auto byteSize =
      IntegerAttributeStorage::totalSizeToAlloc<uint64_t>(elements.size());
  auto rawMem =
      impl.allocator.Allocate(byteSize, alignof(IntegerAttributeStorage));
  // TODO: This uses 64 bit APInts by default without consideration of value.
  auto result = ::new (rawMem) IntegerAttributeStorage{
      {Attribute::Kind::Integer, /*isOrContainsFunction=*/false},
      type,
      elements.size()};
  std::uninitialized_copy(elements.begin(), elements.end(),
                          result->getTrailingObjects<uint64_t>());
  return *existing.first = result;
}

IntegerAttr IntegerAttr::get(Type type, int64_t value) {
  // This uses 64 bit APInts by default for index type.
  auto width = type.isIndex() ? 64 : type.getBitWidth();
  return get(type, APInt(width, value));
}

/// Returns the floating semantics for the given type.
static const fltSemantics &getFloatSemantics(Type type) {
  if (type.isBF16())
    // Treat BF16 like a double. This is unfortunate but BF16 fltSemantics is
    // not defined in LLVM.
    // TODO(jpienaar): add BF16 to LLVM? fltSemantics are internal to APFloat.cc
    // else one could add it.
    //  static const fltSemantics semBF16 = {127, -126, 8, 16};
    return APFloat::IEEEdouble();
  if (type.isF16())
    // Treat F16 as double. This avoids needing to change the tensor element
    // parsing for now. TODO: Fix this to use the correct semantics instead.
    return APFloat::IEEEdouble();
  if (type.isF32())
    return APFloat::IEEEsingle();
  if (type.isF64())
    return APFloat::IEEEdouble();
  llvm_unreachable("non-floating point type used");
}

FloatAttr FloatAttr::get(Type type, double value) {
  Optional<APFloat> val;
  if (type.isBF16() || type.isF16())
    // Treat BF16 and F16 as double. This avoids needing to change the tensor
    // element parsing for now. TODO: Fix this to use the correct semantics
    // instead.
    val = APFloat(value);
  else if (type.isF32())
    val = APFloat(static_cast<float>(value));
  else if (type.isF64())
    val = APFloat(value);
  else {
    bool unused;
    val = APFloat(value);
    auto status =
        (*val).convert(getFloatSemantics(type), APFloat::rmTowardZero, &unused);
    if (status != APFloat::opOK) {
      auto context = type.getContext();
      context->emitError(
          UnknownLoc::get(context),
          "failed to convert floating point value to requested type");
      val.reset();
    }
  }
  return get(type, *val);
}

FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  assert(&getFloatSemantics(type) == &value.getSemantics() &&
         "FloatAttr type doesn't match the type implied by its value");
  auto &impl = type.getContext()->getImpl();

  // Look to see if the float attribute has been created already.
  FloatAttrKeyInfo::KeyTy key({type, value});
  auto existing = impl.floatAttrs.insert_as(nullptr, key);

  // If it has been created, return it.
  if (!existing.second)
    return *existing.first;

  // If it doesn't, create one, unique it and return it.
  const auto &apint = value.bitcastToAPInt();
  // Here one word's bitwidth equals to that of uint64_t.
  auto elements = ArrayRef<uint64_t>(apint.getRawData(), apint.getNumWords());

  auto byteSize =
      FloatAttributeStorage::totalSizeToAlloc<uint64_t>(elements.size());
  auto rawMem =
      impl.allocator.Allocate(byteSize, alignof(FloatAttributeStorage));
  auto result = ::new (rawMem) FloatAttributeStorage{
      {Attribute::Kind::Float, /*isOrContainsFunction=*/false},
      value.getSemantics(),
      type,
      elements.size()};
  std::uninitialized_copy(elements.begin(), elements.end(),
                          result->getTrailingObjects<uint64_t>());
  return *existing.first = result;
}

StringAttr StringAttr::get(StringRef bytes, MLIRContext *context) {
  auto it = context->getImpl().stringAttrs.insert({bytes, nullptr}).first;

  if (it->second)
    return it->second;

  auto result = context->getImpl().allocator.Allocate<StringAttributeStorage>();
  new (result) StringAttributeStorage{{Attribute::Kind::String,
                                       /*isOrContainsFunction=*/false},
                                      it->first()};
  it->second = result;
  return result;
}

ArrayAttr ArrayAttr::get(ArrayRef<Attribute> value, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Look to see if we already have this.
  auto existing = impl.arrayAttrs.insert_as(nullptr, value);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<ArrayAttributeStorage>();

  // Copy the elements into the bump pointer.
  value = impl.copyInto(value);

  // Check to see if any of the elements have a function attr.
  bool hasFunctionAttr = false;
  for (auto elt : value)
    if (elt.isOrContainsFunction()) {
      hasFunctionAttr = true;
      break;
    }

  // Initialize the memory using placement new.
  new (result)
      ArrayAttributeStorage{{Attribute::Kind::Array, hasFunctionAttr}, value};

  // Cache and return it.
  return *existing.first = result;
}

AffineMapAttr AffineMapAttr::get(AffineMap value) {
  auto *context = value.getResult(0).getContext();
  auto &result = context->getImpl().affineMapAttrs[value];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<AffineMapAttributeStorage>();
  new (result) AffineMapAttributeStorage{{Attribute::Kind::AffineMap,
                                          /*isOrContainsFunction=*/false},
                                         value};
  return result;
}

IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  auto *context = value.getConstraint(0).getContext();
  auto &result = context->getImpl().integerSetAttrs[value];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<IntegerSetAttributeStorage>();
  new (result) IntegerSetAttributeStorage{{Attribute::Kind::IntegerSet,
                                           /*isOrContainsFunction=*/false},
                                          value};
  return result;
}

TypeAttr TypeAttr::get(Type type, MLIRContext *context) {
  auto *&result = context->getImpl().typeAttrs[type];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<TypeAttributeStorage>();
  new (result) TypeAttributeStorage{{Attribute::Kind::Type,
                                     /*isOrContainsFunction=*/false},
                                    type};
  return result;
}

FunctionAttr FunctionAttr::get(const Function *value, MLIRContext *context) {
  assert(value && "Cannot get FunctionAttr for a null function");

  auto *&result = context->getImpl().functionAttrs[value];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<FunctionAttributeStorage>();
  new (result) FunctionAttributeStorage{{Attribute::Kind::Function,
                                         /*isOrContainsFunction=*/true},
                                        const_cast<Function *>(value)};
  return result;
}

/// This function is used by the internals of the Function class to null out
/// attributes refering to functions that are about to be deleted.
void FunctionAttr::dropFunctionReference(Function *value) {
  // Check to see if there was an attribute referring to this function.
  auto &functionAttrs = value->getContext()->getImpl().functionAttrs;

  // If not, then we're done.
  auto it = functionAttrs.find(value);
  if (it == functionAttrs.end())
    return;

  // If so, null out the function reference in the attribute (to avoid dangling
  // pointers) and remove the entry from the map so the map doesn't contain
  // dangling keys.
  it->second->value = nullptr;
  functionAttrs.erase(it);
}

/// Perform a three-way comparison between the names of the specified
/// NamedAttributes.
static int compareNamedAttributes(const NamedAttribute *lhs,
                                  const NamedAttribute *rhs) {
  return lhs->first.str().compare(rhs->first.str());
}

/// Given a list of NamedAttribute's, canonicalize the list (sorting
/// by name) and return the unique'd result.  Note that the empty list is
/// represented with a null pointer.
AttributeListStorage *AttributeListStorage::get(ArrayRef<NamedAttribute> attrs,
                                                MLIRContext *context) {
  // We need to sort the element list to canonicalize it, but we also don't want
  // to do a ton of work in the super common case where the element list is
  // already sorted.
  SmallVector<NamedAttribute, 8> storage;
  switch (attrs.size()) {
  case 0:
    // An empty list is represented with a null pointer.
    return nullptr;
  case 1:
    // A single element is already sorted.
    break;
  case 2:
    // Don't invoke a general sort for two element case.
    if (attrs[0].first.str() > attrs[1].first.str()) {
      storage.push_back(attrs[1]);
      storage.push_back(attrs[0]);
      attrs = storage;
    }
    break;
  default:
    // Check to see they are sorted already.
    bool isSorted = true;
    for (unsigned i = 0, e = attrs.size() - 1; i != e; ++i) {
      if (attrs[i].first.str() > attrs[i + 1].first.str()) {
        isSorted = false;
        break;
      }
    }
    // If not, do a general sort.
    if (!isSorted) {
      storage.append(attrs.begin(), attrs.end());
      llvm::array_pod_sort(storage.begin(), storage.end(),
                           compareNamedAttributes);
      attrs = storage;
    }
  }

  auto &impl = context->getImpl();

  // Look to see if we already have this.
  auto existing = impl.attributeLists.insert_as(nullptr, attrs);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // Otherwise, allocate a new AttributeListStorage, unique it and return it.
  auto byteSize =
      AttributeListStorage::totalSizeToAlloc<NamedAttribute>(attrs.size());
  auto rawMem = impl.allocator.Allocate(byteSize, alignof(NamedAttribute));

  //  Placement initialize the AggregateSymbolicValue.
  auto result = ::new (rawMem) AttributeListStorage(attrs.size());
  std::uninitialized_copy(attrs.begin(), attrs.end(),
                          result->getTrailingObjects<NamedAttribute>());
  return *existing.first = result;
}

SplatElementsAttr SplatElementsAttr::get(VectorOrTensorType type,
                                         Attribute elt) {
  // TODO(fengliuai): Add verification that the Attribute matches the element
  // type.
  auto &impl = type.getContext()->getImpl();

  // Look to see if we already have this.
  auto *&result = impl.splatElementsAttrs[{type, elt}];

  // If we already have it, return that value.
  if (result)
    return result;

  // Otherwise, allocate them into the bump pointer.
  result = impl.allocator.Allocate<SplatElementsAttributeStorage>();
  new (result) SplatElementsAttributeStorage{{{Attribute::Kind::SplatElements,
                                               /*isOrContainsFunction=*/false},
                                              type},
                                             elt};

  return result;
}

DenseElementsAttr DenseElementsAttr::get(VectorOrTensorType type,
                                         ArrayRef<char> data) {
  auto bitsRequired = (long)type.getBitWidth() * type.getNumElements();
  (void)bitsRequired;
  assert((bitsRequired <= data.size() * 8L) &&
         "Input data bit size should be larger than that type requires");

  auto &impl = type.getContext()->getImpl();

  // Look to see if this constant is already defined.
  DenseElementsAttrInfo::KeyTy key({type, data});
  auto existing = impl.denseElementsAttrs.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // Otherwise, allocate a new one, unique it and return it.
  auto eltType = type.getElementType();
  switch (eltType.getKind()) {
  case Type::Kind::BF16:
  case Type::Kind::F16:
  case Type::Kind::F32:
  case Type::Kind::F64: {
    auto *result = impl.allocator.Allocate<DenseFPElementsAttributeStorage>();
    auto *copy = (char *)impl.allocator.Allocate(data.size(), 64);
    std::uninitialized_copy(data.begin(), data.end(), copy);
    new (result) DenseFPElementsAttributeStorage{
        {{{Attribute::Kind::DenseFPElements, /*isOrContainsFunction=*/false},
          type},
         {copy, data.size()}}};
    return *existing.first = result;
  }
  case Type::Kind::Integer: {
    auto width = eltType.cast<IntegerType>().getWidth();
    auto *result = impl.allocator.Allocate<DenseIntElementsAttributeStorage>();
    auto *copy = (char *)impl.allocator.Allocate(data.size(), 64);
    std::uninitialized_copy(data.begin(), data.end(), copy);
    new (result) DenseIntElementsAttributeStorage{
        {{{Attribute::Kind::DenseIntElements, /*isOrContainsFunction=*/false},
          type},
         {copy, data.size()}},
        width};
    return *existing.first = result;
  }
  default:
    llvm_unreachable("unexpected element type");
  }
}

OpaqueElementsAttr OpaqueElementsAttr::get(VectorOrTensorType type,
                                           StringRef bytes) {
  assert(TensorType::isValidElementType(type.getElementType()) &&
         "Input element type should be a valid tensor element type");

  auto &impl = type.getContext()->getImpl();

  // Look to see if this constant is already defined.
  OpaqueElementsAttrInfo::KeyTy key({type, bytes});
  auto existing = impl.opaqueElementsAttrs.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // Otherwise, allocate a new one, unique it and return it.
  auto *result = impl.allocator.Allocate<OpaqueElementsAttributeStorage>();
  bytes = bytes.copy(impl.allocator);
  new (result) OpaqueElementsAttributeStorage{
      {{Attribute::Kind::OpaqueElements, /*isOrContainsFunction=*/false}, type},
      bytes};
  return *existing.first = result;
}

SparseElementsAttr SparseElementsAttr::get(VectorOrTensorType type,
                                           DenseIntElementsAttr indices,
                                           DenseElementsAttr values) {
  auto &impl = type.getContext()->getImpl();

  // Look to see if we already have this.
  auto key = std::make_tuple(type, indices, values);
  auto *&result = impl.sparseElementsAttrs[key];

  // If we already have it, return that value.
  if (result)
    return result;

  // Otherwise, allocate them into the bump pointer.
  result = impl.allocator.Allocate<SparseElementsAttributeStorage>();
  new (result) SparseElementsAttributeStorage{{{Attribute::Kind::SparseElements,
                                                /*isOrContainsFunction=*/false},
                                               type},
                                              indices,
                                              values};

  return result;
}

//===----------------------------------------------------------------------===//
// AffineMap and AffineExpr uniquing
//===----------------------------------------------------------------------===//

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> results,
                         ArrayRef<AffineExpr> rangeSizes) {
  // The number of results can't be zero.
  assert(!results.empty());

  assert(rangeSizes.empty() || results.size() == rangeSizes.size());

  auto &impl = results[0].getContext()->getImpl();

  // Check if we already have this affine map.
  auto key = std::make_tuple(dimCount, symbolCount, results, rangeSizes);
  auto existing = impl.affineMaps.insert_as(AffineMap(nullptr), key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *res = impl.allocator.Allocate<detail::AffineMapStorage>();

  // Copy the results and range sizes into the bump pointer.
  results = impl.copyInto(results);
  rangeSizes = impl.copyInto(rangeSizes);

  // Initialize the memory using placement new.
  new (res)
      detail::AffineMapStorage{dimCount, symbolCount, results, rangeSizes};

  // Cache and return it.
  return *existing.first = AffineMap(res);
}

/// Simplify add expression. Return nullptr if it can't be simplified.
static AffineExpr simplifyAdd(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();
  // Fold if both LHS, RHS are a constant.
  if (lhsConst && rhsConst)
    return getAffineConstantExpr(lhsConst.getValue() + rhsConst.getValue(),
                                 lhs.getContext());

  // Canonicalize so that only the RHS is a constant. (4 + d0 becomes d0 + 4).
  // If only one of them is a symbolic expressions, make it the RHS.
  if (lhs.isa<AffineConstantExpr>() ||
      (lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant())) {
    return rhs + lhs;
  }

  // At this point, if there was a constant, it would be on the right.

  // Addition with a zero is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst.getValue() == 0)
      return lhs;
  }
  // Fold successive additions like (d0 + 2) + 3 into d0 + 5.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && rhsConst && lBin.getKind() == AffineExprKind::Add) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>())
      return lBin.getLHS() + (lrhs.getValue() + rhsConst.getValue());
  }

  // When doing successive additions, bring constant to the right: turn (d0 + 2)
  // + d1 into (d0 + d1) + 2.
  if (lBin && lBin.getKind() == AffineExprKind::Add) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      return lBin.getLHS() + rhs + lrhs;
    }
  }

  return nullptr;
}

/// Simplify a multiply expression. Return nullptr if it can't be simplified.
static AffineExpr simplifyMul(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  if (lhsConst && rhsConst)
    return getAffineConstantExpr(lhsConst.getValue() * rhsConst.getValue(),
                                 lhs.getContext());

  assert(lhs.isSymbolicOrConstant() || rhs.isSymbolicOrConstant());

  // Canonicalize the mul expression so that the constant/symbolic term is the
  // RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a
  // constant. (Note that a constant is trivially symbolic).
  if (!rhs.isSymbolicOrConstant() || lhs.isa<AffineConstantExpr>()) {
    // At least one of them has to be symbolic.
    return rhs * lhs;
  }

  // At this point, if there was a constant, it would be on the right.

  // Multiplication with a one is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst.getValue() == 1)
      return lhs;
    // Multiplication with zero.
    if (rhsConst.getValue() == 0)
      return rhsConst;
  }

  // Fold successive multiplications: eg: (d0 * 2) * 3 into d0 * 6.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && rhsConst && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>())
      return lBin.getLHS() * (lrhs.getValue() * rhsConst.getValue());
  }

  // When doing successive multiplication, bring constant to the right: turn (d0
  // * 2) * d1 into (d0 * d1) * 2.
  if (lBin && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      return (lBin.getLHS() * rhs) * lrhs;
    }
  }

  return nullptr;
}

static AffineExpr simplifyFloorDiv(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getAffineConstantExpr(
        floorDiv(lhsConst.getValue(), rhsConst.getValue()), lhs.getContext());

  // Fold floordiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) floordiv 64 = i * 2.
  if (rhsConst.getValue() == 1)
    return lhs;

  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      // rhsConst is known to be positive if a constant.
      if (lrhs.getValue() % rhsConst.getValue() == 0)
        return lBin.getLHS() * (lrhs.getValue() / rhsConst.getValue());
    }
  }

  return nullptr;
}

static AffineExpr simplifyCeilDiv(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getAffineConstantExpr(
        ceilDiv(lhsConst.getValue(), rhsConst.getValue()), lhs.getContext());

  // Fold ceildiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) ceildiv 64 = i * 2.
  if (rhsConst.getValue() == 1)
    return lhs;

  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      // rhsConst is known to be positive if a constant.
      if (lrhs.getValue() % rhsConst.getValue() == 0)
        return lBin.getLHS() * (lrhs.getValue() / rhsConst.getValue());
    }
  }

  return nullptr;
}

static AffineExpr simplifyMod(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getAffineConstantExpr(mod(lhsConst.getValue(), rhsConst.getValue()),
                                 lhs.getContext());

  // Fold modulo of an expression that is known to be a multiple of a constant
  // to zero if that constant is a multiple of the modulo factor. Eg: (i * 128)
  // mod 64 is folded to 0, and less trivially, (i*(j*4*(k*32))) mod 128 = 0.
  if (lhs.getLargestKnownDivisor() % rhsConst.getValue() == 0)
    return getAffineConstantExpr(0, lhs.getContext());

  return nullptr;
  // TODO(bondhugula): In general, this can be simplified more by using the GCD
  // test, or in general using quantifier elimination (add two new variables q
  // and r, and eliminate all variables from the linear system other than r. All
  // of this can be done through mlir/Analysis/'s FlatAffineConstraints.
}

/// Return a binary affine op expression with the specified op type and
/// operands: if it doesn't exist, create it and store it; if it is already
/// present, return from the list. The stored expressions are unique: they are
/// constructed and stored in a simplified/canonicalized form. The result after
/// simplification could be any form of affine expression.
AffineExpr AffineBinaryOpExprStorage::get(AffineExprKind kind, AffineExpr lhs,
                                          AffineExpr rhs) {
  auto &impl = lhs.getContext()->getImpl();

  // Check if we already have this affine expression, and return it if we do.
  auto keyValue = std::make_tuple((unsigned)kind, lhs, rhs);
  auto cached = impl.affineExprs.find(keyValue);
  if (cached != impl.affineExprs.end())
    return cached->second;

  // Simplify the expression if possible.
  AffineExpr simplified;
  switch (kind) {
  case AffineExprKind::Add:
    simplified = simplifyAdd(lhs, rhs);
    break;
  case AffineExprKind::Mul:
    simplified = simplifyMul(lhs, rhs);
    break;
  case AffineExprKind::FloorDiv:
    simplified = simplifyFloorDiv(lhs, rhs);
    break;
  case AffineExprKind::CeilDiv:
    simplified = simplifyCeilDiv(lhs, rhs);
    break;
  case AffineExprKind::Mod:
    simplified = simplifyMod(lhs, rhs);
    break;
  default:
    llvm_unreachable("unexpected binary affine expr");
  }

  // The simplified one would have already been cached; just return it.
  if (simplified)
    return simplified;

  // An expression with these operands will already be in the
  // simplified/canonical form. Create and store it.
  auto *result = impl.allocator.Allocate<AffineBinaryOpExprStorage>();
  // Initialize the memory using placement new.
  new (result) AffineBinaryOpExprStorage{{kind, lhs.getContext()}, lhs, rhs};
  bool inserted = impl.affineExprs.insert({keyValue, result}).second;
  assert(inserted && "the expression shouldn't already exist in the map");
  (void)inserted;
  return result;
}

AffineExpr mlir::getAffineDimExpr(unsigned position, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Check if we need to resize.
  if (position >= impl.dimExprs.size())
    impl.dimExprs.resize(position + 1, nullptr);

  auto *&result = impl.dimExprs[position];
  if (result)
    return result;

  result = impl.allocator.Allocate<AffineDimExprStorage>();
  // Initialize the memory using placement new.
  new (result) AffineDimExprStorage{{AffineExprKind::DimId, context}, position};
  return result;
}

AffineExpr mlir::getAffineSymbolExpr(unsigned position, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Check if we need to resize.
  if (position >= impl.symbolExprs.size())
    impl.symbolExprs.resize(position + 1, nullptr);

  auto *&result = impl.symbolExprs[position];
  if (result)
    return result;

  result = impl.allocator.Allocate<AffineSymbolExprStorage>();
  // Initialize the memory using placement new.
  new (result)
      AffineSymbolExprStorage{{AffineExprKind::SymbolId, context}, position};
  return result;
}

AffineExpr mlir::getAffineConstantExpr(int64_t constant, MLIRContext *context) {
  auto &impl = context->getImpl();
  auto *&result = impl.constExprs[constant];

  if (result)
    return result;

  result = impl.allocator.Allocate<AffineConstantExprStorage>();
  // Initialize the memory using placement new.
  new (result)
      AffineConstantExprStorage{{AffineExprKind::Constant, context}, constant};
  return result;
}

//===----------------------------------------------------------------------===//
// Integer Sets: these are allocated into the bump pointer, and are immutable.
// Unlike AffineMap's, these are uniqued only if they are small.
//===----------------------------------------------------------------------===//

IntegerSet IntegerSet::get(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<AffineExpr> constraints,
                           ArrayRef<bool> eqFlags) {
  // The number of constraints can't be zero.
  assert(!constraints.empty());
  assert(constraints.size() == eqFlags.size());

  bool unique = constraints.size() < IntegerSet::kUniquingThreshold;

  auto &impl = constraints[0].getContext()->getImpl();

  std::pair<DenseSet<IntegerSet, IntegerSetKeyInfo>::Iterator, bool> existing;
  if (unique) {
    // Check if we already have this integer set.
    auto key = std::make_tuple(dimCount, symbolCount, constraints, eqFlags);
    existing = impl.integerSets.insert_as(IntegerSet(nullptr), key);

    // If we already have it, return that value.
    if (!existing.second)
      return *existing.first;
  }

  // On the first use, we allocate them into the bump pointer.
  auto *res = impl.allocator.Allocate<detail::IntegerSetStorage>();

  // Copy the results and equality flags into the bump pointer.
  constraints = impl.copyInto(constraints);
  eqFlags = impl.copyInto(eqFlags);

  // Initialize the memory using placement new.
  new (res)
      detail::IntegerSetStorage{dimCount, symbolCount, constraints, eqFlags};

  if (unique)
    // Cache and return it.
    return *existing.first = IntegerSet(res);

  return IntegerSet(res);
}
