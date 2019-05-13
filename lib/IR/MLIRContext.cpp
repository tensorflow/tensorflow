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
#include "IntegerSetDetail.h"
#include "LocationDetail.h"
#include "SDBMExprDetail.h"
#include "TypeDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::detail;

using llvm::hash_combine;
using llvm::hash_combine_range;

/// A utility function to safely get or create a uniqued instance within the
/// given set container.
template <typename ValueT, typename DenseInfoT, typename KeyT,
          typename ConstructorFn>
static ValueT safeGetOrCreate(DenseSet<ValueT, DenseInfoT> &container,
                              KeyT &&key, llvm::sys::SmartRWMutex<true> &mutex,
                              ConstructorFn &&constructorFn) {
  { // Check for an existing instance in read-only mode.
    llvm::sys::SmartScopedReader<true> instanceLock(mutex);
    auto it = container.find_as(key);
    if (it != container.end())
      return *it;
  }

  // Aquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> instanceLock(mutex);

  // Check for an existing instance again here, because another writer thread
  // may have already created one.
  auto existing = container.insert_as(ValueT(), key);
  if (!existing.second)
    return *existing.first;

  // Otherwise, construct a new instance of the value.
  return *existing.first = constructorFn();
}

/// A utility function to thread-safely get or create a uniqued instance within
/// the given vector container.
template <typename ValueT, typename ConstructorFn>
ValueT safeGetOrCreate(std::vector<ValueT> &container, unsigned position,
                       llvm::sys::SmartRWMutex<true> &mutex,
                       ConstructorFn &&constructorFn) {
  { // Check for an existing instance in read-only mode.
    llvm::sys::SmartScopedReader<true> lock(mutex);
    if (container.size() > position && container[position])
      return container[position];
  }

  // Aquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> lock(mutex);

  // Check if we need to resize.
  if (position >= container.size())
    container.resize(position + 1, nullptr);

  // Check for an existing instance again here, because another writer thread
  // may have already created one.
  auto *&result = container[position];
  if (result)
    return result;

  return result = constructorFn();
}

/// A utility function to safely get or create a uniqued instance within the
/// given map container.
template <typename ContainerTy, typename KeyT, typename ConstructorFn>
static typename ContainerTy::mapped_type
safeGetOrCreate(ContainerTy &container, KeyT &&key,
                llvm::sys::SmartRWMutex<true> &mutex,
                ConstructorFn &&constructorFn) {
  { // Check for an existing instance in read-only mode.
    llvm::sys::SmartScopedReader<true> instanceLock(mutex);
    auto it = container.find(key);
    if (it != container.end())
      return it->second;
  }

  // Aquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> instanceLock(mutex);

  // Check for an existing instance again here, because another writer thread
  // may have already created one.
  auto *&result = container[key];
  if (result)
    return result;

  // Otherwise, construct a new instance of the value.
  return result = constructorFn();
}

namespace {
/// A builtin dialect to define types/etc that are necessary for the validity of
/// the IR.
struct BuiltinDialect : public Dialect {
  BuiltinDialect(MLIRContext *context) : Dialect(/*name=*/"", context) {
    addAttributes<AffineMapAttr, ArrayAttr, BoolAttr, DenseIntElementsAttr,
                  DenseFPElementsAttr, FloatAttr, FunctionAttr, IntegerAttr,
                  IntegerSetAttr, OpaqueElementsAttr, SparseElementsAttr,
                  SplatElementsAttr, StringAttr, TypeAttr, UnitAttr>();
    addTypes<ComplexType, FloatType, FunctionType, IndexType, IntegerType,
             MemRefType, NoneType, OpaqueType, RankedTensorType, TupleType,
             UnrankedTensorType, VectorType>();
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

struct CallSiteLocationKeyInfo : DenseMapInfo<CallSiteLocationStorage *> {
  // Call locations are uniqued based on their held concret location
  // and the caller location.
  using KeyTy = std::pair<Location, Location>;
  using DenseMapInfo<CallSiteLocationStorage *>::isEqual;

  static unsigned getHashValue(CallSiteLocationStorage *key) {
    return getHashValue(KeyTy(key->callee, key->caller));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(key.first, key.second);
  }

  static bool isEqual(const KeyTy &lhs, const CallSiteLocationStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_pair(rhs->callee, rhs->caller);
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
  //===--------------------------------------------------------------------===//
  // Location uniquing
  //===--------------------------------------------------------------------===//

  // Location allocator and mutex for thread safety.
  llvm::BumpPtrAllocator locationAllocator;
  llvm::sys::SmartRWMutex<true> locationMutex;

  /// The singleton for UnknownLoc.
  UnknownLocationStorage theUnknownLoc;

  /// These are filename locations uniqued into this MLIRContext.
  llvm::StringMap<char, llvm::BumpPtrAllocator &> filenames;

  /// FileLineColLoc uniquing.
  DenseMap<std::tuple<const char *, unsigned, unsigned>,
           FileLineColLocationStorage *>
      fileLineColLocs;

  /// NameLocation uniquing.
  DenseMap<const char *, NameLocationStorage *> nameLocs;

  /// CallLocation uniquing.
  DenseSet<CallSiteLocationStorage *, CallSiteLocationKeyInfo> callLocs;

  /// FusedLoc uniquing.
  using FusedLocations = DenseSet<FusedLocationStorage *, FusedLocKeyInfo>;
  FusedLocations fusedLocs;

  //===--------------------------------------------------------------------===//
  // Identifier uniquing
  //===--------------------------------------------------------------------===//

  // Identifier allocator and mutex for thread safety.
  llvm::BumpPtrAllocator identifierAllocator;
  llvm::sys::SmartRWMutex<true> identifierMutex;

  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//
  DiagnosticEngine diagEngine;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// A general purpose mutex to lock access to parts of the context that do not
  /// have a more specific mutex, e.g. registry operations.
  llvm::sys::SmartRWMutex<true> contextMutex;

  /// This is a list of dialects that are created referring to this context.
  /// The MLIRContext owns the objects.
  std::vector<std::unique_ptr<Dialect>> dialects;

  /// This is a mapping from operation name to AbstractOperation for registered
  /// operations.
  llvm::StringMap<AbstractOperation> registeredOperations;

  /// This is a mapping from class identifier to Dialect for registered
  /// attributes and types.
  DenseMap<const ClassID *, Dialect *> registeredDialectSymbols;

  /// These are identifiers uniqued into this MLIRContext.
  llvm::StringMap<char, llvm::BumpPtrAllocator &> identifiers;

  //===--------------------------------------------------------------------===//
  // Affine uniquing
  //===--------------------------------------------------------------------===//

  // Affine allocator and mutex for thread safety.
  llvm::BumpPtrAllocator affineAllocator;
  llvm::sys::SmartRWMutex<true> affineMutex;

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

  //===--------------------------------------------------------------------===//
  // SDBM uniquing
  //===--------------------------------------------------------------------===//
  llvm::BumpPtrAllocator SDBMAllocator;
  llvm::sys::SmartRWMutex<true> SDBMMutex;

  DenseMap<std::tuple<SDBMVaryingExpr, SDBMConstantExpr>,
           SDBMBinaryExprStorage *>
      SDBMSumExprs;
  DenseMap<std::tuple<SDBMPositiveExpr, SDBMConstantExpr>,
           SDBMBinaryExprStorage *>
      SDBMStripeExprs;
  DenseMap<std::tuple<SDBMPositiveExpr, SDBMPositiveExpr>,
           SDBMDiffExprStorage *>
      SDBMDiffExprs;
  std::vector<SDBMPositiveExprStorage *> SDBMDimExprs;
  std::vector<SDBMPositiveExprStorage *> SDBMSymbolExprs;
  DenseMap<SDBMPositiveExpr, SDBMNegExprStorage *> SDBMNegExprs;
  DenseMap<int64_t, SDBMConstantExprStorage *> SDBMConstExprs;

  //===--------------------------------------------------------------------===//
  // Type uniquing
  //===--------------------------------------------------------------------===//
  StorageUniquer typeUniquer;

  //===--------------------------------------------------------------------===//
  // Attribute uniquing
  //===--------------------------------------------------------------------===//
  StorageUniquer attributeUniquer;

  // Attribute list allocator and mutex for thread safety.
  llvm::BumpPtrAllocator attributeAllocator;
  llvm::sys::SmartRWMutex<true> attributeMutex;

  using AttributeListSet =
      DenseSet<AttributeListStorage *, AttributeListKeyInfo>;
  AttributeListSet attributeLists;

public:
  MLIRContextImpl()
      : filenames(locationAllocator), identifiers(identifierAllocator) {}
};
} // end namespace mlir

MLIRContext::MLIRContext() : impl(new MLIRContextImpl()) {
  new BuiltinDialect(this);
  registerAllDialects(this);
}

MLIRContext::~MLIRContext() {}

/// Copy the specified array of elements into memory managed by the provided
/// bump pointer allocator.  This assumes the elements are all PODs.
template <typename T>
static ArrayRef<T> copyArrayRefInto(llvm::BumpPtrAllocator &allocator,
                                    ArrayRef<T> elements) {
  auto result = allocator.Allocate<T>(elements.size());
  std::uninitialized_copy(elements.begin(), elements.end(), result);
  return ArrayRef<T>(result, elements.size());
}

//===----------------------------------------------------------------------===//
// Diagnostic Handlers
//===----------------------------------------------------------------------===//

/// Helper function used to emit a diagnostic with an optionally empty twine
/// message. If the message is empty, then it is not inserted into the
/// diagnostic.
static InFlightDiagnostic emitDiag(MLIRContextImpl &ctx, Location location,
                                   DiagnosticSeverity severity,
                                   const llvm::Twine &message) {
  auto diag = ctx.diagEngine.emit(location, severity);
  if (!message.isTriviallyEmpty())
    diag << message;
  return diag;
}

InFlightDiagnostic MLIRContext::emitError(Location location) {
  return emitError(location, /*message=*/{});
}
InFlightDiagnostic MLIRContext::emitError(Location location,
                                          const llvm::Twine &message) {
  return emitDiag(getImpl(), location, DiagnosticSeverity::Error, message);
}

/// Emit a warning message using the diagnostic engine.
InFlightDiagnostic MLIRContext::emitWarning(Location location) {
  return emitWarning(location, /*message=*/{});
}
InFlightDiagnostic MLIRContext::emitWarning(Location location,
                                            const Twine &message) {
  return emitDiag(getImpl(), location, DiagnosticSeverity::Warning, message);
}

/// Emit a remark message using the diagnostic engine.
InFlightDiagnostic MLIRContext::emitRemark(Location location) {
  return emitRemark(location, /*message=*/{});
}
InFlightDiagnostic MLIRContext::emitRemark(Location location,
                                           const Twine &message) {
  return emitDiag(getImpl(), location, DiagnosticSeverity::Remark, message);
}

/// Returns the diagnostic engine for this context.
DiagnosticEngine &MLIRContext::getDiagEngine() { return getImpl().diagEngine; }

//===----------------------------------------------------------------------===//
// Dialect and Operation Registration
//===----------------------------------------------------------------------===//

/// Return information about all registered IR dialects.
std::vector<Dialect *> MLIRContext::getRegisteredDialects() {
  // Lock access to the context registry.
  llvm::sys::SmartScopedReader<true> registryLock(getImpl().contextMutex);

  std::vector<Dialect *> result;
  result.reserve(getImpl().dialects.size());
  for (auto &dialect : getImpl().dialects)
    result.push_back(dialect.get());
  return result;
}

/// Get a registered IR dialect with the given namespace. If none is found,
/// then return nullptr.
Dialect *MLIRContext::getRegisteredDialect(StringRef name) {
  // Lock access to the context registry.
  llvm::sys::SmartScopedReader<true> registryLock(getImpl().contextMutex);
  for (auto &dialect : getImpl().dialects)
    if (name == dialect->getNamespace())
      return dialect.get();
  return nullptr;
}

/// Register this dialect object with the specified context.  The context
/// takes ownership of the heap allocated dialect.
void Dialect::registerDialect(MLIRContext *context) {
  auto &impl = context->getImpl();

  // Lock access to the context registry.
  llvm::sys::SmartScopedWriter<true> registryLock(impl.contextMutex);
  // Abort if dialect with namespace has already been registered.
  if (llvm::any_of(impl.dialects, [this](std::unique_ptr<Dialect> &dialect) {
        return dialect->getNamespace() == getNamespace();
      })) {
    llvm::report_fatal_error("a dialect with namespace '" +
                             Twine(getNamespace()) +
                             "' has already been registered");
  }
  impl.dialects.push_back(std::unique_ptr<Dialect>(this));
}

/// Return information about all registered operations.  This isn't very
/// efficient, typically you should ask the operations about their properties
/// directly.
std::vector<AbstractOperation *> MLIRContext::getRegisteredOperations() {
  std::vector<std::pair<StringRef, AbstractOperation *>> opsToSort;

  { // Lock access to the context registry.
    llvm::sys::SmartScopedReader<true> registryLock(getImpl().contextMutex);

    // We just have the operations in a non-deterministic hash table order. Dump
    // into a temporary array, then sort it by operation name to get a stable
    // ordering.
    llvm::StringMap<AbstractOperation> &registeredOps =
        getImpl().registeredOperations;

    opsToSort.reserve(registeredOps.size());
    for (auto &elt : registeredOps)
      opsToSort.push_back({elt.first(), &elt.second});
  }

  llvm::array_pod_sort(opsToSort.begin(), opsToSort.end());

  std::vector<AbstractOperation *> result;
  result.reserve(opsToSort.size());
  for (auto &elt : opsToSort)
    result.push_back(elt.second);
  return result;
}

void Dialect::addOperation(AbstractOperation opInfo) {
  assert(opInfo.name.split('.').first == getNamespace() &&
         "op name doesn't start with dialect namespace");
  assert(&opInfo.dialect == this && "Dialect object mismatch");
  auto &impl = context->getImpl();

  // Lock access to the context registry.
  llvm::sys::SmartScopedWriter<true> registryLock(impl.contextMutex);
  if (!impl.registeredOperations.insert({opInfo.name, opInfo}).second) {
    llvm::errs() << "error: operation named '" << opInfo.name
                 << "' is already registered.\n";
    abort();
  }
}

/// Register a dialect-specific symbol(e.g. type) with the current context.
void Dialect::addSymbol(const ClassID *const classID) {
  auto &impl = context->getImpl();

  // Lock access to the context registry.
  llvm::sys::SmartScopedWriter<true> registryLock(impl.contextMutex);
  if (!impl.registeredDialectSymbols.insert({classID, this}).second) {
    llvm::errs() << "error: dialect symbol already registered.\n";
    abort();
  }
}

/// Look up the specified operation in the operation set and return a pointer
/// to it if present.  Otherwise, return a null pointer.
const AbstractOperation *AbstractOperation::lookup(StringRef opName,
                                                   MLIRContext *context) {
  auto &impl = context->getImpl();

  // Lock access to the context registry.
  llvm::sys::SmartScopedReader<true> registryLock(impl.contextMutex);
  auto it = impl.registeredOperations.find(opName);
  if (it != impl.registeredOperations.end())
    return &it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Identifier uniquing
//===----------------------------------------------------------------------===//

/// Return an identifier for the specified string.
Identifier Identifier::get(StringRef str, MLIRContext *context) {
  assert(!str.empty() && "Cannot create an empty identifier");
  assert(str.find('\0') == StringRef::npos &&
         "Cannot create an identifier with a nul character");

  auto &impl = context->getImpl();

  { // Check for an existing identifier in read-only mode.
    llvm::sys::SmartScopedReader<true> contextLock(impl.identifierMutex);
    auto it = impl.identifiers.find(str);
    if (it != impl.identifiers.end())
      return Identifier(it->getKeyData());
  }

  // Aquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> contextLock(impl.identifierMutex);
  auto it = impl.identifiers.insert({str, char()}).first;
  return Identifier(it->getKeyData());
}

//===----------------------------------------------------------------------===//
// Location uniquing
//===----------------------------------------------------------------------===//

UnknownLoc UnknownLoc::get(MLIRContext *context) {
  return &context->getImpl().theUnknownLoc;
}

UniquedFilename UniquedFilename::get(StringRef filename, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Aquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> locationLock(impl.locationMutex);
  auto it = impl.filenames.insert({filename, char()}).first;
  return UniquedFilename(it->getKeyData());
}

FileLineColLoc FileLineColLoc::get(UniquedFilename filename, unsigned line,
                                   unsigned column, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Safely get or create a location instance.
  auto key = std::make_tuple(filename.data(), line, column);
  return safeGetOrCreate(impl.fileLineColLocs, key, impl.locationMutex, [&] {
    return new (impl.locationAllocator.Allocate<FileLineColLocationStorage>())
        FileLineColLocationStorage(filename, line, column);
  });
}

NameLoc NameLoc::get(Identifier name, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Safely get or create a location instance.
  return safeGetOrCreate(impl.nameLocs, name.data(), impl.locationMutex, [&] {
    return new (impl.locationAllocator.Allocate<NameLocationStorage>())
        NameLocationStorage(name);
  });
}

CallSiteLoc CallSiteLoc::get(Location callee, Location caller,
                             MLIRContext *context) {
  auto &impl = context->getImpl();

  // Safely get or create a location instance.
  auto key = std::make_pair(callee, caller);
  return safeGetOrCreate(impl.callLocs, key, impl.locationMutex, [&] {
    return new (impl.locationAllocator.Allocate<CallSiteLocationStorage>())
        CallSiteLocationStorage(callee, caller);
  });
}

Location FusedLoc::get(ArrayRef<Location> locs, Attribute metadata,
                       MLIRContext *context) {
  // Unique the set of locations to be fused.
  llvm::SmallSetVector<Location, 4> decomposedLocs;
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

  // Safely get or create a location instance.
  auto key = std::make_pair(locs, metadata);
  return safeGetOrCreate(impl.fusedLocs, key, impl.locationMutex, [&] {
    auto byteSize =
        FusedLocationStorage::totalSizeToAlloc<Location>(locs.size());
    auto rawMem = impl.locationAllocator.Allocate(
        byteSize, alignof(FusedLocationStorage));
    auto result = new (rawMem) FusedLocationStorage(locs.size(), metadata);

    std::uninitialized_copy(locs.begin(), locs.end(),
                            result->getTrailingObjects<Location>());
    return result;
  });
}

//===----------------------------------------------------------------------===//
// SDBMExpr uniquing
//===----------------------------------------------------------------------===//

SDBMSumExpr SDBMSumExpr::get(SDBMVaryingExpr lhs, SDBMConstantExpr rhs) {
  assert(lhs && "expected SDBM variable expression");
  assert(rhs && "expected SDBM constant");

  MLIRContextImpl &impl = lhs.getContext()->getImpl();

  // If LHS of a sum is another sum, fold the constant RHS parts.
  if (auto lhsSum = lhs.dyn_cast<SDBMSumExpr>()) {
    lhs = lhsSum.getLHS();
    rhs = SDBMConstantExpr::get(rhs.getContext(),
                                rhs.getValue() + lhsSum.getRHS().getValue());
  }

  auto key = std::make_tuple(lhs, rhs);
  return safeGetOrCreate(
      impl.SDBMSumExprs, key, impl.SDBMMutex, [&impl, lhs, rhs] {
        auto *mem = impl.SDBMAllocator.Allocate<SDBMBinaryExprStorage>();
        return new (mem) SDBMBinaryExprStorage(SDBMExprKind::Add,
                                               lhs.getContext(), lhs, rhs);
      });
}

SDBMDiffExpr SDBMDiffExpr::get(SDBMPositiveExpr lhs, SDBMPositiveExpr rhs) {
  assert(lhs && "expected SDBM dimension");
  assert(rhs && "expected SDBM dimension");

  MLIRContextImpl &impl = lhs.getContext()->getImpl();
  auto key = std::make_tuple(lhs, rhs);
  return safeGetOrCreate(
      impl.SDBMDiffExprs, key, impl.SDBMMutex, [&impl, lhs, rhs] {
        auto *mem = impl.SDBMAllocator.Allocate<SDBMDiffExprStorage>();
        return new (mem) SDBMDiffExprStorage(lhs.getContext(), lhs, rhs);
      });
}

SDBMStripeExpr SDBMStripeExpr::get(SDBMPositiveExpr var,
                                   SDBMConstantExpr stripeFactor) {
  assert(var && "expected SDBM variable expression");
  assert(stripeFactor && "expected non-null stripe factor");
  if (stripeFactor.getValue() <= 0)
    llvm::report_fatal_error("non-positive stripe factor");

  MLIRContextImpl &impl = var.getContext()->getImpl();
  auto key = std::make_tuple(var, stripeFactor);
  return safeGetOrCreate(
      impl.SDBMStripeExprs, key, impl.SDBMMutex, [&impl, var, stripeFactor] {
        auto *mem = impl.SDBMAllocator.Allocate<SDBMBinaryExprStorage>();
        return new (mem) SDBMBinaryExprStorage(
            SDBMExprKind::Stripe, var.getContext(), var, stripeFactor);
      });
}

SDBMDimExpr SDBMDimExpr::get(MLIRContext *context, unsigned position) {
  assert(context && "expected non-null context");
  MLIRContextImpl &impl = context->getImpl();
  return safeGetOrCreate(
      impl.SDBMDimExprs, position, impl.SDBMMutex, [&impl, context, position] {
        auto *mem = impl.SDBMAllocator.Allocate<SDBMPositiveExprStorage>();
        return new (mem)
            SDBMPositiveExprStorage(SDBMExprKind::DimId, context, position);
      });
}

SDBMSymbolExpr SDBMSymbolExpr::get(MLIRContext *context, unsigned position) {
  assert(context && "expected non-null context");
  MLIRContextImpl &impl = context->getImpl();
  return safeGetOrCreate(
      impl.SDBMSymbolExprs, position, impl.SDBMMutex,
      [&impl, context, position] {
        auto *mem = impl.SDBMAllocator.Allocate<SDBMPositiveExprStorage>();
        return new (mem)
            SDBMPositiveExprStorage(SDBMExprKind::SymbolId, context, position);
      });
}

SDBMConstantExpr SDBMConstantExpr::get(MLIRContext *context, int64_t value) {
  assert(context && "expected non-null context");
  MLIRContextImpl &impl = context->getImpl();
  return safeGetOrCreate(
      impl.SDBMConstExprs, value, impl.SDBMMutex, [&impl, context, value] {
        auto *mem = impl.SDBMAllocator.Allocate<SDBMConstantExprStorage>();
        return new (mem) SDBMConstantExprStorage(context, value);
      });
}

SDBMNegExpr SDBMNegExpr::get(SDBMPositiveExpr var) {
  assert(var && "expected non-null SDBM variable expression");
  MLIRContextImpl &impl = var.getContext()->getImpl();
  return safeGetOrCreate(impl.SDBMNegExprs, var, impl.SDBMMutex, [&impl, var] {
    auto *mem = impl.SDBMAllocator.Allocate<SDBMNegExprStorage>();
    return new (mem) SDBMNegExprStorage(var);
  });
}

//===----------------------------------------------------------------------===//
// Type uniquing
//===----------------------------------------------------------------------===//

static Dialect &lookupDialectForSymbol(MLIRContext *ctx,
                                       const ClassID *const classID) {
  auto &impl = ctx->getImpl();
  auto it = impl.registeredDialectSymbols.find(classID);
  assert(it != impl.registeredDialectSymbols.end() &&
         "symbol is not registered.");
  return *it->second;
}

/// Returns the storage unqiuer used for constructing type storage instances.
/// This should not be used directly.
StorageUniquer &MLIRContext::getTypeUniquer() { return getImpl().typeUniquer; }

/// Get the dialect that registered the type with the provided typeid.
const Dialect &TypeUniquer::lookupDialectForType(MLIRContext *ctx,
                                                 const ClassID *const typeID) {
  return lookupDialectForSymbol(ctx, typeID);
}

//===----------------------------------------------------------------------===//
// Attribute uniquing
//===----------------------------------------------------------------------===//

/// Returns the storage uniquer used for constructing attribute storage
/// instances. This should not be used directly.
StorageUniquer &MLIRContext::getAttributeUniquer() {
  return getImpl().attributeUniquer;
}

/// Returns a functor used to initialize new attribute storage instances.
std::function<void(AttributeStorage *)>
AttributeUniquer::getInitFn(MLIRContext *ctx, const ClassID *const attrID) {
  return [ctx, attrID](AttributeStorage *storage) {
    storage->initializeDialect(lookupDialectForSymbol(ctx, attrID));

    // If the attribute did not provide a type, then default to NoneType.
    if (!storage->getType())
      storage->setType(NoneType::get(ctx));
  };
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
AttributeListStorage *
AttributeListStorage::get(ArrayRef<NamedAttribute> attrs) {
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

  auto &impl = attrs[0].second.getContext()->getImpl();

  // Safely get or create an attribute instance.
  return safeGetOrCreate(impl.attributeLists, attrs, impl.attributeMutex, [&] {
    auto byteSize =
        AttributeListStorage::totalSizeToAlloc<NamedAttribute>(attrs.size());
    auto rawMem =
        impl.attributeAllocator.Allocate(byteSize, alignof(NamedAttribute));

    //  Placement initialize the AggregateSymbolicValue.
    auto result = ::new (rawMem) AttributeListStorage(attrs.size());
    std::uninitialized_copy(attrs.begin(), attrs.end(),
                            result->getTrailingObjects<NamedAttribute>());
    return result;
  });
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
  auto key = std::make_tuple(dimCount, symbolCount, results, rangeSizes);

  // Safely get or create an AffineMap instance.
  return safeGetOrCreate(impl.affineMaps, key, impl.affineMutex, [&] {
    auto *res = impl.affineAllocator.Allocate<detail::AffineMapStorage>();

    // Copy the results and range sizes into the bump pointer.
    results = copyArrayRefInto(impl.affineAllocator, results);
    rangeSizes = copyArrayRefInto(impl.affineAllocator, rangeSizes);

    // Initialize the memory using placement new.
    new (res)
        detail::AffineMapStorage{dimCount, symbolCount, results, rangeSizes};
    return AffineMap(res);
  });
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

  // Detect and transform "expr - c * (expr floordiv c)" to "expr mod c". This
  // leads to a much more efficient form when 'c' is a power of two, and in
  // general a more compact and readable form.

  // Process '(expr floordiv c) * (-c)'.
  AffineBinaryOpExpr rBinOpExpr = rhs.dyn_cast<AffineBinaryOpExpr>();
  if (!rBinOpExpr)
    return nullptr;

  auto lrhs = rBinOpExpr.getLHS();
  auto rrhs = rBinOpExpr.getRHS();

  // Process lrhs, which is 'expr floordiv c'.
  AffineBinaryOpExpr lrBinOpExpr = lrhs.dyn_cast<AffineBinaryOpExpr>();
  if (!lrBinOpExpr)
    return nullptr;

  auto llrhs = lrBinOpExpr.getLHS();
  auto rlrhs = lrBinOpExpr.getRHS();

  if (lhs == llrhs && rlrhs == -rrhs) {
    return lhs % rlrhs;
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

  { // Check for an existing instance in read-only mode.
    llvm::sys::SmartScopedReader<true> affineLock(impl.affineMutex);
    auto cached = impl.affineExprs.find(keyValue);
    if (cached != impl.affineExprs.end())
      return cached->second;
  }

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

  // Aquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> affineLock(impl.affineMutex);

  // Check for an existing instance again here, because another writer thread
  // may have already created one.
  auto &result = impl.affineExprs.insert({keyValue, nullptr}).first->second;
  if (!result) {
    // An expression with these operands will already be in the
    // simplified/canonical form. Create and store it.
    result = new (impl.affineAllocator.Allocate<AffineBinaryOpExprStorage>())
        AffineBinaryOpExprStorage{{kind, lhs.getContext()}, lhs, rhs};
  }
  return result;
}

AffineExpr mlir::getAffineBinaryOpExpr(AffineExprKind kind, AffineExpr lhs,
                                       AffineExpr rhs) {
  return AffineBinaryOpExprStorage::get(kind, lhs, rhs);
}

AffineExpr mlir::getAffineDimExpr(unsigned position, MLIRContext *context) {
  auto &impl = context->getImpl();

  return safeGetOrCreate(
      impl.dimExprs, position, impl.affineMutex, [&impl, context, position] {
        auto *result = impl.affineAllocator.Allocate<AffineDimExprStorage>();
        // Initialize the memory using placement new.
        new (result)
            AffineDimExprStorage{{AffineExprKind::DimId, context}, position};
        return result;
      });
}

AffineExpr mlir::getAffineSymbolExpr(unsigned position, MLIRContext *context) {
  auto &impl = context->getImpl();

  return safeGetOrCreate(
      impl.symbolExprs, position, impl.affineMutex, [&impl, context, position] {
        auto *result = impl.affineAllocator.Allocate<AffineSymbolExprStorage>();
        // Initialize the memory using placement new.
        new (result) AffineSymbolExprStorage{
            {AffineExprKind::SymbolId, context}, position};
        return result;
      });
}

AffineExpr mlir::getAffineConstantExpr(int64_t constant, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Safely get or create an AffineConstantExpr instance.
  return safeGetOrCreate(impl.constExprs, constant, impl.affineMutex, [&] {
    auto *result = impl.affineAllocator.Allocate<AffineConstantExprStorage>();
    return new (result) AffineConstantExprStorage{
        {AffineExprKind::Constant, context}, constant};
  });
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

  auto &impl = constraints[0].getContext()->getImpl();

  // A utility function to construct a new IntegerSetStorage instance.
  auto constructorFn = [&] {
    auto *res = impl.affineAllocator.Allocate<detail::IntegerSetStorage>();

    // Copy the results and equality flags into the bump pointer.
    constraints = copyArrayRefInto(impl.affineAllocator, constraints);
    eqFlags = copyArrayRefInto(impl.affineAllocator, eqFlags);

    // Initialize the memory using placement new.
    new (res)
        detail::IntegerSetStorage{dimCount, symbolCount, constraints, eqFlags};
    return IntegerSet(res);
  };

  // If this instance is uniqued, then we handle it separately so that multiple
  // threads may simulatenously access existing instances.
  if (constraints.size() < IntegerSet::kUniquingThreshold) {
    auto key = std::make_tuple(dimCount, symbolCount, constraints, eqFlags);
    return safeGetOrCreate(impl.integerSets, key, impl.affineMutex,
                           constructorFn);
  }

  // Otherwise, aquire a writer-lock so that we can safely create the new
  // instance.
  llvm::sys::SmartScopedWriter<true> affineLock(impl.affineMutex);
  return constructorFn();
}
