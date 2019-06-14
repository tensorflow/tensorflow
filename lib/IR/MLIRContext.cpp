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
    addAttributes<AffineMapAttr, ArrayAttr, BoolAttr, DenseElementsAttr,
                  DictionaryAttr, FloatAttr, FunctionAttr, IntegerAttr,
                  IntegerSetAttr, OpaqueAttr, OpaqueElementsAttr,
                  SparseElementsAttr, StringAttr, TypeAttr, UnitAttr>();
    addTypes<ComplexType, FloatType, FunctionType, IndexType, IntegerType,
             MemRefType, NoneType, OpaqueType, RankedTensorType, TupleType,
             UnrankedTensorType, VectorType>();

    // TODO: FuncOp should be moved to a different dialect when it has been
    // fully decoupled from the core.
    addOperations<FuncOp>();
  }
};

struct AffineMapKeyInfo : DenseMapInfo<AffineMap> {
  // Affine maps are uniqued based on their dim/symbol counts and affine
  // expressions.
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>>;
  using DenseMapInfo<AffineMap>::isEqual;

  static unsigned getHashValue(const AffineMap &key) {
    return getHashValue(
        KeyTy(key.getNumDims(), key.getNumSymbols(), key.getResults()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        std::get<0>(key), std::get<1>(key),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()));
  }

  static bool isEqual(const KeyTy &lhs, AffineMap rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs.getNumDims(), rhs.getNumSymbols(),
                                  rhs.getResults());
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

  // Affine expression uniqui'ing.
  StorageUniquer affineUniquer;

  //===--------------------------------------------------------------------===//
  // Type uniquing
  //===--------------------------------------------------------------------===//
  StorageUniquer typeUniquer;

  //===--------------------------------------------------------------------===//
  // Attribute uniquing
  //===--------------------------------------------------------------------===//
  StorageUniquer attributeUniquer;

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
  assert((getNamespace().empty() ||
          opInfo.name.split('.').first == getNamespace()) &&
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

NameLoc NameLoc::get(Identifier name, Location child, MLIRContext *context) {
  auto &impl = context->getImpl();
  assert(!child.isa<NameLoc>() &&
         "a NameLoc cannot be used as a child of another NameLoc");

  // Safely get or create a location instance.
  return safeGetOrCreate(impl.nameLocs, name.data(), impl.locationMutex, [&] {
    return new (impl.locationAllocator.Allocate<NameLocationStorage>())
        NameLocationStorage(name, child);
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
Dialect &TypeUniquer::lookupDialectForType(MLIRContext *ctx,
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

//===----------------------------------------------------------------------===//
// AffineMap uniquing
//===----------------------------------------------------------------------===//

StorageUniquer &MLIRContext::getAffineUniquer() {
  return getImpl().affineUniquer;
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> results) {
  // The number of results can't be zero.
  assert(!results.empty());

  auto &impl = results[0].getContext()->getImpl();
  auto key = std::make_tuple(dimCount, symbolCount, results);

  // Safely get or create an AffineMap instance.
  return safeGetOrCreate(impl.affineMaps, key, impl.affineMutex, [&] {
    auto *res = impl.affineAllocator.Allocate<detail::AffineMapStorage>();

    // Copy the results into the bump pointer.
    results = copyArrayRefInto(impl.affineAllocator, results);

    // Initialize the memory using placement new.
    new (res) detail::AffineMapStorage{dimCount, symbolCount, results};
    return AffineMap(res);
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
