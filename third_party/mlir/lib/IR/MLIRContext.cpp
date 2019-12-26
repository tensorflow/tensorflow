//===- MLIRContext.cpp - MLIR Type Classes --------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
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

  // Acquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> instanceLock(mutex);

  // Check for an existing instance again here, because another writer thread
  // may have already created one.
  auto existing = container.insert_as(ValueT(), key);
  if (!existing.second)
    return *existing.first;

  // Otherwise, construct a new instance of the value.
  return *existing.first = constructorFn();
}

namespace {
/// A builtin dialect to define types/etc that are necessary for the validity of
/// the IR.
struct BuiltinDialect : public Dialect {
  BuiltinDialect(MLIRContext *context) : Dialect(/*name=*/"", context) {
    addAttributes<AffineMapAttr, ArrayAttr, BoolAttr, DenseElementsAttr,
                  DictionaryAttr, FloatAttr, SymbolRefAttr, IntegerAttr,
                  IntegerSetAttr, OpaqueAttr, OpaqueElementsAttr,
                  SparseElementsAttr, StringAttr, TypeAttr, UnitAttr>();
    addAttributes<CallSiteLoc, FileLineColLoc, FusedLoc, NameLoc, OpaqueLoc,
                  UnknownLoc>();

    addTypes<ComplexType, FloatType, FunctionType, IndexType, IntegerType,
             MemRefType, UnrankedMemRefType, NoneType, OpaqueType,
             RankedTensorType, TupleType, UnrankedTensorType, VectorType>();

    // TODO: These operations should be moved to a different dialect when they
    // have been fully decoupled from the core.
    addOperations<FuncOp, ModuleOp, ModuleTerminatorOp>();
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
} // end anonymous namespace.

namespace mlir {
/// This is the implementation of the MLIRContext class, using the pImpl idiom.
/// This class is completely private to this file, so everything is public.
class MLIRContextImpl {
public:
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

  // Affine expression uniquing.
  StorageUniquer affineUniquer;

  //===--------------------------------------------------------------------===//
  // Type uniquing
  //===--------------------------------------------------------------------===//
  StorageUniquer typeUniquer;

  /// Cached Type Instances.
  FloatType bf16Ty, f16Ty, f32Ty, f64Ty;
  IndexType indexTy;
  IntegerType int1Ty, int8Ty, int16Ty, int32Ty, int64Ty, int128Ty;
  NoneType noneType;

  //===--------------------------------------------------------------------===//
  // Attribute uniquing
  //===--------------------------------------------------------------------===//
  StorageUniquer attributeUniquer;

  /// Cached Attribute Instances.
  BoolAttr falseAttr, trueAttr;
  UnitAttr unitAttr;
  UnknownLoc unknownLocAttr;

public:
  MLIRContextImpl() : identifiers(identifierAllocator) {}
};
} // end namespace mlir

MLIRContext::MLIRContext() : impl(new MLIRContextImpl()) {
  new BuiltinDialect(this);
  registerAllDialects(this);

  // Initialize several common attributes and types to avoid the need to lock
  // the context when accessing them.

  //// Types.
  /// Floating-point Types.
  impl->bf16Ty = TypeUniquer::get<FloatType>(this, StandardTypes::BF16);
  impl->f16Ty = TypeUniquer::get<FloatType>(this, StandardTypes::F16);
  impl->f32Ty = TypeUniquer::get<FloatType>(this, StandardTypes::F32);
  impl->f64Ty = TypeUniquer::get<FloatType>(this, StandardTypes::F64);
  /// Index Type.
  impl->indexTy = TypeUniquer::get<IndexType>(this, StandardTypes::Index);
  /// Integer Types.
  impl->int1Ty = TypeUniquer::get<IntegerType>(this, StandardTypes::Integer, 1);
  impl->int8Ty = TypeUniquer::get<IntegerType>(this, StandardTypes::Integer, 8);
  impl->int16Ty =
      TypeUniquer::get<IntegerType>(this, StandardTypes::Integer, 16);
  impl->int32Ty =
      TypeUniquer::get<IntegerType>(this, StandardTypes::Integer, 32);
  impl->int64Ty =
      TypeUniquer::get<IntegerType>(this, StandardTypes::Integer, 64);
  impl->int128Ty =
      TypeUniquer::get<IntegerType>(this, StandardTypes::Integer, 128);
  /// None Type.
  impl->noneType = TypeUniquer::get<NoneType>(this, StandardTypes::None);

  //// Attributes.
  //// Note: These must be registered after the types as they may generate one
  //// of the above types internally.
  /// Bool Attributes.
  // Note: The context is also used within the BoolAttrStorage.
  impl->falseAttr = AttributeUniquer::get<BoolAttr>(
      this, StandardAttributes::Bool, this, false);
  impl->trueAttr = AttributeUniquer::get<BoolAttr>(
      this, StandardAttributes::Bool, this, true);
  /// Unit Attribute.
  impl->unitAttr =
      AttributeUniquer::get<UnitAttr>(this, StandardAttributes::Unit);
  /// Unknown Location Attribute.
  impl->unknownLocAttr = AttributeUniquer::get<UnknownLoc>(
      this, StandardAttributes::UnknownLocation);
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
  std::unique_ptr<Dialect> dialect(this);

  // Lock access to the context registry.
  llvm::sys::SmartScopedWriter<true> registryLock(impl.contextMutex);

  // Get the correct insertion position sorted by namespace.
  auto insertPt =
      llvm::lower_bound(impl.dialects, dialect,
                        [](const std::unique_ptr<Dialect> &lhs,
                           const std::unique_ptr<Dialect> &rhs) {
                          return lhs->getNamespace() < rhs->getNamespace();
                        });

  // Abort if dialect with namespace has already been registered.
  if (insertPt != impl.dialects.end() &&
      (*insertPt)->getNamespace() == getNamespace()) {
    llvm::report_fatal_error("a dialect with namespace '" + getNamespace() +
                             "' has already been registered");
  }
  impl.dialects.insert(insertPt, std::move(dialect));
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

  // Acquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> contextLock(impl.identifierMutex);
  auto it = impl.identifiers.insert({str, char()}).first;
  return Identifier(it->getKeyData());
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

FloatType FloatType::get(StandardTypes::Kind kind, MLIRContext *context) {
  assert(kindof(kind) && "Not a FP kind.");
  switch (kind) {
  case StandardTypes::BF16:
    return context->getImpl().bf16Ty;
  case StandardTypes::F16:
    return context->getImpl().f16Ty;
  case StandardTypes::F32:
    return context->getImpl().f32Ty;
  case StandardTypes::F64:
    return context->getImpl().f64Ty;
  default:
    llvm_unreachable("unexpected floating-point kind");
  }
}

/// Get an instance of the IndexType.
IndexType IndexType::get(MLIRContext *context) {
  return context->getImpl().indexTy;
}

/// Return an existing integer type instance if one is cached within the
/// context.
static IntegerType getCachedIntegerType(unsigned width, MLIRContext *context) {
  switch (width) {
  case 1:
    return context->getImpl().int1Ty;
  case 8:
    return context->getImpl().int8Ty;
  case 16:
    return context->getImpl().int16Ty;
  case 32:
    return context->getImpl().int32Ty;
  case 64:
    return context->getImpl().int64Ty;
  case 128:
    return context->getImpl().int128Ty;
  default:
    return IntegerType();
  }
}

IntegerType IntegerType::get(unsigned width, MLIRContext *context) {
  if (auto cached = getCachedIntegerType(width, context))
    return cached;
  return Base::get(context, StandardTypes::Integer, width);
}

IntegerType IntegerType::getChecked(unsigned width, MLIRContext *context,
                                    Location location) {
  if (auto cached = getCachedIntegerType(width, context))
    return cached;
  return Base::getChecked(location, context, StandardTypes::Integer, width);
}

/// Get an instance of the NoneType.
NoneType NoneType::get(MLIRContext *context) {
  return context->getImpl().noneType;
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

BoolAttr BoolAttr::get(bool value, MLIRContext *context) {
  return value ? context->getImpl().trueAttr : context->getImpl().falseAttr;
}

UnitAttr UnitAttr::get(MLIRContext *context) {
  return context->getImpl().unitAttr;
}

Location UnknownLoc::get(MLIRContext *context) {
  return context->getImpl().unknownLocAttr;
}

//===----------------------------------------------------------------------===//
// AffineMap uniquing
//===----------------------------------------------------------------------===//

StorageUniquer &MLIRContext::getAffineUniquer() {
  return getImpl().affineUniquer;
}

AffineMap AffineMap::getImpl(unsigned dimCount, unsigned symbolCount,
                             ArrayRef<AffineExpr> results,
                             MLIRContext *context) {
  auto &impl = context->getImpl();
  auto key = std::make_tuple(dimCount, symbolCount, results);

  // Safely get or create an AffineMap instance.
  return safeGetOrCreate(impl.affineMaps, key, impl.affineMutex, [&] {
    auto *res = impl.affineAllocator.Allocate<detail::AffineMapStorage>();

    // Copy the results into the bump pointer.
    results = copyArrayRefInto(impl.affineAllocator, results);

    // Initialize the memory using placement new.
    new (res) detail::AffineMapStorage{dimCount, symbolCount, results, context};
    return AffineMap(res);
  });
}

AffineMap AffineMap::get(MLIRContext *context) {
  return getImpl(/*dimCount=*/0, /*symbolCount=*/0, /*results=*/{}, context);
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> results) {
  // The number of results can't be zero.
  assert(!results.empty());
  return getImpl(dimCount, symbolCount, results, results[0].getContext());
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
  // threads may simultaneously access existing instances.
  if (constraints.size() < IntegerSet::kUniquingThreshold) {
    auto key = std::make_tuple(dimCount, symbolCount, constraints, eqFlags);
    return safeGetOrCreate(impl.integerSets, key, impl.affineMutex,
                           constructorFn);
  }

  // Otherwise, acquire a writer-lock so that we can safely create the new
  // instance.
  llvm::sys::SmartScopedWriter<true> affineLock(impl.affineMutex);
  return constructorFn();
}
