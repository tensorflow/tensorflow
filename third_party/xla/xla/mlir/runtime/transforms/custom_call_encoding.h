/* Copyright 2022 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_MLIR_RUNTIME_TRANSFORMS_CUSTOM_CALL_ENCODING_H_
#define XLA_MLIR_RUNTIME_TRANSFORMS_CUSTOM_CALL_ENCODING_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "xla/runtime/custom_call.h"
#include "xla/runtime/type_id.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Helper classes to build XLA custom calls' lowering to the LLVM dialect.
//===----------------------------------------------------------------------===//
//
// Arguments to the custom call API intrinsic are encoded as an array of opaque
// pointers and at the runtime side available as `void**`. Runtime decodes
// opaque pointers to the C++ data structures (see runtime/custom_call.h), and
// passes them to the registered callback. Argument encoding/decoding must be
// compatible, otherwise it's very easy to get a segfault because of an illegal
// memory access.
//
// Attributes are encoded into a separate opaque storage together with names,
// so the runtime side can decode the attributes it needs and check that all
// required attributes were passed to the custom call handler.
//
// Custom call attributes are encoded as module global constants, and at run
// time we only need to pass a pointer to the constant section.
//
// Custom call arguments are encoded as an array of pointers allocated on the
// stack. Each individual argument is also encoded on the stack, because
// arguments are typically run time values and we can't encode them in the
// constant section. Statically known arguments (constants) can be encoded as
// global values together with attributes.

// Forward declare class declared below.
class Globals;
class Allocas;

//===----------------------------------------------------------------------===//
// Custom call arguments encoding.
//===----------------------------------------------------------------------===//

// Encodes argument into stack allocated storage according to the ABI.
class CustomCallArgEncoding {
 public:
  struct Encoded {
    mlir::LLVM::GlobalOp type_id;  // llvm.mlir.global external $type_name : i64

    // Statically known arguments might be encoded as global constants,
    // otherwise it will be `!llvm.alloca 1 x ArgType`.
    std::variant<mlir::LLVM::AllocaOp, mlir::LLVM::GlobalOp> value;
  };

  virtual ~CustomCallArgEncoding() = default;

  virtual mlir::LogicalResult Match(mlir::Value value,
                                    mlir::Value conterted) const = 0;

  virtual mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                          mlir::ImplicitLocOpBuilder &b,
                                          mlir::Value value,
                                          mlir::Value converted) const = 0;
};

// A set of registered custom call arguments encodings.
class CustomCallArgEncodingSet {
 public:
  using Encoded = CustomCallArgEncoding::Encoded;

  // Finds matching argument encoding and tries to encode the values. Returns
  // failure if didn't match values to any of the argument encodings.
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b,
                                  mlir::Value value,
                                  mlir::Value converted) const;

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallArgEncodingSet &Add() {
    (encodings_.emplace_back(std::make_unique<Ts>()), ...);
    return *this;
  }

  template <typename... Ts, typename Arg, typename... Args,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallArgEncodingSet &Add(Arg arg, Args... args) {
    (encodings_.emplace_back(std::make_unique<Ts>(std::forward<Arg>(arg),
                                                  std::forward<Args...>(args))),
     ...);
    return *this;
  }

 private:
  std::vector<std::unique_ptr<CustomCallArgEncoding>> encodings_;
};

//===----------------------------------------------------------------------===//
// Custom call results encoding.
//===----------------------------------------------------------------------===//

// Encodes result into stack allocated storage according to the ABI.
class CustomCallRetEncoding {
 public:
  struct Encoded {
    mlir::LLVM::GlobalOp type_id;  // llvm.mlir.global external $type_name : i64
    mlir::LLVM::AllocaOp value;    // !llvm.alloca 1 x ResultType
  };

  virtual ~CustomCallRetEncoding() = default;

  virtual mlir::LogicalResult Match(mlir::Type type,
                                    mlir::Type converted) const = 0;

  virtual mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                          mlir::ImplicitLocOpBuilder &b,
                                          mlir::Type type,
                                          mlir::Type converted) const = 0;

  virtual mlir::FailureOr<mlir::Value> Decode(
      mlir::ImplicitLocOpBuilder &b, mlir::Type type, mlir::Type converted,
      mlir::LLVM::AllocaOp alloca) const = 0;
};

// A set of registered custom call results encodings.
class CustomCallRetEncodingSet {
 public:
  using Encoded = CustomCallRetEncoding::Encoded;

  // Finds matching result encoding and tries to encode the values. Returns
  // failure if didn't match values to any of the result encodings.
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b,
                                  mlir::Type type, mlir::Type converted) const;

  // Convert the encoded value in alloca back to a value with the converted
  // type. Return failure if the convertion failed.
  mlir::FailureOr<mlir::Value> Decode(mlir::ImplicitLocOpBuilder &b,
                                      mlir::Type type, mlir::Type converted,
                                      mlir::LLVM::AllocaOp alloca) const;

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallRetEncodingSet &Add() {
    (encodings_.emplace_back(std::make_unique<Ts>()), ...);
    return *this;
  }

  template <typename... Ts, typename Arg, typename... Args,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallRetEncodingSet &Add(Arg arg, Args... args) {
    (encodings_.emplace_back(std::make_unique<Ts>(std::forward<Arg>(arg),
                                                  std::forward<Args...>(args))),
     ...);
    return *this;
  }

 private:
  std::vector<std::unique_ptr<CustomCallRetEncoding>> encodings_;
};

//===----------------------------------------------------------------------===//
// Custom call attributes encoding.
//===----------------------------------------------------------------------===//

// Attributes encoding packs attribute name, data type and a value into the
// module global constant, and returns values pointing to the encoded data.
struct CustomCallAttrEncoding {
  static constexpr char kAttrName[] = "__rt_attr_name";
  static constexpr char kAttrValue[] = "__rt_attr_value";

  struct Encoded {
    mlir::LLVM::GlobalOp name;     // llvm.mlir.global <encoded-name>
    mlir::LLVM::GlobalOp type_id;  // llvm.mlir.global external $type_name : i64
    mlir::LLVM::GlobalOp value;    // llvm.mlir.global <encoded-attribute>
  };

  virtual ~CustomCallAttrEncoding() = default;

  virtual mlir::LogicalResult Match(mlir::SymbolTable &sym_table,
                                    std::string_view name,
                                    mlir::Attribute attr) const = 0;

  virtual mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &sym_table,
                                          Globals &g,
                                          mlir::ImplicitLocOpBuilder &b,
                                          std::string_view name,
                                          mlir::Attribute attr) const = 0;
};

// A set of registered custom call attributes encodings.
class CustomCallAttrEncodingSet {
 public:
  using Encoded = CustomCallAttrEncoding::Encoded;

  // Finds matching attribute encoding and tries to encode the attribute.
  // Returns failure if didn't match attribute to any of the encodings.
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &sym_table, Globals &g,
                                  mlir::ImplicitLocOpBuilder &b,
                                  std::string_view name,
                                  mlir::Attribute attr) const;

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallAttrEncodingSet &Add() {
    (encodings_.emplace_back(std::make_unique<Ts>()), ...);
    return *this;
  }

  template <typename... Ts, typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallAttrEncodingSet &Add(ConstructorArg &&arg,
                                 ConstructorArgs &&...args) {
    (encodings_.emplace_back(std::make_unique<Ts>(arg, args...)), ...);
    return *this;
  }

 private:
  std::vector<std::unique_ptr<CustomCallAttrEncoding>> encodings_;
};

//===----------------------------------------------------------------------===//
// A set of helper functions for packing encoding attributes.
//===----------------------------------------------------------------------===//

// Encodes type id as an external LLVM global of type `i64`. The global name is
// defined by the type id name registry. Internally type id implemented as an
// opaque pointer (void*), and type equality check at run time is just a pointer
// comparison. All type id symbols at run time must be resolved to the type id
// instances defined in the current process.
mlir::LLVM::GlobalOp EncodeTypeId(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  TypeID type_id);

// Encodes string as a module global null-terminated string constant + size. We
// reuse the encoding scheme for arrays to store sting with its size, to avoid
// computing the length of the null-terminated string at run time.
mlir::LLVM::GlobalOp EncodeString(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  std::string_view strref,
                                  std::string_view symbol_base);

// Encodes scalar attribute as a global constant.
mlir::LLVM::GlobalOp EncodeScalar(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::Attribute value,
                                  std::string_view symbol_base);

//===----------------------------------------------------------------------===//
// A helper class to create global constants in the module.
//===----------------------------------------------------------------------===//

class Globals {
 public:
  // Global value initializer that build the initialization region.
  using GlobalInitializer =
      std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Attribute)>;

  // Global value initializer that can return failure if it can't initialize the
  // global value from the given attribute.
  using FailureOrGlobalInitializer = std::function<mlir::LogicalResult(
      mlir::ImplicitLocOpBuilder &, mlir::Attribute)>;

  Globals(mlir::ModuleOp module, TypeIDNameRegistry type_id_names)
      : module_(module),
        sym_table_(module_),
        type_id_names_(std::move(type_id_names)) {}

  // Creates a global external variable for the type id.
  mlir::LLVM::GlobalOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   TypeID type_id);

  // Creates a global null-terminated string constant.
  mlir::LLVM::GlobalOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   std::string_view strref,
                                   std::string_view symbol_base);

  // Creates a global constant value from the attribute. Attribute type must be
  // a valid type compatible with LLVM globals.
  mlir::LLVM::GlobalOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   mlir::TypedAttr attr,
                                   std::string_view symbol_base);

  // Creates a global constant value of the given type from the attribute, using
  // optional user-provided global constant initialization.
  mlir::LLVM::GlobalOp GetOrCreate(
      mlir::ImplicitLocOpBuilder &b, mlir::Attribute attr, mlir::Type type,
      std::string_view symbol_base, GlobalInitializer initialize = {},
      mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::Internal);

  // Creates a global constant value of the given type from the attribute, using
  // optional user-provided global constant initialization. Returns failure if
  // user-provided initialization failed to initialize the global value.
  mlir::FailureOr<mlir::LLVM::GlobalOp> TryGetOrCreate(
      mlir::ImplicitLocOpBuilder &b, mlir::Attribute attr, mlir::Type type,
      std::string_view symbol_base, FailureOrGlobalInitializer initialize = {},
      mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::Internal);

  // Returns the address of the global value.
  static mlir::Value AddrOf(mlir::ImplicitLocOpBuilder &b,
                            mlir::LLVM::GlobalOp global);

  mlir::ModuleOp module() { return module_; }

 private:
  // Globals key: {attribute, encoded-type, sym-name}. We can only have global
  // constants of one of the LLVM types, and there could be multiple ways to
  // encode an attribute as an LLVM type, e.g. strings can be stored as null
  // terminated array of bytes, or a pair of string size and and array of bytes.
  using Key = std::tuple<mlir::Attribute, mlir::Type, mlir::StringAttr>;

  mlir::LLVM::GlobalOp Find(Key key);

  mlir::ModuleOp module_;
  mlir::SymbolTable sym_table_;  // symbol table for the `module_`
  llvm::DenseMap<Key, mlir::LLVM::GlobalOp> globals_;

  // A mapping from the TypeID to the unique type name for encoding external
  // globals corresponding to types ids.
  TypeIDNameRegistry type_id_names_;
};

//===----------------------------------------------------------------------===//
// A helper class to create alloca operations for encoded arguments.
//===----------------------------------------------------------------------===//

class EncodingAllocas;

// We reuse allocas for encoding custom call arguments and results, because we
// potentially can have thousands of custom calls, and we do not want to
// accidentally blow up the stack size. It means that we might encode the same
// argument multiple times, but encoding is cheap (few store operations), and
// LLVM can potentially optimize them away.
//
// TODO(ezhulenev): Use `llvm.invariant.start` and `llvm.invariant.end` to mark
// encoded arguments allocas.
class Allocas {
 public:
  ~Allocas();

  mlir::LLVM::AllocaOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   mlir::Type type);

 private:
  friend class EncodingAllocas;

  struct TypedAllocas {
    size_t offset = 0;
    llvm::SmallVector<mlir::LLVM::AllocaOp> allocas;
  };

  explicit Allocas(mlir::Block *block,
                   llvm::DenseMap<mlir::Type, TypedAllocas> *allocas);

  mlir::Block *block_;
  llvm::DenseMap<mlir::Type, TypedAllocas> *allocas_;
};

// Mapping from basic block to allocas.
class EncodingAllocas {
 public:
  Allocas GetForOperation(mlir::Operation *op);

 private:
  friend class Allocas;

  llvm::DenseMap<mlir::Block *,
                 llvm::DenseMap<mlir::Type, Allocas::TypedAllocas>>
      allocas_;
};

//===----------------------------------------------------------------------===//
// Custom call attributes encoding.
//===----------------------------------------------------------------------===//

// Encodes attribute using a scheme compatible with run time attributes decoding
// (see `internal::DecodedAttrs` in the custom call header file).
//
// Returns a value of `!llvm.ptr<ptr<i8>>` (void**) type pointing to the encoded
// attributes array (array of pointers).
//
// This function is used to encode:
//
//   1. Struct attributes as aggregates of nested attributes, where the order of
//      attributes matches the order defined with the `AggregateAttrDef` schema
//      defined below.
//
//   2. Custom call attributes, where the attributes sorted lexicographically by
//      name, to be able to efficiently decode named attributes.
//
mlir::FailureOr<mlir::LLVM::GlobalOp> EncodeAttributes(
    mlir::SymbolTable &sym_table, Globals &g, mlir::ImplicitLocOpBuilder &b,
    const CustomCallAttrEncodingSet &encoding, std::string_view symbol_base,
    llvm::ArrayRef<mlir::NamedAttribute> attrs);

struct StringAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct ScalarAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &g,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct DenseElementsAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct ArrayAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct DenseArrayAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct EmptyArrayAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct SymbolRefAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct UnitAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

struct DictionaryAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &,
                                  mlir::ImplicitLocOpBuilder &,
                                  std::string_view,
                                  mlir::Attribute) const final;
};

// Custom call attribute encoding that encodes enums using their underlying
// scalar type. Type id is based on the enum type passed to the runtime.
//
// This encoding can convert enum types defined in the compiler (e.g. dialect
// enums defined in MLIR) to the enum types used at run time.
template <typename AttrType, typename EnumType,
          typename RuntimeEnumType = EnumType>
struct EnumAttrEncoding : public CustomCallAttrEncoding {
  static_assert(std::is_enum<RuntimeEnumType>::value, "must be an enum class");

  // Convert from the compile time enum to the run time enum.
  using Converter = std::function<RuntimeEnumType(EnumType)>;

  EnumAttrEncoding() {
    static_assert(std::is_same<EnumType, RuntimeEnumType>::value,
                  "requires enum converter");
    convert = [](EnumType value) { return value; };
  }

  explicit EnumAttrEncoding(Converter convert) : convert(std::move(convert)) {}

  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute attr) const final {
    return mlir::success(attr.isa<AttrType>());
  }

  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &, Globals &g,
                                  mlir::ImplicitLocOpBuilder &b,
                                  std::string_view name,
                                  mlir::Attribute attr) const final {
    // Convert enum underlying integral value to an attribute.
    EnumType compile_time_enum = attr.cast<AttrType>().getValue();
    RuntimeEnumType run_time_enum = convert(compile_time_enum);

    using T = std::underlying_type_t<RuntimeEnumType>;
    T underlying_value = static_cast<T>(run_time_enum);

    TypeID type_id = TypeID::get<Tagged<RuntimeEnumType>>();
    mlir::Attribute underlying_attr = AsAttr(b, underlying_value);

    Encoded encoded;
    encoded.name = EncodeString(g, b, name, kAttrName);
    encoded.type_id = EncodeTypeId(g, b, type_id);
    encoded.value = EncodeScalar(g, b, underlying_attr, kAttrValue);

    return encoded;
  }

  static mlir::Attribute AsAttr(mlir::ImplicitLocOpBuilder &b, uint32_t value) {
    return b.getI32IntegerAttr(value);
  }

  Converter convert;
};

// A helper type to define `AttrType` encoding scheme.
template <typename AttrType>
struct AggregateAttrDef {
  template <typename T>
  using Extract = T (AttrType::*)() const;

  template <typename T, typename Attr = mlir::Attribute>
  using Encode = Attr (mlir::Builder::*)(T);

  template <typename T, typename U, typename Attr = mlir::Attribute>
  AggregateAttrDef &Add(std::string name, Extract<T> extract,
                        Encode<U, Attr> encode) {
    bindings.emplace_back([=](AttrType attr, mlir::Builder &b) {
      auto encoded = std::invoke(encode, b, std::invoke(extract, attr));
      return mlir::NamedAttribute(b.getStringAttr(name), encoded);
    });
    return *this;
  }

  AggregateAttrDef &Add(std::string name, Extract<bool> extract) {
    return Add(name, extract, &mlir::Builder::getBoolAttr);
  }

  AggregateAttrDef &Add(std::string name, Extract<int64_t> extract) {
    return Add(name, extract, &mlir::Builder::getI64IntegerAttr);
  }

  AggregateAttrDef &Add(std::string name, Extract<llvm::StringRef> extract) {
    return Add(name, extract, &mlir::Builder::getStringAttr);
  }

  AggregateAttrDef &Add(std::string name,
                        Extract<llvm::ArrayRef<int64_t>> extract) {
    return Add(name, extract, &mlir::Builder::getI64TensorAttr);
  }

  // A list of functions to destruct `AttrType` attribute into the aggregate
  // attributes that will be used for encoding.
  using Bind = std::function<mlir::NamedAttribute(AttrType, mlir::Builder &)>;
  llvm::SmallVector<Bind> bindings;
};

// Custom call attribute encoding for the user-defined attributes which encodes
// them as an aggregate of primitive attributes. It uses the encoding scheme
// compatible with the custom call attributes decoding.
template <typename AttrType, typename RuntimeType = AttrType>
struct AggregateAttrEncoding : public CustomCallAttrEncoding {
  using AttrDef = AggregateAttrDef<AttrType>;

  AggregateAttrEncoding(const CustomCallAttrEncodingSet &encoding,
                        AttrDef attrdef)
      : encoding(encoding), attrdef(std::move(attrdef)) {}

  mlir::LogicalResult Match(mlir::SymbolTable &, std::string_view,
                            mlir::Attribute attr) const final {
    return mlir::success(attr.isa<AttrType>());
  }

  mlir::FailureOr<Encoded> Encode(mlir::SymbolTable &sym_table, Globals &g,
                                  mlir::ImplicitLocOpBuilder &b,
                                  std::string_view name,
                                  mlir::Attribute attr) const final {
    // Extract aggregate attributes from the user-defined attributes.
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    for (auto &bind : attrdef.bindings)
      attrs.emplace_back(bind(attr.cast<AttrType>(), b));

    // Encode extracted attributes as an aggregate.
    auto type_id = TypeID::get<Tagged<RuntimeType>>();
    auto sym = "__rt_aggregate_" + AttrType::getMnemonic();
    auto aggregate =
        EncodeAttributes(sym_table, g, b, encoding, sym.str(), attrs);
    if (mlir::failed(aggregate)) return mlir::failure();

    Encoded encoded;
    encoded.name = EncodeString(g, b, name, kAttrName);
    encoded.type_id = EncodeTypeId(g, b, type_id);
    encoded.value = *aggregate;
    return encoded;
  }

  const CustomCallAttrEncodingSet &encoding;
  AttrDef attrdef;
};

//===----------------------------------------------------------------------===//
// Custom call arguments encoding.
//===----------------------------------------------------------------------===//

// Encodes scalar arguments.
class ScalarArgEncoding : public CustomCallArgEncoding {
 public:
  mlir::LogicalResult Match(mlir::Value, mlir::Value) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Value,
                                  mlir::Value) const final;
};

// Encodes custom call arguments passed as an opaque LLVM pointer (!llvm.ptr)
// using a custom type id. Default constructed encoding encodes `!rt.opaque`
// arguments using a `void*` type id.
class OpaqueArgEncoding : public CustomCallArgEncoding {
 public:
  OpaqueArgEncoding();  // encodes `!rt.opaque` with `void*` type id
  OpaqueArgEncoding(std::function<bool(mlir::Value)> match, TypeID type_id);

  mlir::LogicalResult Match(mlir::Value, mlir::Value) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Value,
                                  mlir::Value) const final;

  template <typename T>
  static auto Match() {
    return [](mlir::Value value) { return value.getType().isa<T>(); };
  }

 private:
  std::function<bool(mlir::Value)> match_;
  TypeID type_id_;
};

// Encodes MemRef arguments according to the (Strided)MemrefView ABI.
class MemrefArgEncoding : public CustomCallArgEncoding {
 public:
  mlir::LogicalResult Match(mlir::Value, mlir::Value) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Value,
                                  mlir::Value) const final;
};

//===----------------------------------------------------------------------===//
// Custom call results encoding.
//===----------------------------------------------------------------------===//

// Encodes scalar operands.
class ScalarRetEncoding : public CustomCallRetEncoding {
 public:
  mlir::LogicalResult Match(mlir::Type, mlir::Type) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                  mlir::Type) const final;
  mlir::FailureOr<mlir::Value> Decode(mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                      mlir::Type,
                                      mlir::LLVM::AllocaOp) const final;
};

// Encodes custom call results returned as an opaque LLVM pointer (!llvm.ptr)
// using a custom type id. Default constructed encoding encodes `!rt.opaque`
// results using a `void*` type id.
class OpaqueRetEncoding : public CustomCallRetEncoding {
 public:
  OpaqueRetEncoding();  // encodes `!rt.opaque` with `void*` type id
  OpaqueRetEncoding(std::function<bool(mlir::Type)> match, TypeID type_id);

  mlir::LogicalResult Match(mlir::Type, mlir::Type) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                  mlir::Type) const final;
  mlir::FailureOr<mlir::Value> Decode(mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                      mlir::Type,
                                      mlir::LLVM::AllocaOp) const final;

  template <typename T>
  static auto Match() {
    return [](mlir::Type type) { return type.isa<T>(); };
  }

 private:
  std::function<bool(mlir::Type)> match_;
  TypeID type_id_;
};

// Encodes MemRef results according to the MemrefView ABI.
class MemrefRetEncoding : public CustomCallRetEncoding {
 public:
  mlir::LogicalResult Match(mlir::Type, mlir::Type) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                  mlir::Type) const final;
  mlir::FailureOr<mlir::Value> Decode(mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                      mlir::Type,
                                      mlir::LLVM::AllocaOp) const final;
};

class AsyncValueRetEncoding : public CustomCallRetEncoding {
 public:
  mlir::LogicalResult Match(mlir::Type, mlir::Type) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, Allocas &a,
                                  mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                  mlir::Type) const final;
  mlir::FailureOr<mlir::Value> Decode(mlir::ImplicitLocOpBuilder &b, mlir::Type,
                                      mlir::Type,
                                      mlir::LLVM::AllocaOp) const final;
};

//===----------------------------------------------------------------------===//
// Default encodings for arguments, attributes and results.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Use `Populate...` functions for adding default encodings.

CustomCallArgEncodingSet DefaultArgEncodings();
CustomCallAttrEncodingSet DefaultAttrEncodings();
CustomCallRetEncodingSet DefaultRetEncodings();

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_TRANSFORMS_CUSTOM_CALL_ENCODING_H_
