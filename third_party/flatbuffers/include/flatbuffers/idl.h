/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_IDL_H_
#define FLATBUFFERS_IDL_H_

#include <map>
#include <memory>
#include <stack>

#include "flatbuffers/base.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/hash.h"
#include "flatbuffers/reflection.h"

#if !defined(FLATBUFFERS_CPP98_STL)
#  include <functional>
#endif  // !defined(FLATBUFFERS_CPP98_STL)

// This file defines the data types representing a parsed IDL (Interface
// Definition Language) / schema file.

// Limits maximum depth of nested objects.
// Prevents stack overflow while parse flatbuffers or json.
#if !defined(FLATBUFFERS_MAX_PARSING_DEPTH)
#  define FLATBUFFERS_MAX_PARSING_DEPTH 64
#endif

namespace flatbuffers {

// The order of these matters for Is*() functions below.
// Additionally, Parser::ParseType assumes bool..string is a contiguous range
// of type tokens.
// clang-format off
#define FLATBUFFERS_GEN_TYPES_SCALAR(TD) \
  TD(NONE,   "",       uint8_t,  byte,   byte,    byte,   uint8,   u8,   UByte) \
  TD(UTYPE,  "",       uint8_t,  byte,   byte,    byte,   uint8,   u8,   UByte) /* begin scalar/int */ \
  TD(BOOL,   "bool",   uint8_t,  boolean,bool,    bool,   bool,    bool, Boolean) \
  TD(CHAR,   "byte",   int8_t,   byte,   int8,    sbyte,  int8,    i8,   Byte) \
  TD(UCHAR,  "ubyte",  uint8_t,  byte,   byte,    byte,   uint8,   u8,   UByte) \
  TD(SHORT,  "short",  int16_t,  short,  int16,   short,  int16,   i16,  Short) \
  TD(USHORT, "ushort", uint16_t, short,  uint16,  ushort, uint16,  u16,  UShort) \
  TD(INT,    "int",    int32_t,  int,    int32,   int,    int32,   i32,  Int) \
  TD(UINT,   "uint",   uint32_t, int,    uint32,  uint,   uint32,  u32,  UInt) \
  TD(LONG,   "long",   int64_t,  long,   int64,   long,   int64,   i64,  Long) \
  TD(ULONG,  "ulong",  uint64_t, long,   uint64,  ulong,  uint64,  u64,  ULong) /* end int */ \
  TD(FLOAT,  "float",  float,    float,  float32, float,  float32, f32,  Float) /* begin float */ \
  TD(DOUBLE, "double", double,   double, float64, double, float64, f64,  Double) /* end float/scalar */
#define FLATBUFFERS_GEN_TYPES_POINTER(TD) \
  TD(STRING, "string", Offset<void>, int, int, StringOffset, int, unused, Int) \
  TD(VECTOR, "",       Offset<void>, int, int, VectorOffset, int, unused, Int) \
  TD(STRUCT, "",       Offset<void>, int, int, int,          int, unused, Int) \
  TD(UNION,  "",       Offset<void>, int, int, int,          int, unused, Int)
#define FLATBUFFERS_GEN_TYPE_ARRAY(TD) \
  TD(ARRAY,  "",       int,          int, int, int,          int, unused, Int)
// The fields are:
// - enum
// - FlatBuffers schema type.
// - C++ type.
// - Java type.
// - Go type.
// - C# / .Net type.
// - Python type.
// - Rust type.
// - Kotlin type.

// using these macros, we can now write code dealing with types just once, e.g.

/*
switch (type) {
  #define FLATBUFFERS_TD(ENUM, IDLTYPE, CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, \
                         RTYPE, KTYPE) \
    case BASE_TYPE_ ## ENUM: \
      // do something specific to CTYPE here
    FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
  #undef FLATBUFFERS_TD
}
*/

#define FLATBUFFERS_GEN_TYPES(TD) \
        FLATBUFFERS_GEN_TYPES_SCALAR(TD) \
        FLATBUFFERS_GEN_TYPES_POINTER(TD) \
        FLATBUFFERS_GEN_TYPE_ARRAY(TD)

// Create an enum for all the types above.
#ifdef __GNUC__
__extension__  // Stop GCC complaining about trailing comma with -Wpendantic.
#endif
enum BaseType {
  #define FLATBUFFERS_TD(ENUM, IDLTYPE, CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, \
                         RTYPE, KTYPE) \
      BASE_TYPE_ ## ENUM,
    FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
  #undef FLATBUFFERS_TD
};

#define FLATBUFFERS_TD(ENUM, IDLTYPE, CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, \
                       RTYPE, KTYPE) \
    static_assert(sizeof(CTYPE) <= sizeof(largest_scalar_t), \
                  "define largest_scalar_t as " #CTYPE);
  FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
#undef FLATBUFFERS_TD

inline bool IsScalar (BaseType t) { return t >= BASE_TYPE_UTYPE &&
                                           t <= BASE_TYPE_DOUBLE; }
inline bool IsInteger(BaseType t) { return t >= BASE_TYPE_UTYPE &&
                                           t <= BASE_TYPE_ULONG; }
inline bool IsFloat  (BaseType t) { return t == BASE_TYPE_FLOAT ||
                                           t == BASE_TYPE_DOUBLE; }
inline bool IsLong   (BaseType t) { return t == BASE_TYPE_LONG ||
                                           t == BASE_TYPE_ULONG; }
inline bool IsBool   (BaseType t) { return t == BASE_TYPE_BOOL; }
inline bool IsOneByte(BaseType t) { return t >= BASE_TYPE_UTYPE &&
                                           t <= BASE_TYPE_UCHAR; }

inline bool IsUnsigned(BaseType t) {
  return (t == BASE_TYPE_UTYPE)  || (t == BASE_TYPE_UCHAR) ||
         (t == BASE_TYPE_USHORT) || (t == BASE_TYPE_UINT)  ||
         (t == BASE_TYPE_ULONG);
}

// clang-format on

extern const char *const kTypeNames[];
extern const char kTypeSizes[];

inline size_t SizeOf(BaseType t) { return kTypeSizes[t]; }

struct StructDef;
struct EnumDef;
class Parser;

// Represents any type in the IDL, which is a combination of the BaseType
// and additional information for vectors/structs_.
struct Type {
  explicit Type(BaseType _base_type = BASE_TYPE_NONE, StructDef *_sd = nullptr,
                EnumDef *_ed = nullptr, uint16_t _fixed_length = 0)
      : base_type(_base_type),
        element(BASE_TYPE_NONE),
        struct_def(_sd),
        enum_def(_ed),
        fixed_length(_fixed_length) {}

  bool operator==(const Type &o) {
    return base_type == o.base_type && element == o.element &&
           struct_def == o.struct_def && enum_def == o.enum_def;
  }

  Type VectorType() const {
    return Type(element, struct_def, enum_def, fixed_length);
  }

  Offset<reflection::Type> Serialize(FlatBufferBuilder *builder) const;

  bool Deserialize(const Parser &parser, const reflection::Type *type);

  BaseType base_type;
  BaseType element;       // only set if t == BASE_TYPE_VECTOR
  StructDef *struct_def;  // only set if t or element == BASE_TYPE_STRUCT
  EnumDef *enum_def;      // set if t == BASE_TYPE_UNION / BASE_TYPE_UTYPE,
                          // or for an integral type derived from an enum.
  uint16_t fixed_length;  // only set if t == BASE_TYPE_ARRAY
};

// Represents a parsed scalar value, it's type, and field offset.
struct Value {
  Value()
      : constant("0"),
        offset(static_cast<voffset_t>(~(static_cast<voffset_t>(0U)))) {}
  Type type;
  std::string constant;
  voffset_t offset;
};

// Helper class that retains the original order of a set of identifiers and
// also provides quick lookup.
template<typename T> class SymbolTable {
 public:
  ~SymbolTable() {
    for (auto it = vec.begin(); it != vec.end(); ++it) { delete *it; }
  }

  bool Add(const std::string &name, T *e) {
    vector_emplace_back(&vec, e);
    auto it = dict.find(name);
    if (it != dict.end()) return true;
    dict[name] = e;
    return false;
  }

  void Move(const std::string &oldname, const std::string &newname) {
    auto it = dict.find(oldname);
    if (it != dict.end()) {
      auto obj = it->second;
      dict.erase(it);
      dict[newname] = obj;
    } else {
      FLATBUFFERS_ASSERT(false);
    }
  }

  T *Lookup(const std::string &name) const {
    auto it = dict.find(name);
    return it == dict.end() ? nullptr : it->second;
  }

 public:
  std::map<std::string, T *> dict;  // quick lookup
  std::vector<T *> vec;             // Used to iterate in order of insertion
};

// A name space, as set in the schema.
struct Namespace {
  Namespace() : from_table(0) {}

  // Given a (potentally unqualified) name, return the "fully qualified" name
  // which has a full namespaced descriptor.
  // With max_components you can request less than the number of components
  // the current namespace has.
  std::string GetFullyQualifiedName(const std::string &name,
                                    size_t max_components = 1000) const;

  std::vector<std::string> components;
  size_t from_table;  // Part of the namespace corresponds to a message/table.
};

inline bool operator<(const Namespace &a, const Namespace &b) {
  size_t min_size = std::min(a.components.size(), b.components.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (a.components[i] != b.components[i])
      return a.components[i] < b.components[i];
  }
  return a.components.size() < b.components.size();
}

// Base class for all definition types (fields, structs_, enums_).
struct Definition {
  Definition()
      : generated(false),
        defined_namespace(nullptr),
        serialized_location(0),
        index(-1),
        refcount(1) {}

  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<reflection::KeyValue>>>
  SerializeAttributes(FlatBufferBuilder *builder, const Parser &parser) const;

  bool DeserializeAttributes(Parser &parser,
                             const Vector<Offset<reflection::KeyValue>> *attrs);

  std::string name;
  std::string file;
  std::vector<std::string> doc_comment;
  SymbolTable<Value> attributes;
  bool generated;  // did we already output code for this definition?
  Namespace *defined_namespace;  // Where it was defined.

  // For use with Serialize()
  uoffset_t serialized_location;
  int index;  // Inside the vector it is stored.
  int refcount;
};

struct FieldDef : public Definition {
  FieldDef()
      : deprecated(false),
        required(false),
        key(false),
        shared(false),
        native_inline(false),
        flexbuffer(false),
        nested_flatbuffer(NULL),
        padding(0) {}

  Offset<reflection::Field> Serialize(FlatBufferBuilder *builder, uint16_t id,
                                      const Parser &parser) const;

  bool Deserialize(Parser &parser, const reflection::Field *field);

  Value value;
  bool deprecated;  // Field is allowed to be present in old data, but can't be.
                    // written in new data nor accessed in new code.
  bool required;    // Field must always be present.
  bool key;         // Field functions as a key for creating sorted vectors.
  bool shared;  // Field will be using string pooling (i.e. CreateSharedString)
                // as default serialization behavior if field is a string.
  bool native_inline;  // Field will be defined inline (instead of as a pointer)
                       // for native tables if field is a struct.
  bool flexbuffer;     // This field contains FlexBuffer data.
  StructDef *nested_flatbuffer;  // This field contains nested FlatBuffer data.
  size_t padding;                // Bytes to always pad after this field.
};

struct StructDef : public Definition {
  StructDef()
      : fixed(false),
        predecl(true),
        sortbysize(true),
        has_key(false),
        minalign(1),
        bytesize(0) {}

  void PadLastField(size_t min_align) {
    auto padding = PaddingBytes(bytesize, min_align);
    bytesize += padding;
    if (fields.vec.size()) fields.vec.back()->padding = padding;
  }

  Offset<reflection::Object> Serialize(FlatBufferBuilder *builder,
                                       const Parser &parser) const;

  bool Deserialize(Parser &parser, const reflection::Object *object);

  SymbolTable<FieldDef> fields;

  bool fixed;       // If it's struct, not a table.
  bool predecl;     // If it's used before it was defined.
  bool sortbysize;  // Whether fields come in the declaration or size order.
  bool has_key;     // It has a key field.
  size_t minalign;  // What the whole object needs to be aligned to.
  size_t bytesize;  // Size if fixed.

  flatbuffers::unique_ptr<std::string> original_location;
};

inline bool IsStruct(const Type &type) {
  return type.base_type == BASE_TYPE_STRUCT && type.struct_def->fixed;
}

inline bool IsVector(const Type &type) {
  return type.base_type == BASE_TYPE_VECTOR;
}

inline bool IsArray(const Type &type) {
  return type.base_type == BASE_TYPE_ARRAY;
}

inline bool IsSeries(const Type &type) {
  return IsVector(type) || IsArray(type);
}

inline bool IsEnum(const Type &type) {
  return type.enum_def != nullptr && IsInteger(type.base_type);
}

inline size_t InlineSize(const Type &type) {
  return IsStruct(type)
             ? type.struct_def->bytesize
             : (IsArray(type)
                    ? InlineSize(type.VectorType()) * type.fixed_length
                    : SizeOf(type.base_type));
}

inline size_t InlineAlignment(const Type &type) {
  if (IsStruct(type)) {
    return type.struct_def->minalign;
  } else if (IsArray(type)) {
    return IsStruct(type.VectorType()) ? type.struct_def->minalign
                                       : SizeOf(type.element);
  } else {
    return SizeOf(type.base_type);
  }
}

struct EnumDef;
struct EnumValBuilder;

struct EnumVal {
  Offset<reflection::EnumVal> Serialize(FlatBufferBuilder *builder, const Parser &parser) const;

  bool Deserialize(const Parser &parser, const reflection::EnumVal *val);

  uint64_t GetAsUInt64() const { return static_cast<uint64_t>(value); }
  int64_t GetAsInt64() const { return value; }
  bool IsZero() const { return 0 == value; }
  bool IsNonZero() const { return !IsZero(); }

  std::string name;
  std::vector<std::string> doc_comment;
  Type union_type;

 private:
  friend EnumDef;
  friend EnumValBuilder;
  friend bool operator==(const EnumVal &lhs, const EnumVal &rhs);

  EnumVal(const std::string &_name, int64_t _val) : name(_name), value(_val) {}
  EnumVal() : value(0) {}

  int64_t value;
};

struct EnumDef : public Definition {
  EnumDef() : is_union(false), uses_multiple_type_instances(false) {}

  Offset<reflection::Enum> Serialize(FlatBufferBuilder *builder,
                                     const Parser &parser) const;

  bool Deserialize(Parser &parser, const reflection::Enum *values);

  template<typename T> void ChangeEnumValue(EnumVal *ev, T new_val);
  void SortByValue();
  void RemoveDuplicates();

  std::string AllFlags() const;
  const EnumVal *MinValue() const;
  const EnumVal *MaxValue() const;
  // Returns the number of integer steps from v1 to v2.
  uint64_t Distance(const EnumVal *v1, const EnumVal *v2) const;
  // Returns the number of integer steps from Min to Max.
  uint64_t Distance() const { return Distance(MinValue(), MaxValue()); }

  EnumVal *ReverseLookup(int64_t enum_idx,
                         bool skip_union_default = false) const;
  EnumVal *FindByValue(const std::string &constant) const;

  std::string ToString(const EnumVal &ev) const {
    return IsUInt64() ? NumToString(ev.GetAsUInt64())
                      : NumToString(ev.GetAsInt64());
  }

  size_t size() const { return vals.vec.size(); }

  const std::vector<EnumVal *> &Vals() const {
    FLATBUFFERS_ASSERT(false == vals.vec.empty());
    return vals.vec;
  }

  const EnumVal *Lookup(const std::string &enum_name) const {
    return vals.Lookup(enum_name);
  }

  bool is_union;
  // Type is a union which uses type aliases where at least one type is
  // available under two different names.
  bool uses_multiple_type_instances;
  Type underlying_type;

 private:
  bool IsUInt64() const {
    return (BASE_TYPE_ULONG == underlying_type.base_type);
  }

  friend EnumValBuilder;
  SymbolTable<EnumVal> vals;
};

inline bool operator==(const EnumVal &lhs, const EnumVal &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const EnumVal &lhs, const EnumVal &rhs) {
  return !(lhs == rhs);
}

inline bool EqualByName(const Type &a, const Type &b) {
  return a.base_type == b.base_type && a.element == b.element &&
         (a.struct_def == b.struct_def ||
          a.struct_def->name == b.struct_def->name) &&
         (a.enum_def == b.enum_def || a.enum_def->name == b.enum_def->name);
}

struct RPCCall : public Definition {
  Offset<reflection::RPCCall> Serialize(FlatBufferBuilder *builder, const Parser &parser) const;

  bool Deserialize(Parser &parser, const reflection::RPCCall *call);

  StructDef *request, *response;
};

struct ServiceDef : public Definition {
  Offset<reflection::Service> Serialize(FlatBufferBuilder *builder, const Parser &parser) const;
  bool Deserialize(Parser &parser, const reflection::Service *service);

  SymbolTable<RPCCall> calls;
};

// Container of options that may apply to any of the source/text generators.
struct IDLOptions {
  bool strict_json;
  bool skip_js_exports;
  bool use_goog_js_export_format;
  bool use_ES6_js_export_format;
  bool output_default_scalars_in_json;
  int indent_step;
  bool output_enum_identifiers;
  bool prefixed_enums;
  bool scoped_enums;
  bool include_dependence_headers;
  bool mutable_buffer;
  bool one_file;
  bool proto_mode;
  bool proto_oneof_union;
  bool generate_all;
  bool skip_unexpected_fields_in_json;
  bool generate_name_strings;
  bool generate_object_based_api;
  bool gen_compare;
  std::string cpp_object_api_pointer_type;
  std::string cpp_object_api_string_type;
  bool cpp_object_api_string_flexible_constructor;
  bool gen_nullable;
  bool java_checkerframework;
  bool gen_generated;
  std::string object_prefix;
  std::string object_suffix;
  bool union_value_namespacing;
  bool allow_non_utf8;
  bool natural_utf8;
  std::string include_prefix;
  bool keep_include_path;
  bool binary_schema_comments;
  bool binary_schema_builtins;
  bool skip_flatbuffers_import;
  std::string go_import;
  std::string go_namespace;
  bool reexport_ts_modules;
  bool js_ts_short_names;
  bool protobuf_ascii_alike;
  bool size_prefixed;
  std::string root_type;
  bool force_defaults;
  bool java_primitive_has_method;
  std::vector<std::string> cpp_includes;

  // Possible options for the more general generator below.
  enum Language {
    kJava = 1 << 0,
    kCSharp = 1 << 1,
    kGo = 1 << 2,
    kCpp = 1 << 3,
    kJs = 1 << 4,
    kPython = 1 << 5,
    kPhp = 1 << 6,
    kJson = 1 << 7,
    kBinary = 1 << 8,
    kTs = 1 << 9,
    kJsonSchema = 1 << 10,
    kDart = 1 << 11,
    kLua = 1 << 12,
    kLobster = 1 << 13,
    kRust = 1 << 14,
    kKotlin = 1 << 15,
    kMAX
  };

  Language lang;

  enum MiniReflect { kNone, kTypes, kTypesAndNames };

  MiniReflect mini_reflect;

  // The corresponding language bit will be set if a language is included
  // for code generation.
  unsigned long lang_to_generate;

  // If set (default behavior), empty string and vector fields will be set to
  // nullptr to make the flatbuffer more compact.
  bool set_empty_to_null;

  IDLOptions()
      : strict_json(false),
        skip_js_exports(false),
        use_goog_js_export_format(false),
        use_ES6_js_export_format(false),
        output_default_scalars_in_json(false),
        indent_step(2),
        output_enum_identifiers(true),
        prefixed_enums(true),
        scoped_enums(false),
        include_dependence_headers(true),
        mutable_buffer(false),
        one_file(false),
        proto_mode(false),
        proto_oneof_union(false),
        generate_all(false),
        skip_unexpected_fields_in_json(false),
        generate_name_strings(false),
        generate_object_based_api(false),
        gen_compare(false),
        cpp_object_api_pointer_type("std::unique_ptr"),
        cpp_object_api_string_flexible_constructor(false),
        gen_nullable(false),
        java_checkerframework(false),
        gen_generated(false),
        object_suffix("T"),
        union_value_namespacing(true),
        allow_non_utf8(false),
        natural_utf8(false),
        keep_include_path(false),
        binary_schema_comments(false),
        binary_schema_builtins(false),
        skip_flatbuffers_import(false),
        reexport_ts_modules(true),
        js_ts_short_names(false),
        protobuf_ascii_alike(false),
        size_prefixed(false),
        force_defaults(false),
        java_primitive_has_method(false),
        lang(IDLOptions::kJava),
        mini_reflect(IDLOptions::kNone),
        lang_to_generate(0),
        set_empty_to_null(true) {}
};

// This encapsulates where the parser is in the current source file.
struct ParserState {
  ParserState()
      : cursor_(nullptr),
        line_start_(nullptr),
        line_(0),
        token_(-1),
        attr_is_trivial_ascii_string_(true) {}

 protected:
  void ResetState(const char *source) {
    cursor_ = source;
    line_ = 0;
    MarkNewLine();
  }

  void MarkNewLine() {
    line_start_ = cursor_;
    line_ += 1;
  }

  int64_t CursorPosition() const {
    FLATBUFFERS_ASSERT(cursor_ && line_start_ && cursor_ >= line_start_);
    return static_cast<int64_t>(cursor_ - line_start_);
  }

  const char *cursor_;
  const char *line_start_;
  int line_;  // the current line being parsed
  int token_;

  // Flag: text in attribute_ is true ASCII string without escape
  // sequences. Only printable ASCII (without [\t\r\n]).
  // Used for number-in-string (and base64 string in future).
  bool attr_is_trivial_ascii_string_;
  std::string attribute_;
  std::vector<std::string> doc_comment_;
};

// A way to make error propagation less error prone by requiring values to be
// checked.
// Once you create a value of this type you must either:
// - Call Check() on it.
// - Copy or assign it to another value.
// Failure to do so leads to an assert.
// This guarantees that this as return value cannot be ignored.
class CheckedError {
 public:
  explicit CheckedError(bool error)
      : is_error_(error), has_been_checked_(false) {}

  CheckedError &operator=(const CheckedError &other) {
    is_error_ = other.is_error_;
    has_been_checked_ = false;
    other.has_been_checked_ = true;
    return *this;
  }

  CheckedError(const CheckedError &other) {
    *this = other;  // Use assignment operator.
  }

  ~CheckedError() { FLATBUFFERS_ASSERT(has_been_checked_); }

  bool Check() {
    has_been_checked_ = true;
    return is_error_;
  }

 private:
  bool is_error_;
  mutable bool has_been_checked_;
};

// Additionally, in GCC we can get these errors statically, for additional
// assurance:
// clang-format off
#ifdef __GNUC__
#define FLATBUFFERS_CHECKED_ERROR CheckedError \
          __attribute__((warn_unused_result))
#else
#define FLATBUFFERS_CHECKED_ERROR CheckedError
#endif
// clang-format on

class Parser : public ParserState {
 public:
  explicit Parser(const IDLOptions &options = IDLOptions())
      : current_namespace_(nullptr),
        empty_namespace_(nullptr),
        root_struct_def_(nullptr),
        opts(options),
        uses_flexbuffers_(false),
        source_(nullptr),
        anonymous_counter(0),
        recurse_protection_counter(0) {
    if (opts.force_defaults) {
      builder_.ForceDefaults(true);
    }
    // Start out with the empty namespace being current.
    empty_namespace_ = new Namespace();
    namespaces_.push_back(empty_namespace_);
    current_namespace_ = empty_namespace_;
    known_attributes_["deprecated"] = true;
    known_attributes_["required"] = true;
    known_attributes_["key"] = true;
    known_attributes_["shared"] = true;
    known_attributes_["hash"] = true;
    known_attributes_["id"] = true;
    known_attributes_["force_align"] = true;
    known_attributes_["bit_flags"] = true;
    known_attributes_["original_order"] = true;
    known_attributes_["nested_flatbuffer"] = true;
    known_attributes_["csharp_partial"] = true;
    known_attributes_["streaming"] = true;
    known_attributes_["idempotent"] = true;
    known_attributes_["cpp_type"] = true;
    known_attributes_["cpp_ptr_type"] = true;
    known_attributes_["cpp_ptr_type_get"] = true;
    known_attributes_["cpp_str_type"] = true;
    known_attributes_["cpp_str_flex_ctor"] = true;
    known_attributes_["native_inline"] = true;
    known_attributes_["native_custom_alloc"] = true;
    known_attributes_["native_type"] = true;
    known_attributes_["native_default"] = true;
    known_attributes_["flexbuffer"] = true;
    known_attributes_["private"] = true;
  }

  ~Parser() {
    for (auto it = namespaces_.begin(); it != namespaces_.end(); ++it) {
      delete *it;
    }
  }

  // Parse the string containing either schema or JSON data, which will
  // populate the SymbolTable's or the FlatBufferBuilder above.
  // include_paths is used to resolve any include statements, and typically
  // should at least include the project path (where you loaded source_ from).
  // include_paths must be nullptr terminated if specified.
  // If include_paths is nullptr, it will attempt to load from the current
  // directory.
  // If the source was loaded from a file and isn't an include file,
  // supply its name in source_filename.
  // All paths specified in this call must be in posix format, if you accept
  // paths from user input, please call PosixPath on them first.
  bool Parse(const char *_source, const char **include_paths = nullptr,
             const char *source_filename = nullptr);

  // Set the root type. May override the one set in the schema.
  bool SetRootType(const char *name);

  // Mark all definitions as already having code generated.
  void MarkGenerated();

  // Get the files recursively included by the given file. The returned
  // container will have at least the given file.
  std::set<std::string> GetIncludedFilesRecursive(
      const std::string &file_name) const;

  // Fills builder_ with a binary version of the schema parsed.
  // See reflection/reflection.fbs
  void Serialize();

  // Deserialize a schema buffer
  bool Deserialize(const uint8_t *buf, const size_t size);

  // Fills internal structure as if the schema passed had been loaded by parsing
  // with Parse except that included filenames will not be populated.
  bool Deserialize(const reflection::Schema* schema);

  Type* DeserializeType(const reflection::Type* type);

  // Checks that the schema represented by this parser is a safe evolution
  // of the schema provided. Returns non-empty error on any problems.
  std::string ConformTo(const Parser &base);

  // Similar to Parse(), but now only accepts JSON to be parsed into a
  // FlexBuffer.
  bool ParseFlexBuffer(const char *source, const char *source_filename,
                       flexbuffers::Builder *builder);

  StructDef *LookupStruct(const std::string &id) const;

  std::string UnqualifiedName(const std::string &fullQualifiedName);

  FLATBUFFERS_CHECKED_ERROR Error(const std::string &msg);

 private:
  void Message(const std::string &msg);
  void Warning(const std::string &msg);
  FLATBUFFERS_CHECKED_ERROR ParseHexNum(int nibbles, uint64_t *val);
  FLATBUFFERS_CHECKED_ERROR Next();
  FLATBUFFERS_CHECKED_ERROR SkipByteOrderMark();
  bool Is(int t) const;
  bool IsIdent(const char *id) const;
  FLATBUFFERS_CHECKED_ERROR Expect(int t);
  std::string TokenToStringId(int t) const;
  EnumDef *LookupEnum(const std::string &id);
  FLATBUFFERS_CHECKED_ERROR ParseNamespacing(std::string *id,
                                             std::string *last);
  FLATBUFFERS_CHECKED_ERROR ParseTypeIdent(Type &type);
  FLATBUFFERS_CHECKED_ERROR ParseType(Type &type);
  FLATBUFFERS_CHECKED_ERROR AddField(StructDef &struct_def,
                                     const std::string &name, const Type &type,
                                     FieldDef **dest);
  FLATBUFFERS_CHECKED_ERROR ParseField(StructDef &struct_def);
  FLATBUFFERS_CHECKED_ERROR ParseString(Value &val);
  FLATBUFFERS_CHECKED_ERROR ParseComma();
  FLATBUFFERS_CHECKED_ERROR ParseAnyValue(Value &val, FieldDef *field,
                                          size_t parent_fieldn,
                                          const StructDef *parent_struct_def,
                                          uoffset_t count,
                                          bool inside_vector = false);
  template<typename F>
  FLATBUFFERS_CHECKED_ERROR ParseTableDelimiters(size_t &fieldn,
                                                 const StructDef *struct_def,
                                                 F body);
  FLATBUFFERS_CHECKED_ERROR ParseTable(const StructDef &struct_def,
                                       std::string *value, uoffset_t *ovalue);
  void SerializeStruct(const StructDef &struct_def, const Value &val);
  void SerializeStruct(FlatBufferBuilder &builder, const StructDef &struct_def,
                       const Value &val);
  template<typename F>
  FLATBUFFERS_CHECKED_ERROR ParseVectorDelimiters(uoffset_t &count, F body);
  FLATBUFFERS_CHECKED_ERROR ParseVector(const Type &type, uoffset_t *ovalue,
                                        FieldDef *field, size_t fieldn);
  FLATBUFFERS_CHECKED_ERROR ParseArray(Value &array);
  FLATBUFFERS_CHECKED_ERROR ParseNestedFlatbuffer(Value &val, FieldDef *field,
                                                  size_t fieldn,
                                                  const StructDef *parent_struct_def);
  FLATBUFFERS_CHECKED_ERROR ParseMetaData(SymbolTable<Value> *attributes);
  FLATBUFFERS_CHECKED_ERROR TryTypedValue(const std::string *name, int dtoken, bool check, Value &e,
                                          BaseType req, bool *destmatch);
  FLATBUFFERS_CHECKED_ERROR ParseHash(Value &e, FieldDef* field);
  FLATBUFFERS_CHECKED_ERROR TokenError();
  FLATBUFFERS_CHECKED_ERROR ParseSingleValue(const std::string *name, Value &e, bool check_now);
  FLATBUFFERS_CHECKED_ERROR ParseEnumFromString(const Type &type, std::string *result);
  StructDef *LookupCreateStruct(const std::string &name,
                                bool create_if_new = true,
                                bool definition = false);
  FLATBUFFERS_CHECKED_ERROR ParseEnum(bool is_union, EnumDef **dest);
  FLATBUFFERS_CHECKED_ERROR ParseNamespace();
  FLATBUFFERS_CHECKED_ERROR StartStruct(const std::string &name,
                                        StructDef **dest);
  FLATBUFFERS_CHECKED_ERROR StartEnum(const std::string &name,
                                      bool is_union,
                                      EnumDef **dest);
  FLATBUFFERS_CHECKED_ERROR ParseDecl();
  FLATBUFFERS_CHECKED_ERROR ParseService();
  FLATBUFFERS_CHECKED_ERROR ParseProtoFields(StructDef *struct_def,
                                             bool isextend, bool inside_oneof);
  FLATBUFFERS_CHECKED_ERROR ParseProtoOption();
  FLATBUFFERS_CHECKED_ERROR ParseProtoKey();
  FLATBUFFERS_CHECKED_ERROR ParseProtoDecl();
  FLATBUFFERS_CHECKED_ERROR ParseProtoCurliesOrIdent();
  FLATBUFFERS_CHECKED_ERROR ParseTypeFromProtoType(Type *type);
  FLATBUFFERS_CHECKED_ERROR SkipAnyJsonValue();
  FLATBUFFERS_CHECKED_ERROR ParseFlexBufferValue(flexbuffers::Builder *builder);
  FLATBUFFERS_CHECKED_ERROR StartParseFile(const char *source,
                                           const char *source_filename);
  FLATBUFFERS_CHECKED_ERROR ParseRoot(const char *_source,
                                    const char **include_paths,
                                    const char *source_filename);
  FLATBUFFERS_CHECKED_ERROR DoParse(const char *_source,
                                           const char **include_paths,
                                           const char *source_filename,
                                           const char *include_filename);
  FLATBUFFERS_CHECKED_ERROR CheckClash(std::vector<FieldDef*> &fields,
                                       StructDef *struct_def,
                                       const char *suffix,
                                       BaseType baseType);

  bool SupportsAdvancedUnionFeatures() const;
  bool SupportsAdvancedArrayFeatures() const;
  Namespace *UniqueNamespace(Namespace *ns);

  FLATBUFFERS_CHECKED_ERROR RecurseError();
  template<typename F> CheckedError Recurse(F f);

 public:
  SymbolTable<Type> types_;
  SymbolTable<StructDef> structs_;
  SymbolTable<EnumDef> enums_;
  SymbolTable<ServiceDef> services_;
  std::vector<Namespace *> namespaces_;
  Namespace *current_namespace_;
  Namespace *empty_namespace_;
  std::string error_;         // User readable error_ if Parse() == false

  FlatBufferBuilder builder_;  // any data contained in the file
  StructDef *root_struct_def_;
  std::string file_identifier_;
  std::string file_extension_;

  std::map<std::string, std::string> included_files_;
  std::map<std::string, std::set<std::string>> files_included_per_file_;
  std::vector<std::string> native_included_files_;

  std::map<std::string, bool> known_attributes_;

  IDLOptions opts;
  bool uses_flexbuffers_;

 private:
  const char *source_;

  std::string file_being_parsed_;

  std::vector<std::pair<Value, FieldDef *>> field_stack_;

  int anonymous_counter;
  int recurse_protection_counter;
};

// Utility functions for multiple generators:

extern std::string MakeCamel(const std::string &in, bool first = true);

extern std::string MakeScreamingCamel(const std::string &in);

// Generate text (JSON) from a given FlatBuffer, and a given Parser
// object that has been populated with the corresponding schema.
// If ident_step is 0, no indentation will be generated. Additionally,
// if it is less than 0, no linefeeds will be generated either.
// See idl_gen_text.cpp.
// strict_json adds "quotes" around field names if true.
// If the flatbuffer cannot be encoded in JSON (e.g., it contains non-UTF-8
// byte arrays in String values), returns false.
extern bool GenerateTextFromTable(const Parser &parser,
                                  const void *table,
                                  const std::string &tablename,
                                  std::string *text);
extern bool GenerateText(const Parser &parser,
                         const void *flatbuffer,
                         std::string *text);
extern bool GenerateTextFile(const Parser &parser,
                             const std::string &path,
                             const std::string &file_name);

// Generate binary files from a given FlatBuffer, and a given Parser
// object that has been populated with the corresponding schema.
// See idl_gen_general.cpp.
extern bool GenerateBinary(const Parser &parser,
                           const std::string &path,
                           const std::string &file_name);

// Generate a C++ header from the definitions in the Parser object.
// See idl_gen_cpp.
extern bool GenerateCPP(const Parser &parser,
                        const std::string &path,
                        const std::string &file_name);

extern bool GenerateDart(const Parser &parser,
                         const std::string &path,
                         const std::string &file_name);

// Generate JavaScript or TypeScript code from the definitions in the Parser object.
// See idl_gen_js.
extern bool GenerateJSTS(const Parser &parser,
                       const std::string &path,
                       const std::string &file_name);

// Generate Go files from the definitions in the Parser object.
// See idl_gen_go.cpp.
extern bool GenerateGo(const Parser &parser,
                       const std::string &path,
                       const std::string &file_name);

// Generate Php code from the definitions in the Parser object.
// See idl_gen_php.
extern bool GeneratePhp(const Parser &parser,
                        const std::string &path,
                        const std::string &file_name);

// Generate Python files from the definitions in the Parser object.
// See idl_gen_python.cpp.
extern bool GeneratePython(const Parser &parser,
                           const std::string &path,
                           const std::string &file_name);

// Generate Lobster files from the definitions in the Parser object.
// See idl_gen_lobster.cpp.
extern bool GenerateLobster(const Parser &parser,
                            const std::string &path,
                            const std::string &file_name);

// Generate Lua files from the definitions in the Parser object.
// See idl_gen_lua.cpp.
extern bool GenerateLua(const Parser &parser,
                      const std::string &path,
                      const std::string &file_name);

// Generate Rust files from the definitions in the Parser object.
// See idl_gen_rust.cpp.
extern bool GenerateRust(const Parser &parser,
                         const std::string &path,
                         const std::string &file_name);

// Generate Json schema file
// See idl_gen_json_schema.cpp.
extern bool GenerateJsonSchema(const Parser &parser,
                           const std::string &path,
                           const std::string &file_name);

extern bool GenerateKotlin(const Parser &parser, const std::string &path,
                           const std::string &file_name);

// Generate Java/C#/.. files from the definitions in the Parser object.
// See idl_gen_general.cpp.
extern bool GenerateGeneral(const Parser &parser,
                            const std::string &path,
                            const std::string &file_name);

// Generate a schema file from the internal representation, useful after
// parsing a .proto schema.
extern std::string GenerateFBS(const Parser &parser,
                               const std::string &file_name);
extern bool GenerateFBS(const Parser &parser,
                        const std::string &path,
                        const std::string &file_name);

// Generate a make rule for the generated JavaScript or TypeScript code.
// See idl_gen_js.cpp.
extern std::string JSTSMakeRule(const Parser &parser,
                              const std::string &path,
                              const std::string &file_name);

// Generate a make rule for the generated C++ header.
// See idl_gen_cpp.cpp.
extern std::string CPPMakeRule(const Parser &parser,
                               const std::string &path,
                               const std::string &file_name);

// Generate a make rule for the generated Dart code
// see idl_gen_dart.cpp
extern std::string DartMakeRule(const Parser &parser,
                                const std::string &path,
                                const std::string &file_name);

// Generate a make rule for the generated Rust code.
// See idl_gen_rust.cpp.
extern std::string RustMakeRule(const Parser &parser,
                                const std::string &path,
                                const std::string &file_name);

// Generate a make rule for the generated Java/C#/... files.
// See idl_gen_general.cpp.
extern std::string GeneralMakeRule(const Parser &parser,
                                   const std::string &path,
                                   const std::string &file_name);

// Generate a make rule for the generated text (JSON) files.
// See idl_gen_text.cpp.
extern std::string TextMakeRule(const Parser &parser,
                                const std::string &path,
                                const std::string &file_names);

// Generate a make rule for the generated binary files.
// See idl_gen_general.cpp.
extern std::string BinaryMakeRule(const Parser &parser,
                                  const std::string &path,
                                  const std::string &file_name);

// Generate GRPC Cpp interfaces.
// See idl_gen_grpc.cpp.
bool GenerateCppGRPC(const Parser &parser,
                     const std::string &path,
                     const std::string &file_name);

// Generate GRPC Go interfaces.
// See idl_gen_grpc.cpp.
bool GenerateGoGRPC(const Parser &parser,
                    const std::string &path,
                    const std::string &file_name);

// Generate GRPC Java classes.
// See idl_gen_grpc.cpp
bool GenerateJavaGRPC(const Parser &parser,
                      const std::string &path,
                      const std::string &file_name);

}  // namespace flatbuffers

#endif  // FLATBUFFERS_IDL_H_
