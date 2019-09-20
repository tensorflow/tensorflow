/*
 * Copyright 2017 Google Inc. All rights reserved.
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

#ifndef FLATBUFFERS_FLEXBUFFERS_H_
#define FLATBUFFERS_FLEXBUFFERS_H_

#include <map>
// Used to select STL variant.
#include "flatbuffers/base.h"
// We use the basic binary writing functions from the regular FlatBuffers.
#include "flatbuffers/util.h"

#ifdef _MSC_VER
#  include <intrin.h>
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4127)  // C4127: conditional expression is constant
#endif

namespace flexbuffers {

class Reference;
class Map;

// These are used in the lower 2 bits of a type field to determine the size of
// the elements (and or size field) of the item pointed to (e.g. vector).
enum BitWidth {
  BIT_WIDTH_8 = 0,
  BIT_WIDTH_16 = 1,
  BIT_WIDTH_32 = 2,
  BIT_WIDTH_64 = 3,
};

// These are used as the upper 6 bits of a type field to indicate the actual
// type.
enum Type {
  FBT_NULL = 0,
  FBT_INT = 1,
  FBT_UINT = 2,
  FBT_FLOAT = 3,
  // Types above stored inline, types below store an offset.
  FBT_KEY = 4,
  FBT_STRING = 5,
  FBT_INDIRECT_INT = 6,
  FBT_INDIRECT_UINT = 7,
  FBT_INDIRECT_FLOAT = 8,
  FBT_MAP = 9,
  FBT_VECTOR = 10,      // Untyped.
  FBT_VECTOR_INT = 11,  // Typed any size (stores no type table).
  FBT_VECTOR_UINT = 12,
  FBT_VECTOR_FLOAT = 13,
  FBT_VECTOR_KEY = 14,
  FBT_VECTOR_STRING = 15,
  FBT_VECTOR_INT2 = 16,  // Typed tuple (no type table, no size field).
  FBT_VECTOR_UINT2 = 17,
  FBT_VECTOR_FLOAT2 = 18,
  FBT_VECTOR_INT3 = 19,  // Typed triple (no type table, no size field).
  FBT_VECTOR_UINT3 = 20,
  FBT_VECTOR_FLOAT3 = 21,
  FBT_VECTOR_INT4 = 22,  // Typed quad (no type table, no size field).
  FBT_VECTOR_UINT4 = 23,
  FBT_VECTOR_FLOAT4 = 24,
  FBT_BLOB = 25,
  FBT_BOOL = 26,
  FBT_VECTOR_BOOL =
      36,  // To Allow the same type of conversion of type to vector type
};

inline bool IsInline(Type t) { return t <= FBT_FLOAT || t == FBT_BOOL; }

inline bool IsTypedVectorElementType(Type t) {
  return (t >= FBT_INT && t <= FBT_STRING) || t == FBT_BOOL;
}

inline bool IsTypedVector(Type t) {
  return (t >= FBT_VECTOR_INT && t <= FBT_VECTOR_STRING) ||
         t == FBT_VECTOR_BOOL;
}

inline bool IsFixedTypedVector(Type t) {
  return t >= FBT_VECTOR_INT2 && t <= FBT_VECTOR_FLOAT4;
}

inline Type ToTypedVector(Type t, size_t fixed_len = 0) {
  FLATBUFFERS_ASSERT(IsTypedVectorElementType(t));
  switch (fixed_len) {
    case 0: return static_cast<Type>(t - FBT_INT + FBT_VECTOR_INT);
    case 2: return static_cast<Type>(t - FBT_INT + FBT_VECTOR_INT2);
    case 3: return static_cast<Type>(t - FBT_INT + FBT_VECTOR_INT3);
    case 4: return static_cast<Type>(t - FBT_INT + FBT_VECTOR_INT4);
    default: FLATBUFFERS_ASSERT(0); return FBT_NULL;
  }
}

inline Type ToTypedVectorElementType(Type t) {
  FLATBUFFERS_ASSERT(IsTypedVector(t));
  return static_cast<Type>(t - FBT_VECTOR_INT + FBT_INT);
}

inline Type ToFixedTypedVectorElementType(Type t, uint8_t *len) {
  FLATBUFFERS_ASSERT(IsFixedTypedVector(t));
  auto fixed_type = t - FBT_VECTOR_INT2;
  *len = static_cast<uint8_t>(fixed_type / 3 +
                              2);  // 3 types each, starting from length 2.
  return static_cast<Type>(fixed_type % 3 + FBT_INT);
}

// TODO: implement proper support for 8/16bit floats, or decide not to
// support them.
typedef int16_t half;
typedef int8_t quarter;

// TODO: can we do this without conditionals using intrinsics or inline asm
// on some platforms? Given branch prediction the method below should be
// decently quick, but it is the most frequently executed function.
// We could do an (unaligned) 64-bit read if we ifdef out the platforms for
// which that doesn't work (or where we'd read into un-owned memory).
template<typename R, typename T1, typename T2, typename T4, typename T8>
R ReadSizedScalar(const uint8_t *data, uint8_t byte_width) {
  return byte_width < 4
             ? (byte_width < 2
                    ? static_cast<R>(flatbuffers::ReadScalar<T1>(data))
                    : static_cast<R>(flatbuffers::ReadScalar<T2>(data)))
             : (byte_width < 8
                    ? static_cast<R>(flatbuffers::ReadScalar<T4>(data))
                    : static_cast<R>(flatbuffers::ReadScalar<T8>(data)));
}

inline int64_t ReadInt64(const uint8_t *data, uint8_t byte_width) {
  return ReadSizedScalar<int64_t, int8_t, int16_t, int32_t, int64_t>(
      data, byte_width);
}

inline uint64_t ReadUInt64(const uint8_t *data, uint8_t byte_width) {
  // This is the "hottest" function (all offset lookups use this), so worth
  // optimizing if possible.
  // TODO: GCC apparently replaces memcpy by a rep movsb, but only if count is a
  // constant, which here it isn't. Test if memcpy is still faster than
  // the conditionals in ReadSizedScalar. Can also use inline asm.
  // clang-format off
  #if defined(_MSC_VER) && (defined(_M_X64) || defined _M_IX86)
    uint64_t u = 0;
    __movsb(reinterpret_cast<uint8_t *>(&u),
            reinterpret_cast<const uint8_t *>(data), byte_width);
    return flatbuffers::EndianScalar(u);
  #else
    return ReadSizedScalar<uint64_t, uint8_t, uint16_t, uint32_t, uint64_t>(
             data, byte_width);
  #endif
  // clang-format on
}

inline double ReadDouble(const uint8_t *data, uint8_t byte_width) {
  return ReadSizedScalar<double, quarter, half, float, double>(data,
                                                               byte_width);
}

inline const uint8_t *Indirect(const uint8_t *offset, uint8_t byte_width) {
  return offset - ReadUInt64(offset, byte_width);
}

template<typename T> const uint8_t *Indirect(const uint8_t *offset) {
  return offset - flatbuffers::ReadScalar<T>(offset);
}

inline BitWidth WidthU(uint64_t u) {
#define FLATBUFFERS_GET_FIELD_BIT_WIDTH(value, width)                   \
  {                                                                     \
    if (!((u) & ~((1ULL << (width)) - 1ULL))) return BIT_WIDTH_##width; \
  }
  FLATBUFFERS_GET_FIELD_BIT_WIDTH(u, 8);
  FLATBUFFERS_GET_FIELD_BIT_WIDTH(u, 16);
  FLATBUFFERS_GET_FIELD_BIT_WIDTH(u, 32);
#undef FLATBUFFERS_GET_FIELD_BIT_WIDTH
  return BIT_WIDTH_64;
}

inline BitWidth WidthI(int64_t i) {
  auto u = static_cast<uint64_t>(i) << 1;
  return WidthU(i >= 0 ? u : ~u);
}

inline BitWidth WidthF(double f) {
  return static_cast<double>(static_cast<float>(f)) == f ? BIT_WIDTH_32
                                                         : BIT_WIDTH_64;
}

// Base class of all types below.
// Points into the data buffer and allows access to one type.
class Object {
 public:
  Object(const uint8_t *data, uint8_t byte_width)
      : data_(data), byte_width_(byte_width) {}

 protected:
  const uint8_t *data_;
  uint8_t byte_width_;
};

// Stores size in `byte_width_` bytes before data_ pointer.
class Sized : public Object {
 public:
  Sized(const uint8_t *data, uint8_t byte_width) : Object(data, byte_width) {}
  size_t size() const {
    return static_cast<size_t>(ReadUInt64(data_ - byte_width_, byte_width_));
  }
};

class String : public Sized {
 public:
  String(const uint8_t *data, uint8_t byte_width) : Sized(data, byte_width) {}

  size_t length() const { return size(); }
  const char *c_str() const { return reinterpret_cast<const char *>(data_); }
  std::string str() const { return std::string(c_str(), length()); }

  static String EmptyString() {
    static const uint8_t empty_string[] = { 0 /*len*/, 0 /*terminator*/ };
    return String(empty_string + 1, 1);
  }
  bool IsTheEmptyString() const { return data_ == EmptyString().data_; }
};

class Blob : public Sized {
 public:
  Blob(const uint8_t *data_buf, uint8_t byte_width)
      : Sized(data_buf, byte_width) {}

  static Blob EmptyBlob() {
    static const uint8_t empty_blob[] = { 0 /*len*/ };
    return Blob(empty_blob + 1, 1);
  }
  bool IsTheEmptyBlob() const { return data_ == EmptyBlob().data_; }
  const uint8_t *data() const { return data_; }
};

class Vector : public Sized {
 public:
  Vector(const uint8_t *data, uint8_t byte_width) : Sized(data, byte_width) {}

  Reference operator[](size_t i) const;

  static Vector EmptyVector() {
    static const uint8_t empty_vector[] = { 0 /*len*/ };
    return Vector(empty_vector + 1, 1);
  }
  bool IsTheEmptyVector() const { return data_ == EmptyVector().data_; }
};

class TypedVector : public Sized {
 public:
  TypedVector(const uint8_t *data, uint8_t byte_width, Type element_type)
      : Sized(data, byte_width), type_(element_type) {}

  Reference operator[](size_t i) const;

  static TypedVector EmptyTypedVector() {
    static const uint8_t empty_typed_vector[] = { 0 /*len*/ };
    return TypedVector(empty_typed_vector + 1, 1, FBT_INT);
  }
  bool IsTheEmptyVector() const {
    return data_ == TypedVector::EmptyTypedVector().data_;
  }

  Type ElementType() { return type_; }

 private:
  Type type_;

  friend Map;
};

class FixedTypedVector : public Object {
 public:
  FixedTypedVector(const uint8_t *data, uint8_t byte_width, Type element_type,
                   uint8_t len)
      : Object(data, byte_width), type_(element_type), len_(len) {}

  Reference operator[](size_t i) const;

  static FixedTypedVector EmptyFixedTypedVector() {
    static const uint8_t fixed_empty_vector[] = { 0 /* unused */ };
    return FixedTypedVector(fixed_empty_vector, 1, FBT_INT, 0);
  }
  bool IsTheEmptyFixedTypedVector() const {
    return data_ == FixedTypedVector::EmptyFixedTypedVector().data_;
  }

  Type ElementType() { return type_; }
  uint8_t size() { return len_; }

 private:
  Type type_;
  uint8_t len_;
};

class Map : public Vector {
 public:
  Map(const uint8_t *data, uint8_t byte_width) : Vector(data, byte_width) {}

  Reference operator[](const char *key) const;
  Reference operator[](const std::string &key) const;

  Vector Values() const { return Vector(data_, byte_width_); }

  TypedVector Keys() const {
    const size_t num_prefixed_fields = 3;
    auto keys_offset = data_ - byte_width_ * num_prefixed_fields;
    return TypedVector(Indirect(keys_offset, byte_width_),
                       static_cast<uint8_t>(
                           ReadUInt64(keys_offset + byte_width_, byte_width_)),
                       FBT_KEY);
  }

  static Map EmptyMap() {
    static const uint8_t empty_map[] = {
      0 /*keys_len*/, 0 /*keys_offset*/, 1 /*keys_width*/, 0 /*len*/
    };
    return Map(empty_map + 4, 1);
  }

  bool IsTheEmptyMap() const { return data_ == EmptyMap().data_; }
};

template<typename T>
void AppendToString(std::string &s, T &&v, bool keys_quoted) {
    s += "[ ";
    for (size_t i = 0; i < v.size(); i++) {
      if (i) s += ", ";
      v[i].ToString(true, keys_quoted, s);
    }
    s += " ]";
}

class Reference {
 public:
  Reference(const uint8_t *data, uint8_t parent_width, uint8_t byte_width,
            Type type)
      : data_(data),
        parent_width_(parent_width),
        byte_width_(byte_width),
        type_(type) {}

  Reference(const uint8_t *data, uint8_t parent_width, uint8_t packed_type)
      : data_(data), parent_width_(parent_width) {
    byte_width_ = 1U << static_cast<BitWidth>(packed_type & 3);
    type_ = static_cast<Type>(packed_type >> 2);
  }

  Type GetType() const { return type_; }

  bool IsNull() const { return type_ == FBT_NULL; }
  bool IsBool() const { return type_ == FBT_BOOL; }
  bool IsInt() const { return type_ == FBT_INT || type_ == FBT_INDIRECT_INT; }
  bool IsUInt() const {
    return type_ == FBT_UINT || type_ == FBT_INDIRECT_UINT;
  }
  bool IsIntOrUint() const { return IsInt() || IsUInt(); }
  bool IsFloat() const {
    return type_ == FBT_FLOAT || type_ == FBT_INDIRECT_FLOAT;
  }
  bool IsNumeric() const { return IsIntOrUint() || IsFloat(); }
  bool IsString() const { return type_ == FBT_STRING; }
  bool IsKey() const { return type_ == FBT_KEY; }
  bool IsVector() const { return type_ == FBT_VECTOR || type_ == FBT_MAP; }
  bool IsUntypedVector() const { return type_ == FBT_VECTOR; }
  bool IsTypedVector() const { return flexbuffers::IsTypedVector(type_); }
  bool IsFixedTypedVector() const { return flexbuffers::IsFixedTypedVector(type_); }
  bool IsAnyVector() const { return (IsTypedVector() || IsFixedTypedVector() || IsVector());}
  bool IsMap() const { return type_ == FBT_MAP; }
  bool IsBlob() const { return type_ == FBT_BLOB; }
  bool AsBool() const {
    return (type_ == FBT_BOOL ? ReadUInt64(data_, parent_width_)
                               : AsUInt64()) != 0;
  }

  // Reads any type as a int64_t. Never fails, does most sensible conversion.
  // Truncates floats, strings are attempted to be parsed for a number,
  // vectors/maps return their size. Returns 0 if all else fails.
  int64_t AsInt64() const {
    if (type_ == FBT_INT) {
      // A fast path for the common case.
      return ReadInt64(data_, parent_width_);
    } else
      switch (type_) {
        case FBT_INDIRECT_INT: return ReadInt64(Indirect(), byte_width_);
        case FBT_UINT: return ReadUInt64(data_, parent_width_);
        case FBT_INDIRECT_UINT: return ReadUInt64(Indirect(), byte_width_);
        case FBT_FLOAT:
          return static_cast<int64_t>(ReadDouble(data_, parent_width_));
        case FBT_INDIRECT_FLOAT:
          return static_cast<int64_t>(ReadDouble(Indirect(), byte_width_));
        case FBT_NULL: return 0;
        case FBT_STRING: return flatbuffers::StringToInt(AsString().c_str());
        case FBT_VECTOR: return static_cast<int64_t>(AsVector().size());
        case FBT_BOOL: return ReadInt64(data_, parent_width_);
        default:
          // Convert other things to int.
          return 0;
      }
  }

  // TODO: could specialize these to not use AsInt64() if that saves
  // extension ops in generated code, and use a faster op than ReadInt64.
  int32_t AsInt32() const { return static_cast<int32_t>(AsInt64()); }
  int16_t AsInt16() const { return static_cast<int16_t>(AsInt64()); }
  int8_t AsInt8() const { return static_cast<int8_t>(AsInt64()); }

  uint64_t AsUInt64() const {
    if (type_ == FBT_UINT) {
      // A fast path for the common case.
      return ReadUInt64(data_, parent_width_);
    } else
      switch (type_) {
        case FBT_INDIRECT_UINT: return ReadUInt64(Indirect(), byte_width_);
        case FBT_INT: return ReadInt64(data_, parent_width_);
        case FBT_INDIRECT_INT: return ReadInt64(Indirect(), byte_width_);
        case FBT_FLOAT:
          return static_cast<uint64_t>(ReadDouble(data_, parent_width_));
        case FBT_INDIRECT_FLOAT:
          return static_cast<uint64_t>(ReadDouble(Indirect(), byte_width_));
        case FBT_NULL: return 0;
        case FBT_STRING: return flatbuffers::StringToUInt(AsString().c_str());
        case FBT_VECTOR: return static_cast<uint64_t>(AsVector().size());
        case FBT_BOOL: return ReadUInt64(data_, parent_width_);
        default:
          // Convert other things to uint.
          return 0;
      }
  }

  uint32_t AsUInt32() const { return static_cast<uint32_t>(AsUInt64()); }
  uint16_t AsUInt16() const { return static_cast<uint16_t>(AsUInt64()); }
  uint8_t AsUInt8() const { return static_cast<uint8_t>(AsUInt64()); }

  double AsDouble() const {
    if (type_ == FBT_FLOAT) {
      // A fast path for the common case.
      return ReadDouble(data_, parent_width_);
    } else
      switch (type_) {
        case FBT_INDIRECT_FLOAT: return ReadDouble(Indirect(), byte_width_);
        case FBT_INT:
          return static_cast<double>(ReadInt64(data_, parent_width_));
        case FBT_UINT:
          return static_cast<double>(ReadUInt64(data_, parent_width_));
        case FBT_INDIRECT_INT:
          return static_cast<double>(ReadInt64(Indirect(), byte_width_));
        case FBT_INDIRECT_UINT:
          return static_cast<double>(ReadUInt64(Indirect(), byte_width_));
        case FBT_NULL: return 0.0;
        case FBT_STRING: return strtod(AsString().c_str(), nullptr);
        case FBT_VECTOR: return static_cast<double>(AsVector().size());
        case FBT_BOOL:
          return static_cast<double>(ReadUInt64(data_, parent_width_));
        default:
          // Convert strings and other things to float.
          return 0;
      }
  }

  float AsFloat() const { return static_cast<float>(AsDouble()); }

  const char *AsKey() const {
    if (type_ == FBT_KEY) {
      return reinterpret_cast<const char *>(Indirect());
    } else {
      return "";
    }
  }

  // This function returns the empty string if you try to read a not-string.
  String AsString() const {
    if (type_ == FBT_STRING) {
      return String(Indirect(), byte_width_);
    } else {
      return String::EmptyString();
    }
  }

  // Unlike AsString(), this will convert any type to a std::string.
  std::string ToString() const {
    std::string s;
    ToString(false, false, s);
    return s;
  }

  // Convert any type to a JSON-like string. strings_quoted determines if
  // string values at the top level receive "" quotes (inside other values
  // they always do). keys_quoted determines if keys are quoted, at any level.
  // TODO(wvo): add further options to have indentation/newlines.
  void ToString(bool strings_quoted, bool keys_quoted, std::string &s) const {
    if (type_ == FBT_STRING) {
      String str(Indirect(), byte_width_);
      if (strings_quoted) {
        flatbuffers::EscapeString(str.c_str(), str.length(), &s, true, false);
      } else {
        s.append(str.c_str(), str.length());
      }
    } else if (IsKey()) {
      auto str = AsKey();
      if (keys_quoted) {
        flatbuffers::EscapeString(str, strlen(str), &s, true, false);
      } else {
        s += str;
      }
    } else if (IsInt()) {
      s += flatbuffers::NumToString(AsInt64());
    } else if (IsUInt()) {
      s += flatbuffers::NumToString(AsUInt64());
    } else if (IsFloat()) {
      s += flatbuffers::NumToString(AsDouble());
    } else if (IsNull()) {
      s += "null";
    } else if (IsBool()) {
      s += AsBool() ? "true" : "false";
    } else if (IsMap()) {
      s += "{ ";
      auto m = AsMap();
      auto keys = m.Keys();
      auto vals = m.Values();
      for (size_t i = 0; i < keys.size(); i++) {
        keys[i].ToString(true, keys_quoted, s);
        s += ": ";
        vals[i].ToString(true, keys_quoted, s);
        if (i < keys.size() - 1) s += ", ";
      }
      s += " }";
    } else if (IsVector()) {
      AppendToString<Vector>(s, AsVector(), keys_quoted);
    } else if (IsTypedVector()) {
      AppendToString<TypedVector>(s, AsTypedVector(), keys_quoted);
    } else if (IsFixedTypedVector()) {
      AppendToString<FixedTypedVector>(s, AsFixedTypedVector(), keys_quoted);
    } else if (IsBlob()) {
      auto blob = AsBlob();
      flatbuffers::EscapeString(reinterpret_cast<const char*>(blob.data()), blob.size(), &s, true, false);
    } else {
      s += "(?)";
    }
  }

  // This function returns the empty blob if you try to read a not-blob.
  // Strings can be viewed as blobs too.
  Blob AsBlob() const {
    if (type_ == FBT_BLOB || type_ == FBT_STRING) {
      return Blob(Indirect(), byte_width_);
    } else {
      return Blob::EmptyBlob();
    }
  }

  // This function returns the empty vector if you try to read a not-vector.
  // Maps can be viewed as vectors too.
  Vector AsVector() const {
    if (type_ == FBT_VECTOR || type_ == FBT_MAP) {
      return Vector(Indirect(), byte_width_);
    } else {
      return Vector::EmptyVector();
    }
  }

  TypedVector AsTypedVector() const {
    if (IsTypedVector()) {
      return TypedVector(Indirect(), byte_width_,
                         ToTypedVectorElementType(type_));
    } else {
      return TypedVector::EmptyTypedVector();
    }
  }

  FixedTypedVector AsFixedTypedVector() const {
    if (IsFixedTypedVector()) {
      uint8_t len = 0;
      auto vtype = ToFixedTypedVectorElementType(type_, &len);
      return FixedTypedVector(Indirect(), byte_width_, vtype, len);
    } else {
      return FixedTypedVector::EmptyFixedTypedVector();
    }
  }

  Map AsMap() const {
    if (type_ == FBT_MAP) {
      return Map(Indirect(), byte_width_);
    } else {
      return Map::EmptyMap();
    }
  }

  template<typename T> T As() const;

  // Experimental: Mutation functions.
  // These allow scalars in an already created buffer to be updated in-place.
  // Since by default scalars are stored in the smallest possible space,
  // the new value may not fit, in which case these functions return false.
  // To avoid this, you can construct the values you intend to mutate using
  // Builder::ForceMinimumBitWidth.
  bool MutateInt(int64_t i) {
    if (type_ == FBT_INT) {
      return Mutate(data_, i, parent_width_, WidthI(i));
    } else if (type_ == FBT_INDIRECT_INT) {
      return Mutate(Indirect(), i, byte_width_, WidthI(i));
    } else if (type_ == FBT_UINT) {
      auto u = static_cast<uint64_t>(i);
      return Mutate(data_, u, parent_width_, WidthU(u));
    } else if (type_ == FBT_INDIRECT_UINT) {
      auto u = static_cast<uint64_t>(i);
      return Mutate(Indirect(), u, byte_width_, WidthU(u));
    } else {
      return false;
    }
  }

  bool MutateBool(bool b) {
    return type_ == FBT_BOOL && Mutate(data_, b, parent_width_, BIT_WIDTH_8);
  }

  bool MutateUInt(uint64_t u) {
    if (type_ == FBT_UINT) {
      return Mutate(data_, u, parent_width_, WidthU(u));
    } else if (type_ == FBT_INDIRECT_UINT) {
      return Mutate(Indirect(), u, byte_width_, WidthU(u));
    } else if (type_ == FBT_INT) {
      auto i = static_cast<int64_t>(u);
      return Mutate(data_, i, parent_width_, WidthI(i));
    } else if (type_ == FBT_INDIRECT_INT) {
      auto i = static_cast<int64_t>(u);
      return Mutate(Indirect(), i, byte_width_, WidthI(i));
    } else {
      return false;
    }
  }

  bool MutateFloat(float f) {
    if (type_ == FBT_FLOAT) {
      return MutateF(data_, f, parent_width_, BIT_WIDTH_32);
    } else if (type_ == FBT_INDIRECT_FLOAT) {
      return MutateF(Indirect(), f, byte_width_, BIT_WIDTH_32);
    } else {
      return false;
    }
  }

  bool MutateFloat(double d) {
    if (type_ == FBT_FLOAT) {
      return MutateF(data_, d, parent_width_, WidthF(d));
    } else if (type_ == FBT_INDIRECT_FLOAT) {
      return MutateF(Indirect(), d, byte_width_, WidthF(d));
    } else {
      return false;
    }
  }

  bool MutateString(const char *str, size_t len) {
    auto s = AsString();
    if (s.IsTheEmptyString()) return false;
    // This is very strict, could allow shorter strings, but that creates
    // garbage.
    if (s.length() != len) return false;
    memcpy(const_cast<char *>(s.c_str()), str, len);
    return true;
  }
  bool MutateString(const char *str) { return MutateString(str, strlen(str)); }
  bool MutateString(const std::string &str) {
    return MutateString(str.data(), str.length());
  }

 private:
  const uint8_t *Indirect() const {
    return flexbuffers::Indirect(data_, parent_width_);
  }

  template<typename T>
  bool Mutate(const uint8_t *dest, T t, size_t byte_width,
              BitWidth value_width) {
    auto fits = static_cast<size_t>(static_cast<size_t>(1U) << value_width) <=
                byte_width;
    if (fits) {
      t = flatbuffers::EndianScalar(t);
      memcpy(const_cast<uint8_t *>(dest), &t, byte_width);
    }
    return fits;
  }

  template<typename T>
  bool MutateF(const uint8_t *dest, T t, size_t byte_width,
               BitWidth value_width) {
    if (byte_width == sizeof(double))
      return Mutate(dest, static_cast<double>(t), byte_width, value_width);
    if (byte_width == sizeof(float))
      return Mutate(dest, static_cast<float>(t), byte_width, value_width);
    FLATBUFFERS_ASSERT(false);
    return false;
  }

  const uint8_t *data_;
  uint8_t parent_width_;
  uint8_t byte_width_;
  Type type_;
};

// Template specialization for As().
template<> inline bool Reference::As<bool>() const { return AsBool(); }

template<> inline int8_t Reference::As<int8_t>() const { return AsInt8(); }
template<> inline int16_t Reference::As<int16_t>() const { return AsInt16(); }
template<> inline int32_t Reference::As<int32_t>() const { return AsInt32(); }
template<> inline int64_t Reference::As<int64_t>() const { return AsInt64(); }

template<> inline uint8_t Reference::As<uint8_t>() const { return AsUInt8(); }
template<> inline uint16_t Reference::As<uint16_t>() const { return AsUInt16(); }
template<> inline uint32_t Reference::As<uint32_t>() const { return AsUInt32(); }
template<> inline uint64_t Reference::As<uint64_t>() const { return AsUInt64(); }

template<> inline double Reference::As<double>() const { return AsDouble(); }
template<> inline float Reference::As<float>() const { return AsFloat(); }

template<> inline String Reference::As<String>() const { return AsString(); }
template<> inline std::string Reference::As<std::string>() const {
  return AsString().str();
}

template<> inline Blob Reference::As<Blob>() const { return AsBlob(); }
template<> inline Vector Reference::As<Vector>() const { return AsVector(); }
template<> inline TypedVector Reference::As<TypedVector>() const {
  return AsTypedVector();
}
template<> inline FixedTypedVector Reference::As<FixedTypedVector>() const {
  return AsFixedTypedVector();
}
template<> inline Map Reference::As<Map>() const { return AsMap(); }

inline uint8_t PackedType(BitWidth bit_width, Type type) {
  return static_cast<uint8_t>(bit_width | (type << 2));
}

inline uint8_t NullPackedType() { return PackedType(BIT_WIDTH_8, FBT_NULL); }

// Vector accessors.
// Note: if you try to access outside of bounds, you get a Null value back
// instead. Normally this would be an assert, but since this is "dynamically
// typed" data, you may not want that (someone sends you a 2d vector and you
// wanted 3d).
// The Null converts seamlessly into a default value for any other type.
// TODO(wvo): Could introduce an #ifdef that makes this into an assert?
inline Reference Vector::operator[](size_t i) const {
  auto len = size();
  if (i >= len) return Reference(nullptr, 1, NullPackedType());
  auto packed_type = (data_ + len * byte_width_)[i];
  auto elem = data_ + i * byte_width_;
  return Reference(elem, byte_width_, packed_type);
}

inline Reference TypedVector::operator[](size_t i) const {
  auto len = size();
  if (i >= len) return Reference(nullptr, 1, NullPackedType());
  auto elem = data_ + i * byte_width_;
  return Reference(elem, byte_width_, 1, type_);
}

inline Reference FixedTypedVector::operator[](size_t i) const {
  if (i >= len_) return Reference(nullptr, 1, NullPackedType());
  auto elem = data_ + i * byte_width_;
  return Reference(elem, byte_width_, 1, type_);
}

template<typename T> int KeyCompare(const void *key, const void *elem) {
  auto str_elem = reinterpret_cast<const char *>(
      Indirect<T>(reinterpret_cast<const uint8_t *>(elem)));
  auto skey = reinterpret_cast<const char *>(key);
  return strcmp(skey, str_elem);
}

inline Reference Map::operator[](const char *key) const {
  auto keys = Keys();
  // We can't pass keys.byte_width_ to the comparison function, so we have
  // to pick the right one ahead of time.
  int (*comp)(const void *, const void *) = nullptr;
  switch (keys.byte_width_) {
    case 1: comp = KeyCompare<uint8_t>; break;
    case 2: comp = KeyCompare<uint16_t>; break;
    case 4: comp = KeyCompare<uint32_t>; break;
    case 8: comp = KeyCompare<uint64_t>; break;
  }
  auto res = std::bsearch(key, keys.data_, keys.size(), keys.byte_width_, comp);
  if (!res) return Reference(nullptr, 1, NullPackedType());
  auto i = (reinterpret_cast<uint8_t *>(res) - keys.data_) / keys.byte_width_;
  return (*static_cast<const Vector *>(this))[i];
}

inline Reference Map::operator[](const std::string &key) const {
  return (*this)[key.c_str()];
}

inline Reference GetRoot(const uint8_t *buffer, size_t size) {
  // See Finish() below for the serialization counterpart of this.
  // The root starts at the end of the buffer, so we parse backwards from there.
  auto end = buffer + size;
  auto byte_width = *--end;
  auto packed_type = *--end;
  end -= byte_width;  // The root data item.
  return Reference(end, byte_width, packed_type);
}

inline Reference GetRoot(const std::vector<uint8_t> &buffer) {
  return GetRoot(flatbuffers::vector_data(buffer), buffer.size());
}

// Flags that configure how the Builder behaves.
// The "Share" flags determine if the Builder automatically tries to pool
// this type. Pooling can reduce the size of serialized data if there are
// multiple maps of the same kind, at the expense of slightly slower
// serialization (the cost of lookups) and more memory use (std::set).
// By default this is on for keys, but off for strings.
// Turn keys off if you have e.g. only one map.
// Turn strings on if you expect many non-unique string values.
// Additionally, sharing key vectors can save space if you have maps with
// identical field populations.
enum BuilderFlag {
  BUILDER_FLAG_NONE = 0,
  BUILDER_FLAG_SHARE_KEYS = 1,
  BUILDER_FLAG_SHARE_STRINGS = 2,
  BUILDER_FLAG_SHARE_KEYS_AND_STRINGS = 3,
  BUILDER_FLAG_SHARE_KEY_VECTORS = 4,
  BUILDER_FLAG_SHARE_ALL = 7,
};

class Builder FLATBUFFERS_FINAL_CLASS {
 public:
  Builder(size_t initial_size = 256,
          BuilderFlag flags = BUILDER_FLAG_SHARE_KEYS)
      : buf_(initial_size),
        finished_(false),
        flags_(flags),
        force_min_bit_width_(BIT_WIDTH_8),
        key_pool(KeyOffsetCompare(buf_)),
        string_pool(StringOffsetCompare(buf_)) {
    buf_.clear();
  }

  /// @brief Get the serialized buffer (after you call `Finish()`).
  /// @return Returns a vector owned by this class.
  const std::vector<uint8_t> &GetBuffer() const {
    Finished();
    return buf_;
  }

  // Size of the buffer. Does not include unfinished values.
  size_t GetSize() const { return buf_.size(); }

  // Reset all state so we can re-use the buffer.
  void Clear() {
    buf_.clear();
    stack_.clear();
    finished_ = false;
    // flags_ remains as-is;
    force_min_bit_width_ = BIT_WIDTH_8;
    key_pool.clear();
    string_pool.clear();
  }

  // All value constructing functions below have two versions: one that
  // takes a key (for placement inside a map) and one that doesn't (for inside
  // vectors and elsewhere).

  void Null() { stack_.push_back(Value()); }
  void Null(const char *key) {
    Key(key);
    Null();
  }

  void Int(int64_t i) { stack_.push_back(Value(i, FBT_INT, WidthI(i))); }
  void Int(const char *key, int64_t i) {
    Key(key);
    Int(i);
  }

  void UInt(uint64_t u) { stack_.push_back(Value(u, FBT_UINT, WidthU(u))); }
  void UInt(const char *key, uint64_t u) {
    Key(key);
    UInt(u);
  }

  void Float(float f) { stack_.push_back(Value(f)); }
  void Float(const char *key, float f) {
    Key(key);
    Float(f);
  }

  void Double(double f) { stack_.push_back(Value(f)); }
  void Double(const char *key, double d) {
    Key(key);
    Double(d);
  }

  void Bool(bool b) { stack_.push_back(Value(b)); }
  void Bool(const char *key, bool b) {
    Key(key);
    Bool(b);
  }

  void IndirectInt(int64_t i) {
    PushIndirect(i, FBT_INDIRECT_INT, WidthI(i));
  }
  void IndirectInt(const char *key, int64_t i) {
    Key(key);
    IndirectInt(i);
  }

  void IndirectUInt(uint64_t u) {
    PushIndirect(u, FBT_INDIRECT_UINT, WidthU(u));
  }
  void IndirectUInt(const char *key, uint64_t u) {
    Key(key);
    IndirectUInt(u);
  }

  void IndirectFloat(float f) {
    PushIndirect(f, FBT_INDIRECT_FLOAT, BIT_WIDTH_32);
  }
  void IndirectFloat(const char *key, float f) {
    Key(key);
    IndirectFloat(f);
  }

  void IndirectDouble(double f) {
    PushIndirect(f, FBT_INDIRECT_FLOAT, WidthF(f));
  }
  void IndirectDouble(const char *key, double d) {
    Key(key);
    IndirectDouble(d);
  }

  size_t Key(const char *str, size_t len) {
    auto sloc = buf_.size();
    WriteBytes(str, len + 1);
    if (flags_ & BUILDER_FLAG_SHARE_KEYS) {
      auto it = key_pool.find(sloc);
      if (it != key_pool.end()) {
        // Already in the buffer. Remove key we just serialized, and use
        // existing offset instead.
        buf_.resize(sloc);
        sloc = *it;
      } else {
        key_pool.insert(sloc);
      }
    }
    stack_.push_back(Value(static_cast<uint64_t>(sloc), FBT_KEY, BIT_WIDTH_8));
    return sloc;
  }

  size_t Key(const char *str) { return Key(str, strlen(str)); }
  size_t Key(const std::string &str) { return Key(str.c_str(), str.size()); }

  size_t String(const char *str, size_t len) {
    auto reset_to = buf_.size();
    auto sloc = CreateBlob(str, len, 1, FBT_STRING);
    if (flags_ & BUILDER_FLAG_SHARE_STRINGS) {
      StringOffset so(sloc, len);
      auto it = string_pool.find(so);
      if (it != string_pool.end()) {
        // Already in the buffer. Remove string we just serialized, and use
        // existing offset instead.
        buf_.resize(reset_to);
        sloc = it->first;
        stack_.back().u_ = sloc;
      } else {
        string_pool.insert(so);
      }
    }
    return sloc;
  }
  size_t String(const char *str) { return String(str, strlen(str)); }
  size_t String(const std::string &str) {
    return String(str.c_str(), str.size());
  }
  void String(const flexbuffers::String &str) {
    String(str.c_str(), str.length());
  }

  void String(const char *key, const char *str) {
    Key(key);
    String(str);
  }
  void String(const char *key, const std::string &str) {
    Key(key);
    String(str);
  }
  void String(const char *key, const flexbuffers::String &str) {
    Key(key);
    String(str);
  }

  size_t Blob(const void *data, size_t len) {
    return CreateBlob(data, len, 0, FBT_BLOB);
  }
  size_t Blob(const std::vector<uint8_t> &v) {
    return CreateBlob(flatbuffers::vector_data(v), v.size(), 0, FBT_BLOB);
  }

  // TODO(wvo): support all the FlexBuffer types (like flexbuffers::String),
  // e.g. Vector etc. Also in overloaded versions.
  // Also some FlatBuffers types?

  size_t StartVector() { return stack_.size(); }
  size_t StartVector(const char *key) {
    Key(key);
    return stack_.size();
  }
  size_t StartMap() { return stack_.size(); }
  size_t StartMap(const char *key) {
    Key(key);
    return stack_.size();
  }

  // TODO(wvo): allow this to specify an aligment greater than the natural
  // alignment.
  size_t EndVector(size_t start, bool typed, bool fixed) {
    auto vec = CreateVector(start, stack_.size() - start, 1, typed, fixed);
    // Remove temp elements and return vector.
    stack_.resize(start);
    stack_.push_back(vec);
    return static_cast<size_t>(vec.u_);
  }

  size_t EndMap(size_t start) {
    // We should have interleaved keys and values on the stack.
    // Make sure it is an even number:
    auto len = stack_.size() - start;
    FLATBUFFERS_ASSERT(!(len & 1));
    len /= 2;
    // Make sure keys are all strings:
    for (auto key = start; key < stack_.size(); key += 2) {
      FLATBUFFERS_ASSERT(stack_[key].type_ == FBT_KEY);
    }
    // Now sort values, so later we can do a binary seach lookup.
    // We want to sort 2 array elements at a time.
    struct TwoValue {
      Value key;
      Value val;
    };
    // TODO(wvo): strict aliasing?
    // TODO(wvo): allow the caller to indicate the data is already sorted
    // for maximum efficiency? With an assert to check sortedness to make sure
    // we're not breaking binary search.
    // Or, we can track if the map is sorted as keys are added which would be
    // be quite cheap (cheaper than checking it here), so we can skip this
    // step automatically when appliccable, and encourage people to write in
    // sorted fashion.
    // std::sort is typically already a lot faster on sorted data though.
    auto dict =
        reinterpret_cast<TwoValue *>(flatbuffers::vector_data(stack_) + start);
    std::sort(dict, dict + len,
              [&](const TwoValue &a, const TwoValue &b) -> bool {
                auto as = reinterpret_cast<const char *>(
                    flatbuffers::vector_data(buf_) + a.key.u_);
                auto bs = reinterpret_cast<const char *>(
                    flatbuffers::vector_data(buf_) + b.key.u_);
                auto comp = strcmp(as, bs);
                // If this assertion hits, you've added two keys with the same
                // value to this map.
                // TODO: Have to check for pointer equality, as some sort
                // implementation apparently call this function with the same
                // element?? Why?
                FLATBUFFERS_ASSERT(comp || &a == &b);
                return comp < 0;
              });
    // First create a vector out of all keys.
    // TODO(wvo): if kBuilderFlagShareKeyVectors is true, see if we can share
    // the first vector.
    auto keys = CreateVector(start, len, 2, true, false);
    auto vec = CreateVector(start + 1, len, 2, false, false, &keys);
    // Remove temp elements and return map.
    stack_.resize(start);
    stack_.push_back(vec);
    return static_cast<size_t>(vec.u_);
  }

  template<typename F> size_t Vector(F f) {
    auto start = StartVector();
    f();
    return EndVector(start, false, false);
  }
  template<typename F, typename T> size_t Vector(F f, T &state) {
    auto start = StartVector();
    f(state);
    return EndVector(start, false, false);
  }
  template<typename F> size_t Vector(const char *key, F f) {
    auto start = StartVector(key);
    f();
    return EndVector(start, false, false);
  }
  template<typename F, typename T>
  size_t Vector(const char *key, F f, T &state) {
    auto start = StartVector(key);
    f(state);
    return EndVector(start, false, false);
  }

  template<typename T> void Vector(const T *elems, size_t len) {
    if (flatbuffers::is_scalar<T>::value) {
      // This path should be a lot quicker and use less space.
      ScalarVector(elems, len, false);
    } else {
      auto start = StartVector();
      for (size_t i = 0; i < len; i++) Add(elems[i]);
      EndVector(start, false, false);
    }
  }
  template<typename T>
  void Vector(const char *key, const T *elems, size_t len) {
    Key(key);
    Vector(elems, len);
  }
  template<typename T> void Vector(const std::vector<T> &vec) {
    Vector(flatbuffers::vector_data(vec), vec.size());
  }

  template<typename F> size_t TypedVector(F f) {
    auto start = StartVector();
    f();
    return EndVector(start, true, false);
  }
  template<typename F, typename T> size_t TypedVector(F f, T &state) {
    auto start = StartVector();
    f(state);
    return EndVector(start, true, false);
  }
  template<typename F> size_t TypedVector(const char *key, F f) {
    auto start = StartVector(key);
    f();
    return EndVector(start, true, false);
  }
  template<typename F, typename T>
  size_t TypedVector(const char *key, F f, T &state) {
    auto start = StartVector(key);
    f(state);
    return EndVector(start, true, false);
  }

  template<typename T> size_t FixedTypedVector(const T *elems, size_t len) {
    // We only support a few fixed vector lengths. Anything bigger use a
    // regular typed vector.
    FLATBUFFERS_ASSERT(len >= 2 && len <= 4);
    // And only scalar values.
    static_assert(flatbuffers::is_scalar<T>::value, "Unrelated types");
    return ScalarVector(elems, len, true);
  }

  template<typename T>
  size_t FixedTypedVector(const char *key, const T *elems, size_t len) {
    Key(key);
    return FixedTypedVector(elems, len);
  }

  template<typename F> size_t Map(F f) {
    auto start = StartMap();
    f();
    return EndMap(start);
  }
  template<typename F, typename T> size_t Map(F f, T &state) {
    auto start = StartMap();
    f(state);
    return EndMap(start);
  }
  template<typename F> size_t Map(const char *key, F f) {
    auto start = StartMap(key);
    f();
    return EndMap(start);
  }
  template<typename F, typename T> size_t Map(const char *key, F f, T &state) {
    auto start = StartMap(key);
    f(state);
    return EndMap(start);
  }
  template<typename T> void Map(const std::map<std::string, T> &map) {
    auto start = StartMap();
    for (auto it = map.begin(); it != map.end(); ++it)
      Add(it->first.c_str(), it->second);
    EndMap(start);
  }

  // If you wish to share a value explicitly (a value not shared automatically
  // through one of the BUILDER_FLAG_SHARE_* flags) you can do so with these
  // functions. Or if you wish to turn those flags off for performance reasons
  // and still do some explicit sharing. For example:
  // builder.IndirectDouble(M_PI);
  // auto id = builder.LastValue();  // Remember where we stored it.
  // .. more code goes here ..
  // builder.ReuseValue(id);  // Refers to same double by offset.
  // LastValue works regardless of wether the value has a key or not.
  // Works on any data type.
  struct Value;
  Value LastValue() { return stack_.back(); }
  void ReuseValue(Value v) {
    stack_.push_back(v);
  }
  void ReuseValue(const char *key, Value v) {
    Key(key);
    ReuseValue(v);
  }

  // Overloaded Add that tries to call the correct function above.
  void Add(int8_t i) { Int(i); }
  void Add(int16_t i) { Int(i); }
  void Add(int32_t i) { Int(i); }
  void Add(int64_t i) { Int(i); }
  void Add(uint8_t u) { UInt(u); }
  void Add(uint16_t u) { UInt(u); }
  void Add(uint32_t u) { UInt(u); }
  void Add(uint64_t u) { UInt(u); }
  void Add(float f) { Float(f); }
  void Add(double d) { Double(d); }
  void Add(bool b) { Bool(b); }
  void Add(const char *str) { String(str); }
  void Add(const std::string &str) { String(str); }
  void Add(const flexbuffers::String &str) { String(str); }

  template<typename T> void Add(const std::vector<T> &vec) { Vector(vec); }

  template<typename T> void Add(const char *key, const T &t) {
    Key(key);
    Add(t);
  }

  template<typename T> void Add(const std::map<std::string, T> &map) {
    Map(map);
  }

  template<typename T> void operator+=(const T &t) { Add(t); }

  // This function is useful in combination with the Mutate* functions above.
  // It forces elements of vectors and maps to have a minimum size, such that
  // they can later be updated without failing.
  // Call with no arguments to reset.
  void ForceMinimumBitWidth(BitWidth bw = BIT_WIDTH_8) {
    force_min_bit_width_ = bw;
  }

  void Finish() {
    // If you hit this assert, you likely have objects that were never included
    // in a parent. You need to have exactly one root to finish a buffer.
    // Check your Start/End calls are matched, and all objects are inside
    // some other object.
    FLATBUFFERS_ASSERT(stack_.size() == 1);

    // Write root value.
    auto byte_width = Align(stack_[0].ElemWidth(buf_.size(), 0));
    WriteAny(stack_[0], byte_width);
    // Write root type.
    Write(stack_[0].StoredPackedType(), 1);
    // Write root size. Normally determined by parent, but root has no parent :)
    Write(byte_width, 1);

    finished_ = true;
  }

 private:
  void Finished() const {
    // If you get this assert, you're attempting to get access a buffer
    // which hasn't been finished yet. Be sure to call
    // Builder::Finish with your root object.
    FLATBUFFERS_ASSERT(finished_);
  }

  // Align to prepare for writing a scalar with a certain size.
  uint8_t Align(BitWidth alignment) {
    auto byte_width = 1U << alignment;
    buf_.insert(buf_.end(), flatbuffers::PaddingBytes(buf_.size(), byte_width),
                0);
    return static_cast<uint8_t>(byte_width);
  }

  void WriteBytes(const void *val, size_t size) {
    buf_.insert(buf_.end(), reinterpret_cast<const uint8_t *>(val),
                reinterpret_cast<const uint8_t *>(val) + size);
  }

  template<typename T> void Write(T val, size_t byte_width) {
    FLATBUFFERS_ASSERT(sizeof(T) >= byte_width);
    val = flatbuffers::EndianScalar(val);
    WriteBytes(&val, byte_width);
  }

  void WriteDouble(double f, uint8_t byte_width) {
    switch (byte_width) {
      case 8: Write(f, byte_width); break;
      case 4: Write(static_cast<float>(f), byte_width); break;
      // case 2: Write(static_cast<half>(f), byte_width); break;
      // case 1: Write(static_cast<quarter>(f), byte_width); break;
      default: FLATBUFFERS_ASSERT(0);
    }
  }

  void WriteOffset(uint64_t o, uint8_t byte_width) {
    auto reloff = buf_.size() - o;
    FLATBUFFERS_ASSERT(byte_width == 8 || reloff < 1ULL << (byte_width * 8));
    Write(reloff, byte_width);
  }

  template<typename T> void PushIndirect(T val, Type type, BitWidth bit_width) {
    auto byte_width = Align(bit_width);
    auto iloc = buf_.size();
    Write(val, byte_width);
    stack_.push_back(Value(static_cast<uint64_t>(iloc), type, bit_width));
  }

  static BitWidth WidthB(size_t byte_width) {
    switch (byte_width) {
      case 1: return BIT_WIDTH_8;
      case 2: return BIT_WIDTH_16;
      case 4: return BIT_WIDTH_32;
      case 8: return BIT_WIDTH_64;
      default: FLATBUFFERS_ASSERT(false); return BIT_WIDTH_64;
    }
  }

  template<typename T> static Type GetScalarType() {
    static_assert(flatbuffers::is_scalar<T>::value, "Unrelated types");
    return flatbuffers::is_floating_point<T>::value
               ? FBT_FLOAT
               : flatbuffers::is_same<T, bool>::value
                     ? FBT_BOOL
                     : (flatbuffers::is_unsigned<T>::value ? FBT_UINT
                                                           : FBT_INT);
  }

 public:
  // This was really intended to be private, except for LastValue/ReuseValue.
  struct Value {
    union {
      int64_t i_;
      uint64_t u_;
      double f_;
    };

    Type type_;

    // For scalars: of itself, for vector: of its elements, for string: length.
    BitWidth min_bit_width_;

    Value() : i_(0), type_(FBT_NULL), min_bit_width_(BIT_WIDTH_8) {}

    Value(bool b)
        : u_(static_cast<uint64_t>(b)),
          type_(FBT_BOOL),
          min_bit_width_(BIT_WIDTH_8) {}

    Value(int64_t i, Type t, BitWidth bw)
        : i_(i), type_(t), min_bit_width_(bw) {}
    Value(uint64_t u, Type t, BitWidth bw)
        : u_(u), type_(t), min_bit_width_(bw) {}

    Value(float f) : f_(f), type_(FBT_FLOAT), min_bit_width_(BIT_WIDTH_32) {}
    Value(double f) : f_(f), type_(FBT_FLOAT), min_bit_width_(WidthF(f)) {}

    uint8_t StoredPackedType(BitWidth parent_bit_width_ = BIT_WIDTH_8) const {
      return PackedType(StoredWidth(parent_bit_width_), type_);
    }

    BitWidth ElemWidth(size_t buf_size, size_t elem_index) const {
      if (IsInline(type_)) {
        return min_bit_width_;
      } else {
        // We have an absolute offset, but want to store a relative offset
        // elem_index elements beyond the current buffer end. Since whether
        // the relative offset fits in a certain byte_width depends on
        // the size of the elements before it (and their alignment), we have
        // to test for each size in turn.
        for (size_t byte_width = 1;
             byte_width <= sizeof(flatbuffers::largest_scalar_t);
             byte_width *= 2) {
          // Where are we going to write this offset?
          auto offset_loc = buf_size +
                            flatbuffers::PaddingBytes(buf_size, byte_width) +
                            elem_index * byte_width;
          // Compute relative offset.
          auto offset = offset_loc - u_;
          // Does it fit?
          auto bit_width = WidthU(offset);
          if (static_cast<size_t>(static_cast<size_t>(1U) << bit_width) ==
              byte_width)
            return bit_width;
        }
        FLATBUFFERS_ASSERT(false);  // Must match one of the sizes above.
        return BIT_WIDTH_64;
      }
    }

    BitWidth StoredWidth(BitWidth parent_bit_width_ = BIT_WIDTH_8) const {
      if (IsInline(type_)) {
        return (std::max)(min_bit_width_, parent_bit_width_);
      } else {
        return min_bit_width_;
      }
    }
  };

 private:
  void WriteAny(const Value &val, uint8_t byte_width) {
    switch (val.type_) {
      case FBT_NULL:
      case FBT_INT: Write(val.i_, byte_width); break;
      case FBT_BOOL:
      case FBT_UINT: Write(val.u_, byte_width); break;
      case FBT_FLOAT: WriteDouble(val.f_, byte_width); break;
      default: WriteOffset(val.u_, byte_width); break;
    }
  }

  size_t CreateBlob(const void *data, size_t len, size_t trailing, Type type) {
    auto bit_width = WidthU(len);
    auto byte_width = Align(bit_width);
    Write<uint64_t>(len, byte_width);
    auto sloc = buf_.size();
    WriteBytes(data, len + trailing);
    stack_.push_back(Value(static_cast<uint64_t>(sloc), type, bit_width));
    return sloc;
  }

  template<typename T>
  size_t ScalarVector(const T *elems, size_t len, bool fixed) {
    auto vector_type = GetScalarType<T>();
    auto byte_width = sizeof(T);
    auto bit_width = WidthB(byte_width);
    // If you get this assert, you're trying to write a vector with a size
    // field that is bigger than the scalars you're trying to write (e.g. a
    // byte vector > 255 elements). For such types, write a "blob" instead.
    // TODO: instead of asserting, could write vector with larger elements
    // instead, though that would be wasteful.
    FLATBUFFERS_ASSERT(WidthU(len) <= bit_width);
    if (!fixed) Write<uint64_t>(len, byte_width);
    auto vloc = buf_.size();
    for (size_t i = 0; i < len; i++) Write(elems[i], byte_width);
    stack_.push_back(Value(static_cast<uint64_t>(vloc),
                           ToTypedVector(vector_type, fixed ? len : 0),
                           bit_width));
    return vloc;
  }

  Value CreateVector(size_t start, size_t vec_len, size_t step, bool typed,
                     bool fixed, const Value *keys = nullptr) {
    FLATBUFFERS_ASSERT(!fixed || typed); // typed=false, fixed=true combination is not supported.
    // Figure out smallest bit width we can store this vector with.
    auto bit_width = (std::max)(force_min_bit_width_, WidthU(vec_len));
    auto prefix_elems = 1;
    if (keys) {
      // If this vector is part of a map, we will pre-fix an offset to the keys
      // to this vector.
      bit_width = (std::max)(bit_width, keys->ElemWidth(buf_.size(), 0));
      prefix_elems += 2;
    }
    Type vector_type = FBT_KEY;
    // Check bit widths and types for all elements.
    for (size_t i = start; i < stack_.size(); i += step) {
      auto elem_width = stack_[i].ElemWidth(buf_.size(), i + prefix_elems);
      bit_width = (std::max)(bit_width, elem_width);
      if (typed) {
        if (i == start) {
          vector_type = stack_[i].type_;
        } else {
          // If you get this assert, you are writing a typed vector with
          // elements that are not all the same type.
          FLATBUFFERS_ASSERT(vector_type == stack_[i].type_);
        }
      }
    }
    // If you get this assert, your fixed types are not one of:
    // Int / UInt / Float / Key.
    FLATBUFFERS_ASSERT(!fixed || IsTypedVectorElementType(vector_type));
    auto byte_width = Align(bit_width);
    // Write vector. First the keys width/offset if available, and size.
    if (keys) {
      WriteOffset(keys->u_, byte_width);
      Write<uint64_t>(1ULL << keys->min_bit_width_, byte_width);
    }
    if (!fixed) Write<uint64_t>(vec_len, byte_width);
    // Then the actual data.
    auto vloc = buf_.size();
    for (size_t i = start; i < stack_.size(); i += step) {
      WriteAny(stack_[i], byte_width);
    }
    // Then the types.
    if (!typed) {
      for (size_t i = start; i < stack_.size(); i += step) {
        buf_.push_back(stack_[i].StoredPackedType(bit_width));
      }
    }
    return Value(static_cast<uint64_t>(vloc),
                 keys ? FBT_MAP
                      : (typed ? ToTypedVector(vector_type, fixed ? vec_len : 0)
                               : FBT_VECTOR),
                 bit_width);
  }

  // You shouldn't really be copying instances of this class.
  Builder(const Builder &);
  Builder &operator=(const Builder &);

  std::vector<uint8_t> buf_;
  std::vector<Value> stack_;

  bool finished_;

  BuilderFlag flags_;

  BitWidth force_min_bit_width_;

  struct KeyOffsetCompare {
    explicit KeyOffsetCompare(const std::vector<uint8_t> &buf) : buf_(&buf) {}
    bool operator()(size_t a, size_t b) const {
      auto stra =
          reinterpret_cast<const char *>(flatbuffers::vector_data(*buf_) + a);
      auto strb =
          reinterpret_cast<const char *>(flatbuffers::vector_data(*buf_) + b);
      return strcmp(stra, strb) < 0;
    }
    const std::vector<uint8_t> *buf_;
  };

  typedef std::pair<size_t, size_t> StringOffset;
  struct StringOffsetCompare {
    explicit StringOffsetCompare(const std::vector<uint8_t> &buf) : buf_(&buf) {}
    bool operator()(const StringOffset &a, const StringOffset &b) const {
      auto stra = reinterpret_cast<const char *>(
          flatbuffers::vector_data(*buf_) + a.first);
      auto strb = reinterpret_cast<const char *>(
          flatbuffers::vector_data(*buf_) + b.first);
      return strncmp(stra, strb, (std::min)(a.second, b.second) + 1) < 0;
    }
    const std::vector<uint8_t> *buf_;
  };

  typedef std::set<size_t, KeyOffsetCompare> KeyOffsetMap;
  typedef std::set<StringOffset, StringOffsetCompare> StringOffsetMap;

  KeyOffsetMap key_pool;
  StringOffsetMap string_pool;
};

}  // namespace flexbuffers

#  if defined(_MSC_VER)
#    pragma warning(pop)
#  endif

#endif  // FLATBUFFERS_FLEXBUFFERS_H_
