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

#include <algorithm>
#include <list>
#include <string>
#include <utility>

#include <cmath>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

namespace flatbuffers {

// Reflects the version at the compiling time of binary(lib/dll/so).
const char *FLATBUFFERS_VERSION() {
  // clang-format off
  return
      FLATBUFFERS_STRING(FLATBUFFERS_VERSION_MAJOR) "."
      FLATBUFFERS_STRING(FLATBUFFERS_VERSION_MINOR) "."
      FLATBUFFERS_STRING(FLATBUFFERS_VERSION_REVISION);
  // clang-format on
}

const double kPi = 3.14159265358979323846;

const char *const kTypeNames[] = {
// clang-format off
  #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
    CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
    IDLTYPE,
    FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
  #undef FLATBUFFERS_TD
  // clang-format on
  nullptr
};

const char kTypeSizes[] = {
// clang-format off
  #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
      CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
      sizeof(CTYPE),
    FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
  #undef FLATBUFFERS_TD
  // clang-format on
};

// The enums in the reflection schema should match the ones we use internally.
// Compare the last element to check if these go out of sync.
static_assert(BASE_TYPE_UNION == static_cast<BaseType>(reflection::Union),
              "enums don't match");

// Any parsing calls have to be wrapped in this macro, which automates
// handling of recursive error checking a bit. It will check the received
// CheckedError object, and return straight away on error.
#define ECHECK(call)           \
  {                            \
    auto ce = (call);          \
    if (ce.Check()) return ce; \
  }

// These two functions are called hundreds of times below, so define a short
// form:
#define NEXT() ECHECK(Next())
#define EXPECT(tok) ECHECK(Expect(tok))

static bool ValidateUTF8(const std::string &str) {
  const char *s = &str[0];
  const char *const sEnd = s + str.length();
  while (s < sEnd) {
    if (FromUTF8(&s) < 0) { return false; }
  }
  return true;
}

// Convert an underscore_based_indentifier in to camelCase.
// Also uppercases the first character if first is true.
std::string MakeCamel(const std::string &in, bool first) {
  std::string s;
  for (size_t i = 0; i < in.length(); i++) {
    if (!i && first)
      s += static_cast<char>(toupper(in[0]));
    else if (in[i] == '_' && i + 1 < in.length())
      s += static_cast<char>(toupper(in[++i]));
    else
      s += in[i];
  }
  return s;
}

// Convert an underscore_based_identifier in to screaming snake case.
std::string MakeScreamingCamel(const std::string &in) {
  std::string s;
  for (size_t i = 0; i < in.length(); i++) {
    if (in[i] != '_')
      s += static_cast<char>(toupper(in[i]));
    else
      s += in[i];
  }
  return s;
}

void DeserializeDoc( std::vector<std::string> &doc,
                     const Vector<Offset<String>> *documentation) {
  if (documentation == nullptr) return;
  for (uoffset_t index = 0; index < documentation->size(); index++)
    doc.push_back(documentation->Get(index)->str());
}

void Parser::Message(const std::string &msg) {
  if (!error_.empty()) error_ += "\n";  // log all warnings and errors
  error_ += file_being_parsed_.length() ? AbsolutePath(file_being_parsed_) : "";
  // clang-format off

  #ifdef _WIN32  // MSVC alike
    error_ +=
        "(" + NumToString(line_) + ", " + NumToString(CursorPosition()) + ")";
  #else  // gcc alike
    if (file_being_parsed_.length()) error_ += ":";
    error_ += NumToString(line_) + ": " + NumToString(CursorPosition());
  #endif
  // clang-format on
  error_ += ": " + msg;
}

void Parser::Warning(const std::string &msg) { Message("warning: " + msg); }

CheckedError Parser::Error(const std::string &msg) {
  Message("error: " + msg);
  return CheckedError(true);
}

inline CheckedError NoError() { return CheckedError(false); }

CheckedError Parser::RecurseError() {
  return Error("maximum parsing recursion of " +
               NumToString(FLATBUFFERS_MAX_PARSING_DEPTH) + " reached");
}

template<typename F> CheckedError Parser::Recurse(F f) {
  if (recurse_protection_counter >= (FLATBUFFERS_MAX_PARSING_DEPTH))
    return RecurseError();
  recurse_protection_counter++;
  auto ce = f();
  recurse_protection_counter--;
  return ce;
}

template<typename T> std::string TypeToIntervalString() {
  return "[" + NumToString((flatbuffers::numeric_limits<T>::lowest)()) + "; " +
         NumToString((flatbuffers::numeric_limits<T>::max)()) + "]";
}

// atot: template version of atoi/atof: convert a string to an instance of T.
template<typename T>
inline CheckedError atot(const char *s, Parser &parser, T *val) {
  auto done = StringToNumber(s, val);
  if (done) return NoError();
  if (0 == *val)
    return parser.Error("invalid number: \"" + std::string(s) + "\"");
  else
    return parser.Error("invalid number: \"" + std::string(s) + "\"" +
                        ", constant does not fit " + TypeToIntervalString<T>());
}
template<>
inline CheckedError atot<Offset<void>>(const char *s, Parser &parser,
                                       Offset<void> *val) {
  (void)parser;
  *val = Offset<void>(atoi(s));
  return NoError();
}

std::string Namespace::GetFullyQualifiedName(const std::string &name,
                                             size_t max_components) const {
  // Early exit if we don't have a defined namespace.
  if (components.empty() || !max_components) { return name; }
  std::string stream_str;
  for (size_t i = 0; i < std::min(components.size(), max_components); i++) {
    if (i) { stream_str += '.'; }
    stream_str += std::string(components[i]);
  }
  if (name.length()) {
    stream_str += '.';
    stream_str += name;
  }
  return stream_str;
}

// Declare tokens we'll use. Single character tokens are represented by their
// ascii character code (e.g. '{'), others above 256.
// clang-format off
#define FLATBUFFERS_GEN_TOKENS(TD) \
  TD(Eof, 256, "end of file") \
  TD(StringConstant, 257, "string constant") \
  TD(IntegerConstant, 258, "integer constant") \
  TD(FloatConstant, 259, "float constant") \
  TD(Identifier, 260, "identifier")
#ifdef __GNUC__
__extension__  // Stop GCC complaining about trailing comma with -Wpendantic.
#endif
enum {
  #define FLATBUFFERS_TOKEN(NAME, VALUE, STRING) kToken ## NAME = VALUE,
    FLATBUFFERS_GEN_TOKENS(FLATBUFFERS_TOKEN)
  #undef FLATBUFFERS_TOKEN
};

static std::string TokenToString(int t) {
  static const char * const tokens[] = {
    #define FLATBUFFERS_TOKEN(NAME, VALUE, STRING) STRING,
      FLATBUFFERS_GEN_TOKENS(FLATBUFFERS_TOKEN)
    #undef FLATBUFFERS_TOKEN
    #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
      CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
      IDLTYPE,
      FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
    #undef FLATBUFFERS_TD
  };
  if (t < 256) {  // A single ascii char token.
    std::string s;
    s.append(1, static_cast<char>(t));
    return s;
  } else {       // Other tokens.
    return tokens[t - 256];
  }
}
// clang-format on

std::string Parser::TokenToStringId(int t) const {
  return t == kTokenIdentifier ? attribute_ : TokenToString(t);
}

// Parses exactly nibbles worth of hex digits into a number, or error.
CheckedError Parser::ParseHexNum(int nibbles, uint64_t *val) {
  FLATBUFFERS_ASSERT(nibbles > 0);
  for (int i = 0; i < nibbles; i++)
    if (!is_xdigit(cursor_[i]))
      return Error("escape code must be followed by " + NumToString(nibbles) +
                   " hex digits");
  std::string target(cursor_, cursor_ + nibbles);
  *val = StringToUInt(target.c_str(), 16);
  cursor_ += nibbles;
  return NoError();
}

CheckedError Parser::SkipByteOrderMark() {
  if (static_cast<unsigned char>(*cursor_) != 0xef) return NoError();
  cursor_++;
  if (static_cast<unsigned char>(*cursor_) != 0xbb)
    return Error("invalid utf-8 byte order mark");
  cursor_++;
  if (static_cast<unsigned char>(*cursor_) != 0xbf)
    return Error("invalid utf-8 byte order mark");
  cursor_++;
  return NoError();
}

static inline bool IsIdentifierStart(char c) {
  return is_alpha(c) || (c == '_');
}

CheckedError Parser::Next() {
  doc_comment_.clear();
  bool seen_newline = cursor_ == source_;
  attribute_.clear();
  attr_is_trivial_ascii_string_ = true;
  for (;;) {
    char c = *cursor_++;
    token_ = c;
    switch (c) {
      case '\0':
        cursor_--;
        token_ = kTokenEof;
        return NoError();
      case ' ':
      case '\r':
      case '\t': break;
      case '\n':
        MarkNewLine();
        seen_newline = true;
        break;
      case '{':
      case '}':
      case '(':
      case ')':
      case '[':
      case ']':
      case ',':
      case ':':
      case ';':
      case '=': return NoError();
      case '\"':
      case '\'': {
        int unicode_high_surrogate = -1;

        while (*cursor_ != c) {
          if (*cursor_ < ' ' && static_cast<signed char>(*cursor_) >= 0)
            return Error("illegal character in string constant");
          if (*cursor_ == '\\') {
            attr_is_trivial_ascii_string_ = false;  // has escape sequence
            cursor_++;
            if (unicode_high_surrogate != -1 && *cursor_ != 'u') {
              return Error(
                  "illegal Unicode sequence (unpaired high surrogate)");
            }
            switch (*cursor_) {
              case 'n':
                attribute_ += '\n';
                cursor_++;
                break;
              case 't':
                attribute_ += '\t';
                cursor_++;
                break;
              case 'r':
                attribute_ += '\r';
                cursor_++;
                break;
              case 'b':
                attribute_ += '\b';
                cursor_++;
                break;
              case 'f':
                attribute_ += '\f';
                cursor_++;
                break;
              case '\"':
                attribute_ += '\"';
                cursor_++;
                break;
              case '\'':
                attribute_ += '\'';
                cursor_++;
                break;
              case '\\':
                attribute_ += '\\';
                cursor_++;
                break;
              case '/':
                attribute_ += '/';
                cursor_++;
                break;
              case 'x': {  // Not in the JSON standard
                cursor_++;
                uint64_t val;
                ECHECK(ParseHexNum(2, &val));
                attribute_ += static_cast<char>(val);
                break;
              }
              case 'u': {
                cursor_++;
                uint64_t val;
                ECHECK(ParseHexNum(4, &val));
                if (val >= 0xD800 && val <= 0xDBFF) {
                  if (unicode_high_surrogate != -1) {
                    return Error(
                        "illegal Unicode sequence (multiple high surrogates)");
                  } else {
                    unicode_high_surrogate = static_cast<int>(val);
                  }
                } else if (val >= 0xDC00 && val <= 0xDFFF) {
                  if (unicode_high_surrogate == -1) {
                    return Error(
                        "illegal Unicode sequence (unpaired low surrogate)");
                  } else {
                    int code_point = 0x10000 +
                                     ((unicode_high_surrogate & 0x03FF) << 10) +
                                     (val & 0x03FF);
                    ToUTF8(code_point, &attribute_);
                    unicode_high_surrogate = -1;
                  }
                } else {
                  if (unicode_high_surrogate != -1) {
                    return Error(
                        "illegal Unicode sequence (unpaired high surrogate)");
                  }
                  ToUTF8(static_cast<int>(val), &attribute_);
                }
                break;
              }
              default: return Error("unknown escape code in string constant");
            }
          } else {  // printable chars + UTF-8 bytes
            if (unicode_high_surrogate != -1) {
              return Error(
                  "illegal Unicode sequence (unpaired high surrogate)");
            }
            // reset if non-printable
            attr_is_trivial_ascii_string_ &= check_ascii_range(*cursor_, ' ', '~');

            attribute_ += *cursor_++;
          }
        }
        if (unicode_high_surrogate != -1) {
          return Error("illegal Unicode sequence (unpaired high surrogate)");
        }
        cursor_++;
        if (!attr_is_trivial_ascii_string_ && !opts.allow_non_utf8 &&
            !ValidateUTF8(attribute_)) {
          return Error("illegal UTF-8 sequence");
        }
        token_ = kTokenStringConstant;
        return NoError();
      }
      case '/':
        if (*cursor_ == '/') {
          const char *start = ++cursor_;
          while (*cursor_ && *cursor_ != '\n' && *cursor_ != '\r') cursor_++;
          if (*start == '/') {  // documentation comment
            if (!seen_newline)
              return Error(
                  "a documentation comment should be on a line on its own");
            doc_comment_.push_back(std::string(start + 1, cursor_));
          }
          break;
        } else if (*cursor_ == '*') {
          cursor_++;
          // TODO: make nested.
          while (*cursor_ != '*' || cursor_[1] != '/') {
            if (*cursor_ == '\n') MarkNewLine();
            if (!*cursor_) return Error("end of file in comment");
            cursor_++;
          }
          cursor_ += 2;
          break;
        }
        FLATBUFFERS_FALLTHROUGH(); // else fall thru
      default:
        const auto has_sign = (c == '+') || (c == '-');
        // '-'/'+' and following identifier - can be a predefined constant like:
        // NAN, INF, PI, etc.
        if (IsIdentifierStart(c) || (has_sign && IsIdentifierStart(*cursor_))) {
          // Collect all chars of an identifier:
          const char *start = cursor_ - 1;
          while (IsIdentifierStart(*cursor_) || is_digit(*cursor_)) cursor_++;
          attribute_.append(start, cursor_);
          token_ = has_sign ? kTokenStringConstant : kTokenIdentifier;
          return NoError();
        }

        auto dot_lvl = (c == '.') ? 0 : 1;  // dot_lvl==0 <=> exactly one '.' seen
        if (!dot_lvl && !is_digit(*cursor_)) return NoError(); // enum?
        // Parser accepts hexadecimal-floating-literal (see C++ 5.13.4).
        if (is_digit(c) || has_sign || !dot_lvl) {
          const auto start = cursor_ - 1;
          auto start_digits = !is_digit(c) ? cursor_ : cursor_ - 1;
          if (!is_digit(c) && is_digit(*cursor_)){
            start_digits = cursor_; // see digit in cursor_ position
            c = *cursor_++;
          }
          // hex-float can't begind with '.'
          auto use_hex = dot_lvl && (c == '0') && is_alpha_char(*cursor_, 'X');
          if (use_hex) start_digits = ++cursor_;  // '0x' is the prefix, skip it
          // Read an integer number or mantisa of float-point number.
          do {
            if (use_hex) {
              while (is_xdigit(*cursor_)) cursor_++;
            } else {
              while (is_digit(*cursor_)) cursor_++;
            }
          } while ((*cursor_ == '.') && (++cursor_) && (--dot_lvl >= 0));
          // Exponent of float-point number.
          if ((dot_lvl >= 0) && (cursor_ > start_digits)) {
            // The exponent suffix of hexadecimal float number is mandatory.
            if (use_hex && !dot_lvl) start_digits = cursor_;
            if ((use_hex && is_alpha_char(*cursor_, 'P')) ||
                is_alpha_char(*cursor_, 'E')) {
              dot_lvl = 0;  // Emulate dot to signal about float-point number.
              cursor_++;
              if (*cursor_ == '+' || *cursor_ == '-') cursor_++;
              start_digits = cursor_;  // the exponent-part has to have digits
              // Exponent is decimal integer number
              while (is_digit(*cursor_)) cursor_++;
              if (*cursor_ == '.') {
                cursor_++;  // If see a dot treat it as part of invalid number.
                dot_lvl = -1;  // Fall thru to Error().
              }
            }
          }
          // Finalize.
          if ((dot_lvl >= 0) && (cursor_ > start_digits)) {
            attribute_.append(start, cursor_);
            token_ = dot_lvl ? kTokenIntegerConstant : kTokenFloatConstant;
            return NoError();
          } else {
            return Error("invalid number: " + std::string(start, cursor_));
          }
        }
        std::string ch;
        ch = c;
        if (false == check_ascii_range(c, ' ', '~')) ch = "code: " + NumToString(c);
        return Error("illegal character: " + ch);
    }
  }
}

// Check if a given token is next.
bool Parser::Is(int t) const { return t == token_; }

bool Parser::IsIdent(const char *id) const {
  return token_ == kTokenIdentifier && attribute_ == id;
}

// Expect a given token to be next, consume it, or error if not present.
CheckedError Parser::Expect(int t) {
  if (t != token_) {
    return Error("expecting: " + TokenToString(t) +
                 " instead got: " + TokenToStringId(token_));
  }
  NEXT();
  return NoError();
}

CheckedError Parser::ParseNamespacing(std::string *id, std::string *last) {
  while (Is('.')) {
    NEXT();
    *id += ".";
    *id += attribute_;
    if (last) *last = attribute_;
    EXPECT(kTokenIdentifier);
  }
  return NoError();
}

EnumDef *Parser::LookupEnum(const std::string &id) {
  // Search thru parent namespaces.
  for (int components = static_cast<int>(current_namespace_->components.size());
       components >= 0; components--) {
    auto ed = enums_.Lookup(
        current_namespace_->GetFullyQualifiedName(id, components));
    if (ed) return ed;
  }
  return nullptr;
}

StructDef *Parser::LookupStruct(const std::string &id) const {
  auto sd = structs_.Lookup(id);
  if (sd) sd->refcount++;
  return sd;
}

CheckedError Parser::ParseTypeIdent(Type &type) {
  std::string id = attribute_;
  EXPECT(kTokenIdentifier);
  ECHECK(ParseNamespacing(&id, nullptr));
  auto enum_def = LookupEnum(id);
  if (enum_def) {
    type = enum_def->underlying_type;
    if (enum_def->is_union) type.base_type = BASE_TYPE_UNION;
  } else {
    type.base_type = BASE_TYPE_STRUCT;
    type.struct_def = LookupCreateStruct(id);
  }
  return NoError();
}

// Parse any IDL type.
CheckedError Parser::ParseType(Type &type) {
  if (token_ == kTokenIdentifier) {
    if (IsIdent("bool")) {
      type.base_type = BASE_TYPE_BOOL;
      NEXT();
    } else if (IsIdent("byte") || IsIdent("int8")) {
      type.base_type = BASE_TYPE_CHAR;
      NEXT();
    } else if (IsIdent("ubyte") || IsIdent("uint8")) {
      type.base_type = BASE_TYPE_UCHAR;
      NEXT();
    } else if (IsIdent("short") || IsIdent("int16")) {
      type.base_type = BASE_TYPE_SHORT;
      NEXT();
    } else if (IsIdent("ushort") || IsIdent("uint16")) {
      type.base_type = BASE_TYPE_USHORT;
      NEXT();
    } else if (IsIdent("int") || IsIdent("int32")) {
      type.base_type = BASE_TYPE_INT;
      NEXT();
    } else if (IsIdent("uint") || IsIdent("uint32")) {
      type.base_type = BASE_TYPE_UINT;
      NEXT();
    } else if (IsIdent("long") || IsIdent("int64")) {
      type.base_type = BASE_TYPE_LONG;
      NEXT();
    } else if (IsIdent("ulong") || IsIdent("uint64")) {
      type.base_type = BASE_TYPE_ULONG;
      NEXT();
    } else if (IsIdent("float") || IsIdent("float32")) {
      type.base_type = BASE_TYPE_FLOAT;
      NEXT();
    } else if (IsIdent("double") || IsIdent("float64")) {
      type.base_type = BASE_TYPE_DOUBLE;
      NEXT();
    } else if (IsIdent("string")) {
      type.base_type = BASE_TYPE_STRING;
      NEXT();
    } else {
      ECHECK(ParseTypeIdent(type));
    }
  } else if (token_ == '[') {
    NEXT();
    Type subtype;
    ECHECK(Recurse([&]() { return ParseType(subtype); }));
    if (IsSeries(subtype)) {
      // We could support this, but it will complicate things, and it's
      // easier to work around with a struct around the inner vector.
      return Error("nested vector types not supported (wrap in table first)");
    }
    if (token_ == ':') {
      NEXT();
      if (token_ != kTokenIntegerConstant) {
        return Error("length of fixed-length array must be an integer value");
      }
      uint16_t fixed_length = 0;
      bool check = StringToNumber(attribute_.c_str(), &fixed_length);
      if (!check || fixed_length < 1) {
        return Error(
            "length of fixed-length array must be positive and fit to "
            "uint16_t type");
      }
      // Check if enum arrays are used in C++ without specifying --scoped-enums
      if ((opts.lang_to_generate & IDLOptions::kCpp) && !opts.scoped_enums &&
          IsEnum(subtype)) {
        return Error(
            "--scoped-enums must be enabled to use enum arrays in C++\n");
      }
      type = Type(BASE_TYPE_ARRAY, subtype.struct_def, subtype.enum_def,
                  fixed_length);
      NEXT();
    } else {
      type = Type(BASE_TYPE_VECTOR, subtype.struct_def, subtype.enum_def);
    }
    type.element = subtype.base_type;
    EXPECT(']');
  } else {
    return Error("illegal type syntax");
  }
  return NoError();
}

CheckedError Parser::AddField(StructDef &struct_def, const std::string &name,
                              const Type &type, FieldDef **dest) {
  auto &field = *new FieldDef();
  field.value.offset =
      FieldIndexToOffset(static_cast<voffset_t>(struct_def.fields.vec.size()));
  field.name = name;
  field.file = struct_def.file;
  field.value.type = type;
  if (struct_def.fixed) {  // statically compute the field offset
    auto size = InlineSize(type);
    auto alignment = InlineAlignment(type);
    // structs_ need to have a predictable format, so we need to align to
    // the largest scalar
    struct_def.minalign = std::max(struct_def.minalign, alignment);
    struct_def.PadLastField(alignment);
    field.value.offset = static_cast<voffset_t>(struct_def.bytesize);
    struct_def.bytesize += size;
  }
  if (struct_def.fields.Add(name, &field))
    return Error("field already exists: " + name);
  *dest = &field;
  return NoError();
}

CheckedError Parser::ParseField(StructDef &struct_def) {
  std::string name = attribute_;

  if (LookupStruct(name))
    return Error("field name can not be the same as table/struct name");

  std::vector<std::string> dc = doc_comment_;
  EXPECT(kTokenIdentifier);
  EXPECT(':');
  Type type;
  ECHECK(ParseType(type));

  if (struct_def.fixed && !IsScalar(type.base_type) && !IsStruct(type) &&
      !IsArray(type))
    return Error("structs_ may contain only scalar or struct fields");

  if (!struct_def.fixed && IsArray(type))
    return Error("fixed-length array in table must be wrapped in struct");

  if (IsArray(type) && !SupportsAdvancedArrayFeatures()) {
    return Error(
        "Arrays are not yet supported in all "
        "the specified programming languages.");
  }

  FieldDef *typefield = nullptr;
  if (type.base_type == BASE_TYPE_UNION) {
    // For union fields, add a second auto-generated field to hold the type,
    // with a special suffix.
    ECHECK(AddField(struct_def, name + UnionTypeFieldSuffix(),
                    type.enum_def->underlying_type, &typefield));
  } else if (type.base_type == BASE_TYPE_VECTOR &&
             type.element == BASE_TYPE_UNION) {
    // Only cpp, js and ts supports the union vector feature so far.
    if (!SupportsAdvancedUnionFeatures()) {
      return Error(
          "Vectors of unions are not yet supported in all "
          "the specified programming languages.");
    }
    // For vector of union fields, add a second auto-generated vector field to
    // hold the types, with a special suffix.
    Type union_vector(BASE_TYPE_VECTOR, nullptr, type.enum_def);
    union_vector.element = BASE_TYPE_UTYPE;
    ECHECK(AddField(struct_def, name + UnionTypeFieldSuffix(), union_vector,
                    &typefield));
  }

  FieldDef *field;
  ECHECK(AddField(struct_def, name, type, &field));

  if (token_ == '=') {
    NEXT();
    ECHECK(ParseSingleValue(&field->name, field->value, true));
    if (!IsScalar(type.base_type) ||
        (struct_def.fixed && field->value.constant != "0"))
      return Error(
            "default values currently only supported for scalars in tables");
  }
  // Append .0 if the value has not it (skip hex and scientific floats).
  // This suffix needed for generated C++ code.
  if (IsFloat(type.base_type)) {
    auto &text = field->value.constant;
    FLATBUFFERS_ASSERT(false == text.empty());
    auto s = text.c_str();
    while(*s == ' ') s++;
    if (*s == '-' || *s == '+') s++;
    // 1) A float constants (nan, inf, pi, etc) is a kind of identifier.
    // 2) A float number needn't ".0" at the end if it has exponent.
    if ((false == IsIdentifierStart(*s)) &&
        (std::string::npos == field->value.constant.find_first_of(".eEpP"))) {
      field->value.constant += ".0";
    }
  }
  if (type.enum_def) {
    // The type.base_type can only be scalar, union, array or vector.
    // Table, struct or string can't have enum_def.
    // Default value of union and vector in NONE, NULL translated to "0".
    FLATBUFFERS_ASSERT(IsInteger(type.base_type) ||
                       (type.base_type == BASE_TYPE_UNION) ||
                       (type.base_type == BASE_TYPE_VECTOR) ||
                       (type.base_type == BASE_TYPE_ARRAY));
    if (type.base_type == BASE_TYPE_VECTOR) {
      // Vector can't use initialization list.
      FLATBUFFERS_ASSERT(field->value.constant == "0");
    } else {
      // All unions should have the NONE ("0") enum value.
      auto in_enum = type.enum_def->attributes.Lookup("bit_flags") ||
                     type.enum_def->FindByValue(field->value.constant);
      if (false == in_enum)
        return Error("default value of " + field->value.constant +
                     " for field " + name + " is not part of enum " +
                     type.enum_def->name);
    }
  }

  field->doc_comment = dc;
  ECHECK(ParseMetaData(&field->attributes));
  field->deprecated = field->attributes.Lookup("deprecated") != nullptr;
  auto hash_name = field->attributes.Lookup("hash");
  if (hash_name) {
    switch ((type.base_type == BASE_TYPE_VECTOR) ? type.element : type.base_type) {
      case BASE_TYPE_SHORT:
      case BASE_TYPE_USHORT: {
        if (FindHashFunction16(hash_name->constant.c_str()) == nullptr)
          return Error("Unknown hashing algorithm for 16 bit types: " +
                       hash_name->constant);
        break;
      }
      case BASE_TYPE_INT:
      case BASE_TYPE_UINT: {
        if (FindHashFunction32(hash_name->constant.c_str()) == nullptr)
          return Error("Unknown hashing algorithm for 32 bit types: " +
                       hash_name->constant);
        break;
      }
      case BASE_TYPE_LONG:
      case BASE_TYPE_ULONG: {
        if (FindHashFunction64(hash_name->constant.c_str()) == nullptr)
          return Error("Unknown hashing algorithm for 64 bit types: " +
                       hash_name->constant);
        break;
      }
      default:
        return Error(
            "only short, ushort, int, uint, long and ulong data types support hashing.");
    }
  }
  auto cpp_type = field->attributes.Lookup("cpp_type");
  if (cpp_type) {
    if (!hash_name)
      return Error("cpp_type can only be used with a hashed field");
    /// forcing cpp_ptr_type to 'naked' if unset
    auto cpp_ptr_type = field->attributes.Lookup("cpp_ptr_type");
    if (!cpp_ptr_type) {
      auto val = new Value();
      val->type = cpp_type->type;
      val->constant = "naked";
      field->attributes.Add("cpp_ptr_type", val);
    }
  }
  if (field->deprecated && struct_def.fixed)
    return Error("can't deprecate fields in a struct");
  field->required = field->attributes.Lookup("required") != nullptr;
  if (field->required &&
      (struct_def.fixed || IsScalar(type.base_type)))
    return Error("only non-scalar fields in tables may be 'required'");
  field->key = field->attributes.Lookup("key") != nullptr;
  if (field->key) {
    if (struct_def.has_key) return Error("only one field may be set as 'key'");
    struct_def.has_key = true;
    if (!IsScalar(type.base_type)) {
      field->required = true;
      if (type.base_type != BASE_TYPE_STRING)
        return Error("'key' field must be string or scalar type");
    }
  }
  field->shared = field->attributes.Lookup("shared") != nullptr;
  if (field->shared && field->value.type.base_type != BASE_TYPE_STRING)
    return Error("shared can only be defined on strings");

  auto field_native_custom_alloc =
      field->attributes.Lookup("native_custom_alloc");
  if (field_native_custom_alloc)
    return Error(
        "native_custom_alloc can only be used with a table or struct "
        "definition");

  field->native_inline = field->attributes.Lookup("native_inline") != nullptr;
  if (field->native_inline && !IsStruct(field->value.type))
    return Error("native_inline can only be defined on structs");

  auto nested = field->attributes.Lookup("nested_flatbuffer");
  if (nested) {
    if (nested->type.base_type != BASE_TYPE_STRING)
      return Error(
          "nested_flatbuffer attribute must be a string (the root type)");
    if (type.base_type != BASE_TYPE_VECTOR || type.element != BASE_TYPE_UCHAR)
      return Error(
          "nested_flatbuffer attribute may only apply to a vector of ubyte");
    // This will cause an error if the root type of the nested flatbuffer
    // wasn't defined elsewhere.
    field->nested_flatbuffer = LookupCreateStruct(nested->constant);
  }

  if (field->attributes.Lookup("flexbuffer")) {
    field->flexbuffer = true;
    uses_flexbuffers_ = true;
    if (type.base_type != BASE_TYPE_VECTOR ||
        type.element != BASE_TYPE_UCHAR)
      return Error("flexbuffer attribute may only apply to a vector of ubyte");
  }

  if (typefield) {
    if (!IsScalar(typefield->value.type.base_type)) {
      // this is a union vector field
      typefield->required = field->required;
    }
    // If this field is a union, and it has a manually assigned id,
    // the automatically added type field should have an id as well (of N - 1).
    auto attr = field->attributes.Lookup("id");
    if (attr) {
      auto id = atoi(attr->constant.c_str());
      auto val = new Value();
      val->type = attr->type;
      val->constant = NumToString(id - 1);
      typefield->attributes.Add("id", val);
    }
  }

  EXPECT(';');
  return NoError();
}

CheckedError Parser::ParseString(Value &val) {
  auto s = attribute_;
  EXPECT(kTokenStringConstant);
  val.constant = NumToString(builder_.CreateString(s).o);
  return NoError();
}

CheckedError Parser::ParseComma() {
  if (!opts.protobuf_ascii_alike) EXPECT(',');
  return NoError();
}

CheckedError Parser::ParseAnyValue(Value &val, FieldDef *field,
                                   size_t parent_fieldn,
                                   const StructDef *parent_struct_def,
                                   uoffset_t count,
                                   bool inside_vector) {
  switch (val.type.base_type) {
    case BASE_TYPE_UNION: {
      FLATBUFFERS_ASSERT(field);
      std::string constant;
      Vector<uint8_t> *vector_of_union_types = nullptr;
      // Find corresponding type field we may have already parsed.
      for (auto elem = field_stack_.rbegin() + count;
           elem != field_stack_.rbegin() + parent_fieldn + count; ++elem) {
        auto &type = elem->second->value.type;
        if (type.enum_def == val.type.enum_def) {
          if (inside_vector) {
            if (type.base_type == BASE_TYPE_VECTOR &&
                type.element == BASE_TYPE_UTYPE) {
              // Vector of union type field.
              uoffset_t offset;
              ECHECK(atot(elem->first.constant.c_str(), *this, &offset));
              vector_of_union_types = reinterpret_cast<Vector<uint8_t> *>(
                                        builder_.GetCurrentBufferPointer() +
                                        builder_.GetSize() - offset);
              break;
            }
          } else {
            if (type.base_type == BASE_TYPE_UTYPE) {
              // Union type field.
              constant = elem->first.constant;
              break;
            }
          }
        }
      }
      if (constant.empty() && !inside_vector) {
        // We haven't seen the type field yet. Sadly a lot of JSON writers
        // output these in alphabetical order, meaning it comes after this
        // value. So we scan past the value to find it, then come back here.
        // We currently don't do this for vectors of unions because the
        // scanning/serialization logic would get very complicated.
        auto type_name = field->name + UnionTypeFieldSuffix();
        FLATBUFFERS_ASSERT(parent_struct_def);
        auto type_field = parent_struct_def->fields.Lookup(type_name);
        FLATBUFFERS_ASSERT(type_field);  // Guaranteed by ParseField().
        // Remember where we are in the source file, so we can come back here.
        auto backup = *static_cast<ParserState *>(this);
        ECHECK(SkipAnyJsonValue());  // The table.
        ECHECK(ParseComma());
        auto next_name = attribute_;
        if (Is(kTokenStringConstant)) {
          NEXT();
        } else {
          EXPECT(kTokenIdentifier);
        }
        if (next_name == type_name) {
          EXPECT(':');
          Value type_val = type_field->value;
          ECHECK(ParseAnyValue(type_val, type_field, 0, nullptr, 0));
          constant = type_val.constant;
          // Got the information we needed, now rewind:
          *static_cast<ParserState *>(this) = backup;
        }
      }
      if (constant.empty() && !vector_of_union_types) {
        return Error("missing type field for this union value: " +
                     field->name);
      }
      uint8_t enum_idx;
      if (vector_of_union_types) {
        enum_idx = vector_of_union_types->Get(count);
      } else {
        ECHECK(atot(constant.c_str(), *this, &enum_idx));
      }
      auto enum_val = val.type.enum_def->ReverseLookup(enum_idx, true);
      if (!enum_val) return Error("illegal type id for: " + field->name);
      if (enum_val->union_type.base_type == BASE_TYPE_STRUCT) {
        ECHECK(ParseTable(*enum_val->union_type.struct_def, &val.constant,
                          nullptr));
        if (enum_val->union_type.struct_def->fixed) {
          // All BASE_TYPE_UNION values are offsets, so turn this into one.
          SerializeStruct(*enum_val->union_type.struct_def, val);
          builder_.ClearOffsets();
          val.constant = NumToString(builder_.GetSize());
        }
      } else if (enum_val->union_type.base_type == BASE_TYPE_STRING) {
        ECHECK(ParseString(val));
      } else {
        FLATBUFFERS_ASSERT(false);
      }
      break;
    }
    case BASE_TYPE_STRUCT:
      ECHECK(ParseTable(*val.type.struct_def, &val.constant, nullptr));
      break;
    case BASE_TYPE_STRING: {
      ECHECK(ParseString(val));
      break;
    }
    case BASE_TYPE_VECTOR: {
      uoffset_t off;
      ECHECK(ParseVector(val.type.VectorType(), &off, field, parent_fieldn));
      val.constant = NumToString(off);
      break;
    }
    case BASE_TYPE_ARRAY: {
      ECHECK(ParseArray(val));
      break;
    }
    case BASE_TYPE_INT:
    case BASE_TYPE_UINT:
    case BASE_TYPE_LONG:
    case BASE_TYPE_ULONG: {
      if (field && field->attributes.Lookup("hash") &&
          (token_ == kTokenIdentifier || token_ == kTokenStringConstant)) {
        ECHECK(ParseHash(val, field));
      } else {
        ECHECK(ParseSingleValue(field ? &field->name : nullptr, val, false));
      }
      break;
    }
    default:
      ECHECK(ParseSingleValue(field ? &field->name : nullptr, val, false));
      break;
  }
  return NoError();
}

void Parser::SerializeStruct(const StructDef &struct_def, const Value &val) {
  SerializeStruct(builder_, struct_def, val);
}

void Parser::SerializeStruct(FlatBufferBuilder &builder,
                             const StructDef &struct_def, const Value &val) {
  FLATBUFFERS_ASSERT(val.constant.length() == struct_def.bytesize);
  builder.Align(struct_def.minalign);
  builder.PushBytes(reinterpret_cast<const uint8_t *>(val.constant.c_str()),
                    struct_def.bytesize);
  builder.AddStructOffset(val.offset, builder.GetSize());
}

template <typename F>
CheckedError Parser::ParseTableDelimiters(size_t &fieldn,
                                          const StructDef *struct_def,
                                          F body) {
  // We allow tables both as JSON object{ .. } with field names
  // or vector[..] with all fields in order
  char terminator = '}';
  bool is_nested_vector = struct_def && Is('[');
  if (is_nested_vector) {
    NEXT();
    terminator = ']';
  } else {
    EXPECT('{');
  }
  for (;;) {
    if ((!opts.strict_json || !fieldn) && Is(terminator)) break;
    std::string name;
    if (is_nested_vector) {
      if (fieldn >= struct_def->fields.vec.size()) {
        return Error("too many unnamed fields in nested array");
      }
      name = struct_def->fields.vec[fieldn]->name;
    } else {
      name = attribute_;
      if (Is(kTokenStringConstant)) {
        NEXT();
      } else {
        EXPECT(opts.strict_json ? kTokenStringConstant : kTokenIdentifier);
      }
      if (!opts.protobuf_ascii_alike || !(Is('{') || Is('['))) EXPECT(':');
    }
    ECHECK(body(name, fieldn, struct_def));
    if (Is(terminator)) break;
    ECHECK(ParseComma());
  }
  NEXT();
  if (is_nested_vector && fieldn != struct_def->fields.vec.size()) {
    return Error("wrong number of unnamed fields in table vector");
  }
  return NoError();
}

CheckedError Parser::ParseTable(const StructDef &struct_def, std::string *value,
                                uoffset_t *ovalue) {
  size_t fieldn_outer = 0;
  auto err = ParseTableDelimiters(
      fieldn_outer, &struct_def,
      [&](const std::string &name, size_t &fieldn,
          const StructDef *struct_def_inner) -> CheckedError {
        if (name == "$schema") {
          ECHECK(Expect(kTokenStringConstant));
          return NoError();
        }
        auto field = struct_def_inner->fields.Lookup(name);
        if (!field) {
          if (!opts.skip_unexpected_fields_in_json) {
            return Error("unknown field: " + name);
          } else {
            ECHECK(SkipAnyJsonValue());
          }
        } else {
          if (IsIdent("null") && !IsScalar(field->value.type.base_type)) {
            ECHECK(Next());  // Ignore this field.
          } else {
            Value val = field->value;
            if (field->flexbuffer) {
              flexbuffers::Builder builder(1024,
                                           flexbuffers::BUILDER_FLAG_SHARE_ALL);
              ECHECK(ParseFlexBufferValue(&builder));
              builder.Finish();
              // Force alignment for nested flexbuffer
              builder_.ForceVectorAlignment(builder.GetSize(), sizeof(uint8_t),
                                            sizeof(largest_scalar_t));
              auto off = builder_.CreateVector(builder.GetBuffer());
              val.constant = NumToString(off.o);
            } else if (field->nested_flatbuffer) {
              ECHECK(
                  ParseNestedFlatbuffer(val, field, fieldn, struct_def_inner));
            } else {
              ECHECK(Recurse([&]() {
                return ParseAnyValue(val, field, fieldn, struct_def_inner, 0);
              }));
            }
            // Hardcoded insertion-sort with error-check.
            // If fields are specified in order, then this loop exits
            // immediately.
            auto elem = field_stack_.rbegin();
            for (; elem != field_stack_.rbegin() + fieldn; ++elem) {
              auto existing_field = elem->second;
              if (existing_field == field)
                return Error("field set more than once: " + field->name);
              if (existing_field->value.offset < field->value.offset) break;
            }
            // Note: elem points to before the insertion point, thus .base()
            // points to the correct spot.
            field_stack_.insert(elem.base(), std::make_pair(val, field));
            fieldn++;
          }
        }
        return NoError();
      });
  ECHECK(err);

  // Check if all required fields are parsed.
  for (auto field_it = struct_def.fields.vec.begin();
       field_it != struct_def.fields.vec.end(); ++field_it) {
    auto required_field = *field_it;
    if (!required_field->required) { continue; }
    bool found = false;
    for (auto pf_it = field_stack_.end() - fieldn_outer;
         pf_it != field_stack_.end(); ++pf_it) {
      auto parsed_field = pf_it->second;
      if (parsed_field == required_field) {
        found = true;
        break;
      }
    }
    if (!found) {
      return Error("required field is missing: " + required_field->name +
                   " in " + struct_def.name);
    }
  }

  if (struct_def.fixed && fieldn_outer != struct_def.fields.vec.size())
    return Error("struct: wrong number of initializers: " + struct_def.name);

  auto start = struct_def.fixed ? builder_.StartStruct(struct_def.minalign)
                                : builder_.StartTable();

  for (size_t size = struct_def.sortbysize ? sizeof(largest_scalar_t) : 1; size;
       size /= 2) {
    // Go through elements in reverse, since we're building the data backwards.
    for (auto it = field_stack_.rbegin();
         it != field_stack_.rbegin() + fieldn_outer; ++it) {
      auto &field_value = it->first;
      auto field = it->second;
      if (!struct_def.sortbysize ||
          size == SizeOf(field_value.type.base_type)) {
        switch (field_value.type.base_type) {
          // clang-format off
          #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
            CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
            case BASE_TYPE_ ## ENUM: \
              builder_.Pad(field->padding); \
              if (struct_def.fixed) { \
                CTYPE val; \
                ECHECK(atot(field_value.constant.c_str(), *this, &val)); \
                builder_.PushElement(val); \
              } else { \
                CTYPE val, valdef; \
                ECHECK(atot(field_value.constant.c_str(), *this, &val)); \
                ECHECK(atot(field->value.constant.c_str(), *this, &valdef)); \
                builder_.AddElement(field_value.offset, val, valdef); \
              } \
              break;
            FLATBUFFERS_GEN_TYPES_SCALAR(FLATBUFFERS_TD);
          #undef FLATBUFFERS_TD
          #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
            CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
            case BASE_TYPE_ ## ENUM: \
              builder_.Pad(field->padding); \
              if (IsStruct(field->value.type)) { \
                SerializeStruct(*field->value.type.struct_def, field_value); \
              } else { \
                CTYPE val; \
                ECHECK(atot(field_value.constant.c_str(), *this, &val)); \
                builder_.AddOffset(field_value.offset, val); \
              } \
              break;
            FLATBUFFERS_GEN_TYPES_POINTER(FLATBUFFERS_TD);
          #undef FLATBUFFERS_TD
            case BASE_TYPE_ARRAY:
              builder_.Pad(field->padding);
              builder_.PushBytes(
                reinterpret_cast<const uint8_t*>(field_value.constant.c_str()),
                InlineSize(field_value.type));
              break;
              // clang-format on
        }
      }
    }
  }
  for (size_t i = 0; i < fieldn_outer; i++) field_stack_.pop_back();

  if (struct_def.fixed) {
    builder_.ClearOffsets();
    builder_.EndStruct();
    FLATBUFFERS_ASSERT(value);
    // Temporarily store this struct in the value string, since it is to
    // be serialized in-place elsewhere.
    value->assign(
        reinterpret_cast<const char *>(builder_.GetCurrentBufferPointer()),
        struct_def.bytesize);
    builder_.PopBytes(struct_def.bytesize);
    FLATBUFFERS_ASSERT(!ovalue);
  } else {
    auto val = builder_.EndTable(start);
    if (ovalue) *ovalue = val;
    if (value) *value = NumToString(val);
  }
  return NoError();
}

template <typename F>
CheckedError Parser::ParseVectorDelimiters(uoffset_t &count, F body) {
  EXPECT('[');
  for (;;) {
    if ((!opts.strict_json || !count) && Is(']')) break;
    ECHECK(body(count));
    count++;
    if (Is(']')) break;
    ECHECK(ParseComma());
  }
  NEXT();
  return NoError();
}

CheckedError Parser::ParseVector(const Type &type, uoffset_t *ovalue,
                                 FieldDef *field, size_t fieldn) {
  uoffset_t count = 0;
  auto err = ParseVectorDelimiters(count, [&](uoffset_t &) -> CheckedError {
    Value val;
    val.type = type;
    ECHECK(Recurse([&]() {
      return ParseAnyValue(val, field, fieldn, nullptr, count, true);
    }));
    field_stack_.push_back(std::make_pair(val, nullptr));
    return NoError();
  });
  ECHECK(err);

  builder_.StartVector(count * InlineSize(type) / InlineAlignment(type),
                       InlineAlignment(type));
  for (uoffset_t i = 0; i < count; i++) {
    // start at the back, since we're building the data backwards.
    auto &val = field_stack_.back().first;
    switch (val.type.base_type) {
      // clang-format off
      #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
        CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
        case BASE_TYPE_ ## ENUM: \
          if (IsStruct(val.type)) SerializeStruct(*val.type.struct_def, val); \
          else { \
             CTYPE elem; \
             ECHECK(atot(val.constant.c_str(), *this, &elem)); \
             builder_.PushElement(elem); \
          } \
          break;
        FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
      #undef FLATBUFFERS_TD
      // clang-format on
    }
    field_stack_.pop_back();
  }

  builder_.ClearOffsets();
  *ovalue = builder_.EndVector(count);
  return NoError();
}

CheckedError Parser::ParseArray(Value &array) {
  std::vector<Value> stack;
  FlatBufferBuilder builder;
  const auto &type = array.type.VectorType();
  auto length = array.type.fixed_length;
  uoffset_t count = 0;
  auto err = ParseVectorDelimiters(count, [&](uoffset_t &) -> CheckedError {
    vector_emplace_back(&stack, Value());
    auto &val = stack.back();
    val.type = type;
    if (IsStruct(type)) {
      ECHECK(ParseTable(*val.type.struct_def, &val.constant, nullptr));
    } else {
      ECHECK(ParseSingleValue(nullptr, val, false));
    }
    return NoError();
  });
  ECHECK(err);
  if (length != count) return Error("Fixed-length array size is incorrect.");

  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    auto &val = *it;
    // clang-format off
    switch (val.type.base_type) {
      #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
        CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
        case BASE_TYPE_ ## ENUM: \
          if (IsStruct(val.type)) { \
            SerializeStruct(builder, *val.type.struct_def, val); \
          } else { \
            CTYPE elem; \
            ECHECK(atot(val.constant.c_str(), *this, &elem)); \
            builder.PushElement(elem); \
          } \
        break;
        FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
      #undef FLATBUFFERS_TD
      default: FLATBUFFERS_ASSERT(0);
    }
    // clang-format on
  }

  array.constant.assign(
      reinterpret_cast<const char *>(builder.GetCurrentBufferPointer()),
      InlineSize(array.type));
  return NoError();
}

CheckedError Parser::ParseNestedFlatbuffer(Value &val, FieldDef *field,
                                           size_t fieldn,
                                           const StructDef *parent_struct_def) {
  if (token_ == '[') {  // backwards compat for 'legacy' ubyte buffers
    ECHECK(ParseAnyValue(val, field, fieldn, parent_struct_def, 0));
  } else {
    auto cursor_at_value_begin = cursor_;
    ECHECK(SkipAnyJsonValue());
    std::string substring(cursor_at_value_begin - 1, cursor_ - 1);

    // Create and initialize new parser
    Parser nested_parser;
    FLATBUFFERS_ASSERT(field->nested_flatbuffer);
    nested_parser.root_struct_def_ = field->nested_flatbuffer;
    nested_parser.enums_ = enums_;
    nested_parser.opts = opts;
    nested_parser.uses_flexbuffers_ = uses_flexbuffers_;

    // Parse JSON substring into new flatbuffer builder using nested_parser
    bool ok = nested_parser.Parse(substring.c_str(), nullptr, nullptr);

    // Clean nested_parser to avoid deleting the elements in
    // the SymbolTables on destruction
    nested_parser.enums_.dict.clear();
    nested_parser.enums_.vec.clear();

    if (!ok) {
      ECHECK(Error(nested_parser.error_));
    }
    // Force alignment for nested flatbuffer
    builder_.ForceVectorAlignment(nested_parser.builder_.GetSize(), sizeof(uint8_t),
                                  nested_parser.builder_.GetBufferMinAlignment());

    auto off = builder_.CreateVector(nested_parser.builder_.GetBufferPointer(),
                                     nested_parser.builder_.GetSize());
    val.constant = NumToString(off.o);
  }
  return NoError();
}

CheckedError Parser::ParseMetaData(SymbolTable<Value> *attributes) {
  if (Is('(')) {
    NEXT();
    for (;;) {
      auto name = attribute_;
      if (false == (Is(kTokenIdentifier) || Is(kTokenStringConstant)))
        return Error("attribute name must be either identifier or string: " +
          name);
      if (known_attributes_.find(name) == known_attributes_.end())
        return Error("user define attributes must be declared before use: " +
                     name);
      NEXT();
      auto e = new Value();
      attributes->Add(name, e);
      if (Is(':')) {
        NEXT();
        ECHECK(ParseSingleValue(&name, *e, true));
      }
      if (Is(')')) {
        NEXT();
        break;
      }
      EXPECT(',');
    }
  }
  return NoError();
}

CheckedError Parser::TryTypedValue(const std::string *name, int dtoken,
                                   bool check, Value &e, BaseType req,
                                   bool *destmatch) {
  bool match = dtoken == token_;
  if (match) {
    FLATBUFFERS_ASSERT(*destmatch == false);
    *destmatch = true;
    e.constant = attribute_;
    // Check token match
    if (!check) {
      if (e.type.base_type == BASE_TYPE_NONE) {
        e.type.base_type = req;
      } else {
        return Error(
            std::string("type mismatch: expecting: ") +
            kTypeNames[e.type.base_type] + ", found: " + kTypeNames[req] +
            ", name: " + (name ? *name : "") + ", value: " + e.constant);
      }
    }
    // The exponent suffix of hexadecimal float-point number is mandatory.
    // A hex-integer constant is forbidden as an initializer of float number.
    if ((kTokenFloatConstant != dtoken) && IsFloat(e.type.base_type)) {
      const auto &s = e.constant;
      const auto k = s.find_first_of("0123456789.");
      if ((std::string::npos != k) && (s.length() > (k + 1)) &&
          (s[k] == '0' && is_alpha_char(s[k + 1], 'X')) &&
          (std::string::npos == s.find_first_of("pP", k + 2))) {
        return Error(
            "invalid number, the exponent suffix of hexadecimal "
            "floating-point literals is mandatory: \"" +
            s + "\"");
      }
    }

    NEXT();
  }
  return NoError();
}

CheckedError Parser::ParseEnumFromString(const Type &type,
                                         std::string *result) {
  const auto base_type =
      type.enum_def ? type.enum_def->underlying_type.base_type : type.base_type;
  if (!IsInteger(base_type)) return Error("not a valid value for this field");
  uint64_t u64 = 0;
  for (size_t pos = 0; pos != std::string::npos;) {
    const auto delim = attribute_.find_first_of(' ', pos);
    const auto last = (std::string::npos == delim);
    auto word = attribute_.substr(pos, !last ? delim - pos : std::string::npos);
    pos = !last ? delim + 1 : std::string::npos;
    const EnumVal *ev = nullptr;
    if (type.enum_def) {
      ev = type.enum_def->Lookup(word);
    } else {
      auto dot = word.find_first_of('.');
      if (std::string::npos == dot)
        return Error("enum values need to be qualified by an enum type");
      auto enum_def_str = word.substr(0, dot);
      const auto enum_def = LookupEnum(enum_def_str);
      if (!enum_def) return Error("unknown enum: " + enum_def_str);
      auto enum_val_str = word.substr(dot + 1);
      ev = enum_def->Lookup(enum_val_str);
    }
    if (!ev) return Error("unknown enum value: " + word);
    u64 |= ev->GetAsUInt64();
  }
  *result = IsUnsigned(base_type) ? NumToString(u64)
                                  : NumToString(static_cast<int64_t>(u64));
  return NoError();
}

CheckedError Parser::ParseHash(Value &e, FieldDef *field) {
  FLATBUFFERS_ASSERT(field);
  Value *hash_name = field->attributes.Lookup("hash");
  switch (e.type.base_type) {
    case BASE_TYPE_SHORT: {
      auto hash = FindHashFunction16(hash_name->constant.c_str());
      int16_t hashed_value = static_cast<int16_t>(hash(attribute_.c_str()));
      e.constant = NumToString(hashed_value);
      break;
    }
    case BASE_TYPE_USHORT: {
      auto hash = FindHashFunction16(hash_name->constant.c_str());
      uint16_t hashed_value = hash(attribute_.c_str());
      e.constant = NumToString(hashed_value);
      break;
    }
    case BASE_TYPE_INT: {
      auto hash = FindHashFunction32(hash_name->constant.c_str());
      int32_t hashed_value = static_cast<int32_t>(hash(attribute_.c_str()));
      e.constant = NumToString(hashed_value);
      break;
    }
    case BASE_TYPE_UINT: {
      auto hash = FindHashFunction32(hash_name->constant.c_str());
      uint32_t hashed_value = hash(attribute_.c_str());
      e.constant = NumToString(hashed_value);
      break;
    }
    case BASE_TYPE_LONG: {
      auto hash = FindHashFunction64(hash_name->constant.c_str());
      int64_t hashed_value = static_cast<int64_t>(hash(attribute_.c_str()));
      e.constant = NumToString(hashed_value);
      break;
    }
    case BASE_TYPE_ULONG: {
      auto hash = FindHashFunction64(hash_name->constant.c_str());
      uint64_t hashed_value = hash(attribute_.c_str());
      e.constant = NumToString(hashed_value);
      break;
    }
    default: FLATBUFFERS_ASSERT(0);
  }
  NEXT();
  return NoError();
}

CheckedError Parser::TokenError() {
  return Error("cannot parse value starting with: " + TokenToStringId(token_));
}

// Re-pack helper (ParseSingleValue) to normalize defaults of scalars.
template<typename T> inline void SingleValueRepack(Value &e, T val) {
  // Remove leading zeros.
  if (IsInteger(e.type.base_type)) { e.constant = NumToString(val); }
}
#if defined(FLATBUFFERS_HAS_NEW_STRTOD) && (FLATBUFFERS_HAS_NEW_STRTOD > 0)
// Normilaze defaults NaN to unsigned quiet-NaN(0).
static inline void SingleValueRepack(Value& e, float val) {
  if (val != val) e.constant = "nan";
}
static inline void SingleValueRepack(Value& e, double val) {
  if (val != val) e.constant = "nan";
}
#endif

CheckedError Parser::ParseSingleValue(const std::string *name, Value &e,
                                      bool check_now) {
  // First see if this could be a conversion function:
  if (token_ == kTokenIdentifier && *cursor_ == '(') {
    // todo: Extract processing of conversion functions to ParseFunction.
    const auto functionname = attribute_;
    if (!IsFloat(e.type.base_type)) {
      return Error(functionname + ": type of argument mismatch, expecting: " +
                   kTypeNames[BASE_TYPE_DOUBLE] +
                   ", found: " + kTypeNames[e.type.base_type] +
                   ", name: " + (name ? *name : "") + ", value: " + e.constant);
    }
    NEXT();
    EXPECT('(');
    ECHECK(Recurse([&]() { return ParseSingleValue(name, e, false); }));
    EXPECT(')');
    // calculate with double precision
    double x, y = 0.0;
    ECHECK(atot(e.constant.c_str(), *this, &x));
    auto func_match = false;
    // clang-format off
    #define FLATBUFFERS_FN_DOUBLE(name, op) \
      if (!func_match && functionname == name) { y = op; func_match = true; }
    FLATBUFFERS_FN_DOUBLE("deg", x / kPi * 180);
    FLATBUFFERS_FN_DOUBLE("rad", x * kPi / 180);
    FLATBUFFERS_FN_DOUBLE("sin", sin(x));
    FLATBUFFERS_FN_DOUBLE("cos", cos(x));
    FLATBUFFERS_FN_DOUBLE("tan", tan(x));
    FLATBUFFERS_FN_DOUBLE("asin", asin(x));
    FLATBUFFERS_FN_DOUBLE("acos", acos(x));
    FLATBUFFERS_FN_DOUBLE("atan", atan(x));
    // TODO(wvo): add more useful conversion functions here.
    #undef FLATBUFFERS_FN_DOUBLE
    // clang-format on
    if (true != func_match) {
      return Error(std::string("Unknown conversion function: ") + functionname +
                   ", field name: " + (name ? *name : "") +
                   ", value: " + e.constant);
    }
    e.constant = NumToString(y);
    return NoError();
  }

  auto match = false;
  const auto in_type = e.type.base_type;
  // clang-format off
  #define IF_ECHECK_(force, dtoken, check, req)    \
    if (!match && ((check) || IsConstTrue(force))) \
    ECHECK(TryTypedValue(name, dtoken, check, e, req, &match))
  #define TRY_ECHECK(dtoken, check, req) IF_ECHECK_(false, dtoken, check, req)
  #define FORCE_ECHECK(dtoken, check, req) IF_ECHECK_(true, dtoken, check, req)
  // clang-format on

  if (token_ == kTokenStringConstant || token_ == kTokenIdentifier) {
    const auto kTokenStringOrIdent = token_;
    // The string type is a most probable type, check it first.
    TRY_ECHECK(kTokenStringConstant, in_type == BASE_TYPE_STRING,
               BASE_TYPE_STRING);

    // avoid escaped and non-ascii in the string
    if (!match && (token_ == kTokenStringConstant) && IsScalar(in_type) &&
        !attr_is_trivial_ascii_string_) {
      return Error(
          std::string("type mismatch or invalid value, an initializer of "
                      "non-string field must be trivial ASCII string: type: ") +
          kTypeNames[in_type] + ", name: " + (name ? *name : "") +
          ", value: " + attribute_);
    }

    // A boolean as true/false. Boolean as Integer check below.
    if (!match && IsBool(in_type)) {
      auto is_true = attribute_ == "true";
      if (is_true || attribute_ == "false") {
        attribute_ = is_true ? "1" : "0";
        // accepts both kTokenStringConstant and kTokenIdentifier
        TRY_ECHECK(kTokenStringOrIdent, IsBool(in_type), BASE_TYPE_BOOL);
      }
    }
    // Check if this could be a string/identifier enum value.
    // Enum can have only true integer base type.
    if (!match && IsInteger(in_type) && !IsBool(in_type) &&
        IsIdentifierStart(*attribute_.c_str())) {
      ECHECK(ParseEnumFromString(e.type, &e.constant));
      NEXT();
      match = true;
    }
    // Parse a float/integer number from the string.
    if (!match) check_now = true;  // Re-pack if parsed from string literal.
    if (!match && (token_ == kTokenStringConstant) && IsScalar(in_type)) {
      // remove trailing whitespaces from attribute_
      auto last = attribute_.find_last_not_of(' ');
      if (std::string::npos != last)  // has non-whitespace
        attribute_.resize(last + 1);
    }
    // Float numbers or nan, inf, pi, etc.
    TRY_ECHECK(kTokenStringOrIdent, IsFloat(in_type), BASE_TYPE_FLOAT);
    // An integer constant in string.
    TRY_ECHECK(kTokenStringOrIdent, IsInteger(in_type), BASE_TYPE_INT);
    // Unknown tokens will be interpreted as string type.
    // An attribute value may be a scalar or string constant.
    FORCE_ECHECK(kTokenStringConstant, in_type == BASE_TYPE_STRING,
                 BASE_TYPE_STRING);
  } else {
    // Try a float number.
    TRY_ECHECK(kTokenFloatConstant, IsFloat(in_type), BASE_TYPE_FLOAT);
    // Integer token can init any scalar (integer of float).
    FORCE_ECHECK(kTokenIntegerConstant, IsScalar(in_type), BASE_TYPE_INT);
  }
#undef FORCE_ECHECK
#undef TRY_ECHECK
#undef IF_ECHECK_

  if (!match) {
    std::string msg;
    msg += "Cannot assign token starting with '" + TokenToStringId(token_) +
           "' to value of <" + std::string(kTypeNames[in_type]) + "> type.";
    return Error(msg);
  }
  const auto match_type = e.type.base_type; // may differ from in_type
  // The check_now flag must be true when parse a fbs-schema.
  // This flag forces to check default scalar values or metadata of field.
  // For JSON parser the flag should be false.
  // If it is set for JSON each value will be checked twice (see ParseTable).
  if (check_now && IsScalar(match_type)) {
    // clang-format off
    switch (match_type) {
    #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
            CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
            case BASE_TYPE_ ## ENUM: {\
                CTYPE val; \
                ECHECK(atot(e.constant.c_str(), *this, &val)); \
                SingleValueRepack(e, val); \
              break; }
    FLATBUFFERS_GEN_TYPES_SCALAR(FLATBUFFERS_TD);
    #undef FLATBUFFERS_TD
    default: break;
    }
    // clang-format on
  }
  return NoError();
}

StructDef *Parser::LookupCreateStruct(const std::string &name,
                                      bool create_if_new, bool definition) {
  std::string qualified_name = current_namespace_->GetFullyQualifiedName(name);
  // See if it exists pre-declared by an unqualified use.
  auto struct_def = LookupStruct(name);
  if (struct_def && struct_def->predecl) {
    if (definition) {
      // Make sure it has the current namespace, and is registered under its
      // qualified name.
      struct_def->defined_namespace = current_namespace_;
      structs_.Move(name, qualified_name);
    }
    return struct_def;
  }
  // See if it exists pre-declared by an qualified use.
  struct_def = LookupStruct(qualified_name);
  if (struct_def && struct_def->predecl) {
    if (definition) {
      // Make sure it has the current namespace.
      struct_def->defined_namespace = current_namespace_;
    }
    return struct_def;
  }
  if (!definition) {
    // Search thru parent namespaces.
    for (size_t components = current_namespace_->components.size();
         components && !struct_def; components--) {
      struct_def = LookupStruct(
          current_namespace_->GetFullyQualifiedName(name, components - 1));
    }
  }
  if (!struct_def && create_if_new) {
    struct_def = new StructDef();
    if (definition) {
      structs_.Add(qualified_name, struct_def);
      struct_def->name = name;
      struct_def->defined_namespace = current_namespace_;
    } else {
      // Not a definition.
      // Rather than failing, we create a "pre declared" StructDef, due to
      // circular references, and check for errors at the end of parsing.
      // It is defined in the current namespace, as the best guess what the
      // final namespace will be.
      structs_.Add(name, struct_def);
      struct_def->name = name;
      struct_def->defined_namespace = current_namespace_;
      struct_def->original_location.reset(
          new std::string(file_being_parsed_ + ":" + NumToString(line_)));
    }
  }
  return struct_def;
}

const EnumVal *EnumDef::MinValue() const {
  return vals.vec.empty() ? nullptr : vals.vec.front();
}
const EnumVal *EnumDef::MaxValue() const {
  return vals.vec.empty() ? nullptr : vals.vec.back();
}

template<typename T> static uint64_t EnumDistanceImpl(T e1, T e2) {
  if (e1 < e2) { std::swap(e1, e2); }  // use std for scalars
  // Signed overflow may occur, use unsigned calculation.
  // The unsigned overflow is well-defined by C++ standard (modulo 2^n).
  return static_cast<uint64_t>(e1) - static_cast<uint64_t>(e2);
}

uint64_t EnumDef::Distance(const EnumVal *v1, const EnumVal *v2) const {
  return IsUInt64() ? EnumDistanceImpl(v1->GetAsUInt64(), v2->GetAsUInt64())
                    : EnumDistanceImpl(v1->GetAsInt64(), v2->GetAsInt64());
}

std::string EnumDef::AllFlags() const {
  FLATBUFFERS_ASSERT(attributes.Lookup("bit_flags"));
  uint64_t u64 = 0;
  for (auto it = Vals().begin(); it != Vals().end(); ++it) {
    u64 |= (*it)->GetAsUInt64();
  }
  return IsUInt64() ? NumToString(u64) : NumToString(static_cast<int64_t>(u64));
}

EnumVal *EnumDef::ReverseLookup(int64_t enum_idx,
                                bool skip_union_default) const {
  auto skip_first = static_cast<int>(is_union && skip_union_default);
  for (auto it = Vals().begin() + skip_first; it != Vals().end(); ++it) {
    if ((*it)->GetAsInt64() == enum_idx) { return *it; }
  }
  return nullptr;
}

EnumVal *EnumDef::FindByValue(const std::string &constant) const {
  int64_t i64;
  auto done = false;
  if (IsUInt64()) {
    uint64_t u64;  // avoid reinterpret_cast of pointers
    done = StringToNumber(constant.c_str(), &u64);
    i64 = static_cast<int64_t>(u64);
  } else {
    done = StringToNumber(constant.c_str(), &i64);
  }
  FLATBUFFERS_ASSERT(done);
  if (!done) return nullptr;
  return ReverseLookup(i64, false);
}

void EnumDef::SortByValue() {
  auto &v = vals.vec;
  if (IsUInt64())
    std::sort(v.begin(), v.end(), [](const EnumVal *e1, const EnumVal *e2) {
      return e1->GetAsUInt64() < e2->GetAsUInt64();
    });
  else
    std::sort(v.begin(), v.end(), [](const EnumVal *e1, const EnumVal *e2) {
      return e1->GetAsInt64() < e2->GetAsInt64();
    });
}

void EnumDef::RemoveDuplicates() {
  // This method depends form SymbolTable implementation!
  // 1) vals.vec - owner (raw pointer)
  // 2) vals.dict - access map
  auto first = vals.vec.begin();
  auto last = vals.vec.end();
  if (first == last) return;
  auto result = first;
  while (++first != last) {
    if ((*result)->value != (*first)->value) {
      *(++result) = *first;
    } else {
      auto ev = *first;
      for (auto it = vals.dict.begin(); it != vals.dict.end(); ++it) {
        if (it->second == ev) it->second = *result;  // reassign
      }
      delete ev;  // delete enum value
      *first = nullptr;
    }
  }
  vals.vec.erase(++result, last);
}

template<typename T> void EnumDef::ChangeEnumValue(EnumVal *ev, T new_value) {
  ev->value = static_cast<int64_t>(new_value);
}

namespace EnumHelper {
template<BaseType E> struct EnumValType { typedef int64_t type; };
template<> struct EnumValType<BASE_TYPE_ULONG> { typedef uint64_t type; };
}  // namespace EnumHelper

struct EnumValBuilder {
  EnumVal *CreateEnumerator(const std::string &ev_name) {
    FLATBUFFERS_ASSERT(!temp);
    auto first = enum_def.vals.vec.empty();
    user_value = first;
    temp = new EnumVal(ev_name, first ? 0 : enum_def.vals.vec.back()->value);
    return temp;
  }

  EnumVal *CreateEnumerator(const std::string &ev_name, int64_t val) {
    FLATBUFFERS_ASSERT(!temp);
    user_value = true;
    temp = new EnumVal(ev_name, val);
    return temp;
  }

  FLATBUFFERS_CHECKED_ERROR AcceptEnumerator(const std::string &name) {
    FLATBUFFERS_ASSERT(temp);
    ECHECK(ValidateValue(&temp->value, false == user_value));
    FLATBUFFERS_ASSERT((temp->union_type.enum_def == nullptr) ||
                       (temp->union_type.enum_def == &enum_def));
    auto not_unique = enum_def.vals.Add(name, temp);
    temp = nullptr;
    if (not_unique) return parser.Error("enum value already exists: " + name);
    return NoError();
  }

  FLATBUFFERS_CHECKED_ERROR AcceptEnumerator() {
    return AcceptEnumerator(temp->name);
  }

  FLATBUFFERS_CHECKED_ERROR AssignEnumeratorValue(const std::string &value) {
    user_value = true;
    auto fit = false;
    auto ascending = false;
    if (enum_def.IsUInt64()) {
      uint64_t u64;
      fit = StringToNumber(value.c_str(), &u64);
      ascending = u64 > temp->GetAsUInt64();
      temp->value = static_cast<int64_t>(u64);  // well-defined since C++20.
    } else {
      int64_t i64;
      fit = StringToNumber(value.c_str(), &i64);
      ascending = i64 > temp->GetAsInt64();
      temp->value = i64;
    }
    if (!fit) return parser.Error("enum value does not fit, \"" + value + "\"");
    if (!ascending && strict_ascending && !enum_def.vals.vec.empty())
      return parser.Error("enum values must be specified in ascending order");
    return NoError();
  }

  template<BaseType E, typename CTYPE>
  inline FLATBUFFERS_CHECKED_ERROR ValidateImpl(int64_t *ev, int m) {
    typedef typename EnumHelper::EnumValType<E>::type T;  // int64_t or uint64_t
    static_assert(sizeof(T) == sizeof(int64_t), "invalid EnumValType");
    const auto v = static_cast<T>(*ev);
    auto up = static_cast<T>((flatbuffers::numeric_limits<CTYPE>::max)());
    auto dn = static_cast<T>((flatbuffers::numeric_limits<CTYPE>::lowest)());
    if (v < dn || v > (up - m)) {
      return parser.Error("enum value does not fit, \"" + NumToString(v) +
                          (m ? " + 1\"" : "\"") + " out of " +
                          TypeToIntervalString<CTYPE>());
    }
    *ev = static_cast<int64_t>(v + m);  // well-defined since C++20.
    return NoError();
  }

  FLATBUFFERS_CHECKED_ERROR ValidateValue(int64_t *ev, bool next) {
    // clang-format off
    switch (enum_def.underlying_type.base_type) {
    #define FLATBUFFERS_TD(ENUM, IDLTYPE, CTYPE, JTYPE, GTYPE, NTYPE,   \
                           PTYPE, RTYPE, KTYPE)                         \
      case BASE_TYPE_##ENUM: {                                          \
        if (!IsInteger(BASE_TYPE_##ENUM)) break;                        \
        return ValidateImpl<BASE_TYPE_##ENUM, CTYPE>(ev, next ? 1 : 0); \
      }
      FLATBUFFERS_GEN_TYPES_SCALAR(FLATBUFFERS_TD);
    #undef FLATBUFFERS_TD
    default: break;
    }
    // clang-format on
    return parser.Error("fatal: invalid enum underlying type");
  }

  EnumValBuilder(Parser &_parser, EnumDef &_enum_def, bool strict_order = true)
      : parser(_parser),
        enum_def(_enum_def),
        temp(nullptr),
        strict_ascending(strict_order),
        user_value(false) {}

  ~EnumValBuilder() { delete temp; }

  Parser &parser;
  EnumDef &enum_def;
  EnumVal *temp;
  const bool strict_ascending;
  bool user_value;
};

CheckedError Parser::ParseEnum(const bool is_union, EnumDef **dest) {
  std::vector<std::string> enum_comment = doc_comment_;
  NEXT();
  std::string enum_name = attribute_;
  EXPECT(kTokenIdentifier);
  EnumDef *enum_def;
  ECHECK(StartEnum(enum_name, is_union, &enum_def));
  enum_def->doc_comment = enum_comment;
  if (!is_union && !opts.proto_mode) {
    // Give specialized error message, since this type spec used to
    // be optional in the first FlatBuffers release.
    if (!Is(':')) {
      return Error(
          "must specify the underlying integer type for this"
          " enum (e.g. \': short\', which was the default).");
    } else {
      NEXT();
    }
    // Specify the integer type underlying this enum.
    ECHECK(ParseType(enum_def->underlying_type));
    if (!IsInteger(enum_def->underlying_type.base_type) ||
        IsBool(enum_def->underlying_type.base_type))
      return Error("underlying enum type must be integral");
    // Make this type refer back to the enum it was derived from.
    enum_def->underlying_type.enum_def = enum_def;
  }
  ECHECK(ParseMetaData(&enum_def->attributes));
  const auto underlying_type = enum_def->underlying_type.base_type;
  if (enum_def->attributes.Lookup("bit_flags") &&
      !IsUnsigned(underlying_type)) {
    // todo: Convert to the Error in the future?
    Warning("underlying type of bit_flags enum must be unsigned");
  }
  // Protobuf allows them to be specified in any order, so sort afterwards.
  const auto strict_ascending = (false == opts.proto_mode);
  EnumValBuilder evb(*this, *enum_def, strict_ascending);
  EXPECT('{');
  // A lot of code generatos expect that an enum is not-empty.
  if ((is_union || Is('}')) && !opts.proto_mode) {
    evb.CreateEnumerator("NONE");
    ECHECK(evb.AcceptEnumerator());
  }
  std::set<std::pair<BaseType, StructDef *>> union_types;
  while (!Is('}')) {
    if (opts.proto_mode && attribute_ == "option") {
      ECHECK(ParseProtoOption());
    } else {
      auto &ev = *evb.CreateEnumerator(attribute_);
      auto full_name = ev.name;
      ev.doc_comment = doc_comment_;
      EXPECT(kTokenIdentifier);
      if (is_union) {
        ECHECK(ParseNamespacing(&full_name, &ev.name));
        if (opts.union_value_namespacing) {
          // Since we can't namespace the actual enum identifiers, turn
          // namespace parts into part of the identifier.
          ev.name = full_name;
          std::replace(ev.name.begin(), ev.name.end(), '.', '_');
        }
        if (Is(':')) {
          NEXT();
          ECHECK(ParseType(ev.union_type));
          if (ev.union_type.base_type != BASE_TYPE_STRUCT &&
              ev.union_type.base_type != BASE_TYPE_STRING)
            return Error("union value type may only be table/struct/string");
        } else {
          ev.union_type = Type(BASE_TYPE_STRUCT, LookupCreateStruct(full_name));
        }
        if (!enum_def->uses_multiple_type_instances) {
          auto ins = union_types.insert(std::make_pair(
              ev.union_type.base_type, ev.union_type.struct_def));
          enum_def->uses_multiple_type_instances = (false == ins.second);
        }
      }

      if (Is('=')) {
        NEXT();
        ECHECK(evb.AssignEnumeratorValue(attribute_));
        EXPECT(kTokenIntegerConstant);
      } else if (false == strict_ascending) {
        // The opts.proto_mode flag is active.
        return Error("Protobuf mode doesn't allow implicit enum values.");
      }

      ECHECK(evb.AcceptEnumerator());

      if (opts.proto_mode && Is('[')) {
        NEXT();
        // ignore attributes on enums.
        while (token_ != ']') NEXT();
        NEXT();
      }
    }
    if (!Is(opts.proto_mode ? ';' : ',')) break;
    NEXT();
  }
  EXPECT('}');

  // At this point, the enum can be empty if input is invalid proto-file.
  if (!enum_def->size())
    return Error("incomplete enum declaration, values not found");

  if (enum_def->attributes.Lookup("bit_flags")) {
    const auto base_width = static_cast<uint64_t>(8 * SizeOf(underlying_type));
    for (auto it = enum_def->Vals().begin(); it != enum_def->Vals().end();
         ++it) {
      auto ev = *it;
      const auto u = ev->GetAsUInt64();
      // Stop manipulations with the sign.
      if (!IsUnsigned(underlying_type) && u == (base_width - 1))
        return Error("underlying type of bit_flags enum must be unsigned");
      if (u >= base_width)
        return Error("bit flag out of range of underlying integral type");
      enum_def->ChangeEnumValue(ev, 1ULL << u);
    }
  }

  if (false == strict_ascending)
    enum_def->SortByValue();  // Must be sorted to use MinValue/MaxValue.

  if (dest) *dest = enum_def;
  types_.Add(current_namespace_->GetFullyQualifiedName(enum_def->name),
             new Type(BASE_TYPE_UNION, nullptr, enum_def));
  return NoError();
}

CheckedError Parser::StartStruct(const std::string &name, StructDef **dest) {
  auto &struct_def = *LookupCreateStruct(name, true, true);
  if (!struct_def.predecl) return Error("datatype already exists: " + name);
  struct_def.predecl = false;
  struct_def.name = name;
  struct_def.file = file_being_parsed_;
  // Move this struct to the back of the vector just in case it was predeclared,
  // to preserve declaration order.
  *std::remove(structs_.vec.begin(), structs_.vec.end(), &struct_def) =
      &struct_def;
  *dest = &struct_def;
  return NoError();
}

CheckedError Parser::CheckClash(std::vector<FieldDef *> &fields,
                                StructDef *struct_def, const char *suffix,
                                BaseType basetype) {
  auto len = strlen(suffix);
  for (auto it = fields.begin(); it != fields.end(); ++it) {
    auto &fname = (*it)->name;
    if (fname.length() > len &&
        fname.compare(fname.length() - len, len, suffix) == 0 &&
        (*it)->value.type.base_type != BASE_TYPE_UTYPE) {
      auto field =
          struct_def->fields.Lookup(fname.substr(0, fname.length() - len));
      if (field && field->value.type.base_type == basetype)
        return Error("Field " + fname +
                     " would clash with generated functions for field " +
                     field->name);
    }
  }
  return NoError();
}

bool Parser::SupportsAdvancedUnionFeatures() const {
  return opts.lang_to_generate != 0 &&
         (opts.lang_to_generate & ~(IDLOptions::kCpp | IDLOptions::kJs |
                                    IDLOptions::kTs | IDLOptions::kPhp |
                                    IDLOptions::kJava | IDLOptions::kCSharp |
                                    IDLOptions::kKotlin |
                                    IDLOptions::kBinary)) == 0;
}

bool Parser::SupportsAdvancedArrayFeatures() const {
  return (opts.lang_to_generate &
          ~(IDLOptions::kCpp | IDLOptions::kPython | IDLOptions::kJava |
            IDLOptions::kCSharp | IDLOptions::kJsonSchema | IDLOptions::kJson |
            IDLOptions::kBinary)) == 0;
}

Namespace *Parser::UniqueNamespace(Namespace *ns) {
  for (auto it = namespaces_.begin(); it != namespaces_.end(); ++it) {
    if (ns->components == (*it)->components) {
      delete ns;
      return *it;
    }
  }
  namespaces_.push_back(ns);
  return ns;
}

std::string Parser::UnqualifiedName(const std::string &full_qualified_name) {
  Namespace *ns = new Namespace();

  std::size_t current, previous = 0;
  current = full_qualified_name.find('.');
  while (current != std::string::npos) {
    ns->components.push_back(
        full_qualified_name.substr(previous, current - previous));
    previous = current + 1;
    current = full_qualified_name.find('.', previous);
  }
  current_namespace_ = UniqueNamespace(ns);
  return full_qualified_name.substr(previous, current - previous);
}

static bool compareFieldDefs(const FieldDef *a, const FieldDef *b) {
  auto a_id = atoi(a->attributes.Lookup("id")->constant.c_str());
  auto b_id = atoi(b->attributes.Lookup("id")->constant.c_str());
  return a_id < b_id;
}

CheckedError Parser::ParseDecl() {
  std::vector<std::string> dc = doc_comment_;
  bool fixed = IsIdent("struct");
  if (!fixed && !IsIdent("table")) return Error("declaration expected");
  NEXT();
  std::string name = attribute_;
  EXPECT(kTokenIdentifier);
  StructDef *struct_def;
  ECHECK(StartStruct(name, &struct_def));
  struct_def->doc_comment = dc;
  struct_def->fixed = fixed;
  ECHECK(ParseMetaData(&struct_def->attributes));
  struct_def->sortbysize =
      struct_def->attributes.Lookup("original_order") == nullptr && !fixed;
  EXPECT('{');
  while (token_ != '}') ECHECK(ParseField(*struct_def));
  auto force_align = struct_def->attributes.Lookup("force_align");
  if (fixed) {
    if (force_align) {
      auto align = static_cast<size_t>(atoi(force_align->constant.c_str()));
      if (force_align->type.base_type != BASE_TYPE_INT ||
          align < struct_def->minalign || align > FLATBUFFERS_MAX_ALIGNMENT ||
          align & (align - 1))
        return Error(
            "force_align must be a power of two integer ranging from the"
            "struct\'s natural alignment to " +
            NumToString(FLATBUFFERS_MAX_ALIGNMENT));
      struct_def->minalign = align;
    }
    if (!struct_def->bytesize) return Error("size 0 structs not allowed");
  }
  struct_def->PadLastField(struct_def->minalign);
  // Check if this is a table that has manual id assignments
  auto &fields = struct_def->fields.vec;
  if (!fixed && fields.size()) {
    size_t num_id_fields = 0;
    for (auto it = fields.begin(); it != fields.end(); ++it) {
      if ((*it)->attributes.Lookup("id")) num_id_fields++;
    }
    // If any fields have ids..
    if (num_id_fields) {
      // Then all fields must have them.
      if (num_id_fields != fields.size())
        return Error(
            "either all fields or no fields must have an 'id' attribute");
      // Simply sort by id, then the fields are the same as if no ids had
      // been specified.
      std::sort(fields.begin(), fields.end(), compareFieldDefs);
      // Verify we have a contiguous set, and reassign vtable offsets.
      for (int i = 0; i < static_cast<int>(fields.size()); i++) {
        if (i != atoi(fields[i]->attributes.Lookup("id")->constant.c_str()))
          return Error("field id\'s must be consecutive from 0, id " +
                       NumToString(i) + " missing or set twice");
        fields[i]->value.offset = FieldIndexToOffset(static_cast<voffset_t>(i));
      }
    }
  }

  ECHECK(
      CheckClash(fields, struct_def, UnionTypeFieldSuffix(), BASE_TYPE_UNION));
  ECHECK(CheckClash(fields, struct_def, "Type", BASE_TYPE_UNION));
  ECHECK(CheckClash(fields, struct_def, "_length", BASE_TYPE_VECTOR));
  ECHECK(CheckClash(fields, struct_def, "Length", BASE_TYPE_VECTOR));
  ECHECK(CheckClash(fields, struct_def, "_byte_vector", BASE_TYPE_STRING));
  ECHECK(CheckClash(fields, struct_def, "ByteVector", BASE_TYPE_STRING));
  EXPECT('}');
  types_.Add(current_namespace_->GetFullyQualifiedName(struct_def->name),
             new Type(BASE_TYPE_STRUCT, struct_def, nullptr));
  return NoError();
}

CheckedError Parser::ParseService() {
  std::vector<std::string> service_comment = doc_comment_;
  NEXT();
  auto service_name = attribute_;
  EXPECT(kTokenIdentifier);
  auto &service_def = *new ServiceDef();
  service_def.name = service_name;
  service_def.file = file_being_parsed_;
  service_def.doc_comment = service_comment;
  service_def.defined_namespace = current_namespace_;
  if (services_.Add(current_namespace_->GetFullyQualifiedName(service_name),
                    &service_def))
    return Error("service already exists: " + service_name);
  ECHECK(ParseMetaData(&service_def.attributes));
  EXPECT('{');
  do {
    std::vector<std::string> doc_comment = doc_comment_;
    auto rpc_name = attribute_;
    EXPECT(kTokenIdentifier);
    EXPECT('(');
    Type reqtype, resptype;
    ECHECK(ParseTypeIdent(reqtype));
    EXPECT(')');
    EXPECT(':');
    ECHECK(ParseTypeIdent(resptype));
    if (reqtype.base_type != BASE_TYPE_STRUCT || reqtype.struct_def->fixed ||
        resptype.base_type != BASE_TYPE_STRUCT || resptype.struct_def->fixed)
      return Error("rpc request and response types must be tables");
    auto &rpc = *new RPCCall();
    rpc.name = rpc_name;
    rpc.request = reqtype.struct_def;
    rpc.response = resptype.struct_def;
    rpc.doc_comment = doc_comment;
    if (service_def.calls.Add(rpc_name, &rpc))
      return Error("rpc already exists: " + rpc_name);
    ECHECK(ParseMetaData(&rpc.attributes));
    EXPECT(';');
  } while (token_ != '}');
  NEXT();
  return NoError();
}

bool Parser::SetRootType(const char *name) {
  root_struct_def_ = LookupStruct(name);
  if (!root_struct_def_)
    root_struct_def_ =
        LookupStruct(current_namespace_->GetFullyQualifiedName(name));
  return root_struct_def_ != nullptr;
}

void Parser::MarkGenerated() {
  // This function marks all existing definitions as having already
  // been generated, which signals no code for included files should be
  // generated.
  for (auto it = enums_.vec.begin(); it != enums_.vec.end(); ++it) {
    (*it)->generated = true;
  }
  for (auto it = structs_.vec.begin(); it != structs_.vec.end(); ++it) {
    if (!(*it)->predecl) { (*it)->generated = true; }
  }
  for (auto it = services_.vec.begin(); it != services_.vec.end(); ++it) {
    (*it)->generated = true;
  }
}

CheckedError Parser::ParseNamespace() {
  NEXT();
  auto ns = new Namespace();
  namespaces_.push_back(ns);  // Store it here to not leak upon error.
  if (token_ != ';') {
    for (;;) {
      ns->components.push_back(attribute_);
      EXPECT(kTokenIdentifier);
      if (Is('.')) NEXT() else break;
    }
  }
  namespaces_.pop_back();
  current_namespace_ = UniqueNamespace(ns);
  EXPECT(';');
  return NoError();
}

// Best effort parsing of .proto declarations, with the aim to turn them
// in the closest corresponding FlatBuffer equivalent.
// We parse everything as identifiers instead of keywords, since we don't
// want protobuf keywords to become invalid identifiers in FlatBuffers.
CheckedError Parser::ParseProtoDecl() {
  bool isextend = IsIdent("extend");
  if (IsIdent("package")) {
    // These are identical in syntax to FlatBuffer's namespace decl.
    ECHECK(ParseNamespace());
  } else if (IsIdent("message") || isextend) {
    std::vector<std::string> struct_comment = doc_comment_;
    NEXT();
    StructDef *struct_def = nullptr;
    Namespace *parent_namespace = nullptr;
    if (isextend) {
      if (Is('.')) NEXT();  // qualified names may start with a . ?
      auto id = attribute_;
      EXPECT(kTokenIdentifier);
      ECHECK(ParseNamespacing(&id, nullptr));
      struct_def = LookupCreateStruct(id, false);
      if (!struct_def)
        return Error("cannot extend unknown message type: " + id);
    } else {
      std::string name = attribute_;
      EXPECT(kTokenIdentifier);
      ECHECK(StartStruct(name, &struct_def));
      // Since message definitions can be nested, we create a new namespace.
      auto ns = new Namespace();
      // Copy of current namespace.
      *ns = *current_namespace_;
      // But with current message name.
      ns->components.push_back(name);
      ns->from_table++;
      parent_namespace = current_namespace_;
      current_namespace_ = UniqueNamespace(ns);
    }
    struct_def->doc_comment = struct_comment;
    ECHECK(ParseProtoFields(struct_def, isextend, false));
    if (!isextend) { current_namespace_ = parent_namespace; }
    if (Is(';')) NEXT();
  } else if (IsIdent("enum")) {
    // These are almost the same, just with different terminator:
    EnumDef *enum_def;
    ECHECK(ParseEnum(false, &enum_def));
    if (Is(';')) NEXT();
    // Temp: remove any duplicates, as .fbs files can't handle them.
    enum_def->RemoveDuplicates();
  } else if (IsIdent("syntax")) {  // Skip these.
    NEXT();
    EXPECT('=');
    EXPECT(kTokenStringConstant);
    EXPECT(';');
  } else if (IsIdent("option")) {  // Skip these.
    ECHECK(ParseProtoOption());
    EXPECT(';');
  } else if (IsIdent("service")) {  // Skip these.
    NEXT();
    EXPECT(kTokenIdentifier);
    ECHECK(ParseProtoCurliesOrIdent());
  } else {
    return Error("don\'t know how to parse .proto declaration starting with " +
                 TokenToStringId(token_));
  }
  return NoError();
}

CheckedError Parser::StartEnum(const std::string &enum_name, bool is_union,
                               EnumDef **dest) {
  auto &enum_def = *new EnumDef();
  enum_def.name = enum_name;
  enum_def.file = file_being_parsed_;
  enum_def.doc_comment = doc_comment_;
  enum_def.is_union = is_union;
  enum_def.defined_namespace = current_namespace_;
  if (enums_.Add(current_namespace_->GetFullyQualifiedName(enum_name),
                 &enum_def))
    return Error("enum already exists: " + enum_name);
  enum_def.underlying_type.base_type = is_union ? BASE_TYPE_UTYPE
                                                : BASE_TYPE_INT;
  enum_def.underlying_type.enum_def = &enum_def;
  if (dest) *dest = &enum_def;
  return NoError();
}

CheckedError Parser::ParseProtoFields(StructDef *struct_def, bool isextend,
                                      bool inside_oneof) {
  EXPECT('{');
  while (token_ != '}') {
    if (IsIdent("message") || IsIdent("extend") || IsIdent("enum")) {
      // Nested declarations.
      ECHECK(ParseProtoDecl());
    } else if (IsIdent("extensions")) {  // Skip these.
      NEXT();
      EXPECT(kTokenIntegerConstant);
      if (Is(kTokenIdentifier)) {
        NEXT();  // to
        NEXT();  // num
      }
      EXPECT(';');
    } else if (IsIdent("option")) {  // Skip these.
      ECHECK(ParseProtoOption());
      EXPECT(';');
    } else if (IsIdent("reserved")) {  // Skip these.
      NEXT();
      while (!Is(';')) { NEXT(); }  // A variety of formats, just skip.
      NEXT();
    } else {
      std::vector<std::string> field_comment = doc_comment_;
      // Parse the qualifier.
      bool required = false;
      bool repeated = false;
      bool oneof = false;
      if (!inside_oneof) {
        if (IsIdent("optional")) {
          // This is the default.
          NEXT();
        } else if (IsIdent("required")) {
          required = true;
          NEXT();
        } else if (IsIdent("repeated")) {
          repeated = true;
          NEXT();
        } else if (IsIdent("oneof")) {
          oneof = true;
          NEXT();
        } else {
          // can't error, proto3 allows decls without any of the above.
        }
      }
      StructDef *anonymous_struct = nullptr;
      EnumDef *oneof_union = nullptr;
      Type type;
      if (IsIdent("group") || oneof) {
        if (!oneof) NEXT();
        if (oneof && opts.proto_oneof_union) {
          auto name = MakeCamel(attribute_, true) + "Union";
          ECHECK(StartEnum(name, true, &oneof_union));
          type = Type(BASE_TYPE_UNION, nullptr, oneof_union);
        } else {
          auto name = "Anonymous" + NumToString(anonymous_counter++);
          ECHECK(StartStruct(name, &anonymous_struct));
          type = Type(BASE_TYPE_STRUCT, anonymous_struct);
        }
      } else {
        ECHECK(ParseTypeFromProtoType(&type));
      }
      // Repeated elements get mapped to a vector.
      if (repeated) {
        type.element = type.base_type;
        type.base_type = BASE_TYPE_VECTOR;
        if (type.element == BASE_TYPE_VECTOR) {
          // We have a vector or vectors, which FlatBuffers doesn't support.
          // For now make it a vector of string (since the source is likely
          // "repeated bytes").
          // TODO(wvo): A better solution would be to wrap this in a table.
          type.element = BASE_TYPE_STRING;
        }
      }
      std::string name = attribute_;
      EXPECT(kTokenIdentifier);
      if (!oneof) {
        // Parse the field id. Since we're just translating schemas, not
        // any kind of binary compatibility, we can safely ignore these, and
        // assign our own.
        EXPECT('=');
        EXPECT(kTokenIntegerConstant);
      }
      FieldDef *field = nullptr;
      if (isextend) {
        // We allow a field to be re-defined when extending.
        // TODO: are there situations where that is problematic?
        field = struct_def->fields.Lookup(name);
      }
      if (!field) ECHECK(AddField(*struct_def, name, type, &field));
      field->doc_comment = field_comment;
      if (!IsScalar(type.base_type)) field->required = required;
      // See if there's a default specified.
      if (Is('[')) {
        NEXT();
        for (;;) {
          auto key = attribute_;
          ECHECK(ParseProtoKey());
          EXPECT('=');
          auto val = attribute_;
          ECHECK(ParseProtoCurliesOrIdent());
          if (key == "default") {
            // Temp: skip non-numeric defaults (enums).
            auto numeric = strpbrk(val.c_str(), "0123456789-+.");
            if (IsScalar(type.base_type) && numeric == val.c_str())
              field->value.constant = val;
          } else if (key == "deprecated") {
            field->deprecated = val == "true";
          }
          if (!Is(',')) break;
          NEXT();
        }
        EXPECT(']');
      }
      if (anonymous_struct) {
        ECHECK(ParseProtoFields(anonymous_struct, false, oneof));
        if (Is(';')) NEXT();
      } else if (oneof_union) {
        // Parse into a temporary StructDef, then transfer fields into an
        // EnumDef describing the oneof as a union.
        StructDef oneof_struct;
        ECHECK(ParseProtoFields(&oneof_struct, false, oneof));
        if (Is(';')) NEXT();
        for (auto field_it = oneof_struct.fields.vec.begin();
             field_it != oneof_struct.fields.vec.end(); ++field_it) {
          const auto &oneof_field = **field_it;
          const auto &oneof_type = oneof_field.value.type;
          if (oneof_type.base_type != BASE_TYPE_STRUCT ||
              !oneof_type.struct_def || oneof_type.struct_def->fixed)
            return Error("oneof '" + name +
                "' cannot be mapped to a union because member '" +
                oneof_field.name + "' is not a table type.");
          EnumValBuilder evb(*this, *oneof_union);
          auto ev = evb.CreateEnumerator(oneof_type.struct_def->name);
          ev->union_type = oneof_type;
          ev->doc_comment = oneof_field.doc_comment;
          ECHECK(evb.AcceptEnumerator(oneof_field.name));
        }
      } else {
        EXPECT(';');
      }
    }
  }
  NEXT();
  return NoError();
}

CheckedError Parser::ParseProtoKey() {
  if (token_ == '(') {
    NEXT();
    // Skip "(a.b)" style custom attributes.
    while (token_ == '.' || token_ == kTokenIdentifier) NEXT();
    EXPECT(')');
    while (Is('.')) {
      NEXT();
      EXPECT(kTokenIdentifier);
    }
  } else {
    EXPECT(kTokenIdentifier);
  }
  return NoError();
}

CheckedError Parser::ParseProtoCurliesOrIdent() {
  if (Is('{')) {
    NEXT();
    for (int nesting = 1; nesting;) {
      if (token_ == '{')
        nesting++;
      else if (token_ == '}')
        nesting--;
      NEXT();
    }
  } else {
    NEXT();  // Any single token.
  }
  return NoError();
}

CheckedError Parser::ParseProtoOption() {
  NEXT();
  ECHECK(ParseProtoKey());
  EXPECT('=');
  ECHECK(ParseProtoCurliesOrIdent());
  return NoError();
}

// Parse a protobuf type, and map it to the corresponding FlatBuffer one.
CheckedError Parser::ParseTypeFromProtoType(Type *type) {
  struct type_lookup {
    const char *proto_type;
    BaseType fb_type, element;
  };
  static type_lookup lookup[] = {
    { "float", BASE_TYPE_FLOAT, BASE_TYPE_NONE },
    { "double", BASE_TYPE_DOUBLE, BASE_TYPE_NONE },
    { "int32", BASE_TYPE_INT, BASE_TYPE_NONE },
    { "int64", BASE_TYPE_LONG, BASE_TYPE_NONE },
    { "uint32", BASE_TYPE_UINT, BASE_TYPE_NONE },
    { "uint64", BASE_TYPE_ULONG, BASE_TYPE_NONE },
    { "sint32", BASE_TYPE_INT, BASE_TYPE_NONE },
    { "sint64", BASE_TYPE_LONG, BASE_TYPE_NONE },
    { "fixed32", BASE_TYPE_UINT, BASE_TYPE_NONE },
    { "fixed64", BASE_TYPE_ULONG, BASE_TYPE_NONE },
    { "sfixed32", BASE_TYPE_INT, BASE_TYPE_NONE },
    { "sfixed64", BASE_TYPE_LONG, BASE_TYPE_NONE },
    { "bool", BASE_TYPE_BOOL, BASE_TYPE_NONE },
    { "string", BASE_TYPE_STRING, BASE_TYPE_NONE },
    { "bytes", BASE_TYPE_VECTOR, BASE_TYPE_UCHAR },
    { nullptr, BASE_TYPE_NONE, BASE_TYPE_NONE }
  };
  for (auto tl = lookup; tl->proto_type; tl++) {
    if (attribute_ == tl->proto_type) {
      type->base_type = tl->fb_type;
      type->element = tl->element;
      NEXT();
      return NoError();
    }
  }
  if (Is('.')) NEXT();  // qualified names may start with a . ?
  ECHECK(ParseTypeIdent(*type));
  return NoError();
}

CheckedError Parser::SkipAnyJsonValue() {
  switch (token_) {
    case '{': {
      size_t fieldn_outer = 0;
      return ParseTableDelimiters(
          fieldn_outer, nullptr,
          [&](const std::string &, size_t &fieldn,
              const StructDef *) -> CheckedError {
            ECHECK(Recurse([&]() { return SkipAnyJsonValue(); }));
            fieldn++;
            return NoError();
          });
    }
    case '[': {
      uoffset_t count = 0;
      return ParseVectorDelimiters(count, [&](uoffset_t &) -> CheckedError {
        return Recurse([&]() { return SkipAnyJsonValue(); });
      });
    }
    case kTokenStringConstant:
    case kTokenIntegerConstant:
    case kTokenFloatConstant: NEXT(); break;
    default:
      if (IsIdent("true") || IsIdent("false") || IsIdent("null")) {
        NEXT();
      } else
        return TokenError();
  }
  return NoError();
}

CheckedError Parser::ParseFlexBufferValue(flexbuffers::Builder *builder) {
  switch (token_) {
    case '{': {
      auto start = builder->StartMap();
      size_t fieldn_outer = 0;
      auto err =
          ParseTableDelimiters(fieldn_outer, nullptr,
                               [&](const std::string &name, size_t &fieldn,
                                   const StructDef *) -> CheckedError {
                                 builder->Key(name);
                                 ECHECK(ParseFlexBufferValue(builder));
                                 fieldn++;
                                 return NoError();
                               });
      ECHECK(err);
      builder->EndMap(start);
      break;
    }
    case '[': {
      auto start = builder->StartVector();
      uoffset_t count = 0;
      ECHECK(ParseVectorDelimiters(count, [&](uoffset_t &) -> CheckedError {
        return ParseFlexBufferValue(builder);
      }));
      builder->EndVector(start, false, false);
      break;
    }
    case kTokenStringConstant:
      builder->String(attribute_);
      EXPECT(kTokenStringConstant);
      break;
    case kTokenIntegerConstant:
      builder->Int(StringToInt(attribute_.c_str()));
      EXPECT(kTokenIntegerConstant);
      break;
    case kTokenFloatConstant:
      builder->Double(strtod(attribute_.c_str(), nullptr));
      EXPECT(kTokenFloatConstant);
      break;
    default:
      if (IsIdent("true")) {
        builder->Bool(true);
        NEXT();
      } else if (IsIdent("false")) {
        builder->Bool(false);
        NEXT();
      } else if (IsIdent("null")) {
        builder->Null();
        NEXT();
      } else
        return TokenError();
  }
  return NoError();
}

bool Parser::ParseFlexBuffer(const char *source, const char *source_filename,
                             flexbuffers::Builder *builder) {
  auto ok = !StartParseFile(source, source_filename).Check() &&
            !ParseFlexBufferValue(builder).Check();
  if (ok) builder->Finish();
  return ok;
}

bool Parser::Parse(const char *source, const char **include_paths,
                   const char *source_filename) {
  FLATBUFFERS_ASSERT(0 == recurse_protection_counter);
  auto r = !ParseRoot(source, include_paths, source_filename).Check();
  FLATBUFFERS_ASSERT(0 == recurse_protection_counter);
  return r;
}

CheckedError Parser::StartParseFile(const char *source,
                                    const char *source_filename) {
  file_being_parsed_ = source_filename ? source_filename : "";
  source_ = source;
  ResetState(source_);
  error_.clear();
  ECHECK(SkipByteOrderMark());
  NEXT();
  if (Is(kTokenEof)) return Error("input file is empty");
  return NoError();
}

CheckedError Parser::ParseRoot(const char *source, const char **include_paths,
                               const char *source_filename) {
  ECHECK(DoParse(source, include_paths, source_filename, nullptr));

  // Check that all types were defined.
  for (auto it = structs_.vec.begin(); it != structs_.vec.end();) {
    auto &struct_def = **it;
    if (struct_def.predecl) {
      if (opts.proto_mode) {
        // Protos allow enums to be used before declaration, so check if that
        // is the case here.
        EnumDef *enum_def = nullptr;
        for (size_t components =
                 struct_def.defined_namespace->components.size() + 1;
             components && !enum_def; components--) {
          auto qualified_name =
              struct_def.defined_namespace->GetFullyQualifiedName(
                  struct_def.name, components - 1);
          enum_def = LookupEnum(qualified_name);
        }
        if (enum_def) {
          // This is pretty slow, but a simple solution for now.
          auto initial_count = struct_def.refcount;
          for (auto struct_it = structs_.vec.begin();
               struct_it != structs_.vec.end(); ++struct_it) {
            auto &sd = **struct_it;
            for (auto field_it = sd.fields.vec.begin();
                 field_it != sd.fields.vec.end(); ++field_it) {
              auto &field = **field_it;
              if (field.value.type.struct_def == &struct_def) {
                field.value.type.struct_def = nullptr;
                field.value.type.enum_def = enum_def;
                auto &bt = field.value.type.base_type == BASE_TYPE_VECTOR
                               ? field.value.type.element
                               : field.value.type.base_type;
                FLATBUFFERS_ASSERT(bt == BASE_TYPE_STRUCT);
                bt = enum_def->underlying_type.base_type;
                struct_def.refcount--;
                enum_def->refcount++;
              }
            }
          }
          if (struct_def.refcount)
            return Error("internal: " + NumToString(struct_def.refcount) + "/" +
                         NumToString(initial_count) +
                         " use(s) of pre-declaration enum not accounted for: " +
                         enum_def->name);
          structs_.dict.erase(structs_.dict.find(struct_def.name));
          it = structs_.vec.erase(it);
          delete &struct_def;
          continue;  // Skip error.
        }
      }
      auto err = "type referenced but not defined (check namespace): " +
                 struct_def.name;
      if (struct_def.original_location)
        err += ", originally at: " + *struct_def.original_location;
      return Error(err);
    }
    ++it;
  }

  // This check has to happen here and not earlier, because only now do we
  // know for sure what the type of these are.
  for (auto it = enums_.vec.begin(); it != enums_.vec.end(); ++it) {
    auto &enum_def = **it;
    if (enum_def.is_union) {
      for (auto val_it = enum_def.Vals().begin();
           val_it != enum_def.Vals().end(); ++val_it) {
        auto &val = **val_it;
        if (!SupportsAdvancedUnionFeatures() && val.union_type.struct_def &&
            val.union_type.struct_def->fixed)
          return Error(
              "only tables can be union elements in the generated language: " +
              val.name);
      }
    }
  }
  return NoError();
}

CheckedError Parser::DoParse(const char *source, const char **include_paths,
                             const char *source_filename,
                             const char *include_filename) {
  if (source_filename) {
    if (included_files_.find(source_filename) == included_files_.end()) {
      included_files_[source_filename] =
          include_filename ? include_filename : "";
      files_included_per_file_[source_filename] = std::set<std::string>();
    } else {
      return NoError();
    }
  }
  if (!include_paths) {
    static const char *current_directory[] = { "", nullptr };
    include_paths = current_directory;
  }
  field_stack_.clear();
  builder_.Clear();
  // Start with a blank namespace just in case this file doesn't have one.
  current_namespace_ = empty_namespace_;

  ECHECK(StartParseFile(source, source_filename));

  // Includes must come before type declarations:
  for (;;) {
    // Parse pre-include proto statements if any:
    if (opts.proto_mode && (attribute_ == "option" || attribute_ == "syntax" ||
                            attribute_ == "package")) {
      ECHECK(ParseProtoDecl());
    } else if (IsIdent("native_include")) {
      NEXT();
      vector_emplace_back(&native_included_files_, attribute_);
      EXPECT(kTokenStringConstant);
      EXPECT(';');
    } else if (IsIdent("include") || (opts.proto_mode && IsIdent("import"))) {
      NEXT();
      if (opts.proto_mode && attribute_ == "public") NEXT();
      auto name = flatbuffers::PosixPath(attribute_.c_str());
      EXPECT(kTokenStringConstant);
      // Look for the file in include_paths.
      std::string filepath;
      for (auto paths = include_paths; paths && *paths; paths++) {
        filepath = flatbuffers::ConCatPathFileName(*paths, name);
        if (FileExists(filepath.c_str())) break;
      }
      if (filepath.empty())
        return Error("unable to locate include file: " + name);
      if (source_filename)
        files_included_per_file_[source_filename].insert(filepath);
      if (included_files_.find(filepath) == included_files_.end()) {
        // We found an include file that we have not parsed yet.
        // Load it and parse it.
        std::string contents;
        if (!LoadFile(filepath.c_str(), true, &contents))
          return Error("unable to load include file: " + name);
        ECHECK(DoParse(contents.c_str(), include_paths, filepath.c_str(),
                       name.c_str()));
        // We generally do not want to output code for any included files:
        if (!opts.generate_all) MarkGenerated();
        // Reset these just in case the included file had them, and the
        // parent doesn't.
        root_struct_def_ = nullptr;
        file_identifier_.clear();
        file_extension_.clear();
        // This is the easiest way to continue this file after an include:
        // instead of saving and restoring all the state, we simply start the
        // file anew. This will cause it to encounter the same include
        // statement again, but this time it will skip it, because it was
        // entered into included_files_.
        // This is recursive, but only go as deep as the number of include
        // statements.
        if (source_filename) {
          included_files_.erase(source_filename);
        }
        return DoParse(source, include_paths, source_filename,
                       include_filename);
      }
      EXPECT(';');
    } else {
      break;
    }
  }
  // Now parse all other kinds of declarations:
  while (token_ != kTokenEof) {
    if (opts.proto_mode) {
      ECHECK(ParseProtoDecl());
    } else if (IsIdent("namespace")) {
      ECHECK(ParseNamespace());
    } else if (token_ == '{') {
      if (!root_struct_def_)
        return Error("no root type set to parse json with");
      if (builder_.GetSize()) {
        return Error("cannot have more than one json object in a file");
      }
      uoffset_t toff;
      ECHECK(ParseTable(*root_struct_def_, nullptr, &toff));
      if (opts.size_prefixed) {
        builder_.FinishSizePrefixed(Offset<Table>(toff), file_identifier_.length()
                                                             ? file_identifier_.c_str()
                                                             : nullptr);
      } else {
        builder_.Finish(Offset<Table>(toff), file_identifier_.length()
                                                 ? file_identifier_.c_str()
                                                 : nullptr);
      }
      // Check that JSON file doesn't contain more objects or IDL directives.
      // Comments after JSON are allowed.
      EXPECT(kTokenEof);
    } else if (IsIdent("enum")) {
      ECHECK(ParseEnum(false, nullptr));
    } else if (IsIdent("union")) {
      ECHECK(ParseEnum(true, nullptr));
    } else if (IsIdent("root_type")) {
      NEXT();
      auto root_type = attribute_;
      EXPECT(kTokenIdentifier);
      ECHECK(ParseNamespacing(&root_type, nullptr));
      if (opts.root_type.empty()) {
        if (!SetRootType(root_type.c_str()))
          return Error("unknown root type: " + root_type);
        if (root_struct_def_->fixed)
          return Error("root type must be a table");
      }
      EXPECT(';');
    } else if (IsIdent("file_identifier")) {
      NEXT();
      file_identifier_ = attribute_;
      EXPECT(kTokenStringConstant);
      if (file_identifier_.length() != FlatBufferBuilder::kFileIdentifierLength)
        return Error("file_identifier must be exactly " +
                     NumToString(FlatBufferBuilder::kFileIdentifierLength) +
                     " characters");
      EXPECT(';');
    } else if (IsIdent("file_extension")) {
      NEXT();
      file_extension_ = attribute_;
      EXPECT(kTokenStringConstant);
      EXPECT(';');
    } else if (IsIdent("include")) {
      return Error("includes must come before declarations");
    } else if (IsIdent("attribute")) {
      NEXT();
      auto name = attribute_;
      if (Is(kTokenIdentifier)) {
        NEXT();
      } else {
        EXPECT(kTokenStringConstant);
      }
      EXPECT(';');
      known_attributes_[name] = false;
    } else if (IsIdent("rpc_service")) {
      ECHECK(ParseService());
    } else {
      ECHECK(ParseDecl());
    }
  }
  return NoError();
}

std::set<std::string> Parser::GetIncludedFilesRecursive(
    const std::string &file_name) const {
  std::set<std::string> included_files;
  std::list<std::string> to_process;

  if (file_name.empty()) return included_files;
  to_process.push_back(file_name);

  while (!to_process.empty()) {
    std::string current = to_process.front();
    to_process.pop_front();
    included_files.insert(current);

    // Workaround the lack of const accessor in C++98 maps.
    auto &new_files =
        (*const_cast<std::map<std::string, std::set<std::string>> *>(
            &files_included_per_file_))[current];
    for (auto it = new_files.begin(); it != new_files.end(); ++it) {
      if (included_files.find(*it) == included_files.end())
        to_process.push_back(*it);
    }
  }

  return included_files;
}

// Schema serialization functionality:

template<typename T> bool compareName(const T *a, const T *b) {
  return a->defined_namespace->GetFullyQualifiedName(a->name) <
         b->defined_namespace->GetFullyQualifiedName(b->name);
}

template<typename T> void AssignIndices(const std::vector<T *> &defvec) {
  // Pre-sort these vectors, such that we can set the correct indices for them.
  auto vec = defvec;
  std::sort(vec.begin(), vec.end(), compareName<T>);
  for (int i = 0; i < static_cast<int>(vec.size()); i++) vec[i]->index = i;
}

void Parser::Serialize() {
  builder_.Clear();
  AssignIndices(structs_.vec);
  AssignIndices(enums_.vec);
  std::vector<Offset<reflection::Object>> object_offsets;
  for (auto it = structs_.vec.begin(); it != structs_.vec.end(); ++it) {
    auto offset = (*it)->Serialize(&builder_, *this);
    object_offsets.push_back(offset);
    (*it)->serialized_location = offset.o;
  }
  std::vector<Offset<reflection::Enum>> enum_offsets;
  for (auto it = enums_.vec.begin(); it != enums_.vec.end(); ++it) {
    auto offset = (*it)->Serialize(&builder_, *this);
    enum_offsets.push_back(offset);
    (*it)->serialized_location = offset.o;
  }
  std::vector<Offset<reflection::Service>> service_offsets;
  for (auto it = services_.vec.begin(); it != services_.vec.end(); ++it) {
    auto offset = (*it)->Serialize(&builder_, *this);
    service_offsets.push_back(offset);
    (*it)->serialized_location = offset.o;
  }
  auto objs__ = builder_.CreateVectorOfSortedTables(&object_offsets);
  auto enum__ = builder_.CreateVectorOfSortedTables(&enum_offsets);
  auto fiid__ = builder_.CreateString(file_identifier_);
  auto fext__ = builder_.CreateString(file_extension_);
  auto serv__ = builder_.CreateVectorOfSortedTables(&service_offsets);
  auto schema_offset =
      reflection::CreateSchema(builder_, objs__, enum__, fiid__, fext__,
        (root_struct_def_ ? root_struct_def_->serialized_location : 0),
        serv__);
  if (opts.size_prefixed) {
    builder_.FinishSizePrefixed(schema_offset, reflection::SchemaIdentifier());
  } else {
    builder_.Finish(schema_offset, reflection::SchemaIdentifier());
  }
}

static Namespace *GetNamespace(
    const std::string &qualified_name, std::vector<Namespace *> &namespaces,
    std::map<std::string, Namespace *> &namespaces_index) {
  size_t dot = qualified_name.find_last_of('.');
  std::string namespace_name = (dot != std::string::npos)
                                   ? std::string(qualified_name.c_str(), dot)
                                   : "";
  Namespace *&ns = namespaces_index[namespace_name];

  if (!ns) {
    ns = new Namespace();
    namespaces.push_back(ns);

    size_t pos = 0;

    for (;;) {
      dot = qualified_name.find('.', pos);
      if (dot == std::string::npos) { break; }
      ns->components.push_back(qualified_name.substr(pos, dot - pos));
      pos = dot + 1;
    }
  }

  return ns;
}

Offset<reflection::Object> StructDef::Serialize(FlatBufferBuilder *builder,
                                                const Parser &parser) const {
  std::vector<Offset<reflection::Field>> field_offsets;
  for (auto it = fields.vec.begin(); it != fields.vec.end(); ++it) {
    field_offsets.push_back((*it)->Serialize(
        builder, static_cast<uint16_t>(it - fields.vec.begin()), parser));
  }
  auto qualified_name = defined_namespace->GetFullyQualifiedName(name);
  auto name__ = builder->CreateString(qualified_name);
  auto flds__ = builder->CreateVectorOfSortedTables(&field_offsets);
  auto attr__ = SerializeAttributes(builder, parser);
  auto docs__ = parser.opts.binary_schema_comments
                ? builder->CreateVectorOfStrings(doc_comment)
                : 0;
  return reflection::CreateObject(*builder, name__, flds__, fixed,
                                  static_cast<int>(minalign),
                                  static_cast<int>(bytesize),
                                  attr__, docs__);
}

bool StructDef::Deserialize(Parser &parser, const reflection::Object *object) {
  if (!DeserializeAttributes(parser, object->attributes()))
    return false;
  DeserializeDoc(doc_comment, object->documentation());
  name = parser.UnqualifiedName(object->name()->str());
  predecl = false;
  sortbysize = attributes.Lookup("original_order") == nullptr && !fixed;
  const auto& of = *(object->fields());
  auto indexes = std::vector<uoffset_t>(of.size());
  for (uoffset_t i = 0; i < of.size(); i++) indexes[of.Get(i)->id()] = i;
  size_t tmp_struct_size = 0;
  for (size_t i = 0; i < indexes.size(); i++) {
    auto field = of.Get(indexes[i]);
    auto field_def = new FieldDef();
    if (!field_def->Deserialize(parser, field) ||
        fields.Add(field_def->name, field_def)) {
      delete field_def;
      return false;
    }
    if (fixed) {
      // Recompute padding since that's currently not serialized.
      auto size = InlineSize(field_def->value.type);
      auto next_field =
          i + 1 < indexes.size()
          ? of.Get(indexes[i+1])
          : nullptr;
      tmp_struct_size += size;
      field_def->padding =
          next_field ? (next_field->offset() - field_def->value.offset) - size
                     : PaddingBytes(tmp_struct_size, minalign);
      tmp_struct_size += field_def->padding;
    }
  }
  FLATBUFFERS_ASSERT(static_cast<int>(tmp_struct_size) == object->bytesize());
  return true;
}

Offset<reflection::Field> FieldDef::Serialize(FlatBufferBuilder *builder,
                                              uint16_t id,
                                              const Parser &parser) const {
  auto name__ = builder->CreateString(name);
  auto type__ = value.type.Serialize(builder);
  auto attr__ = SerializeAttributes(builder, parser);
  auto docs__ = parser.opts.binary_schema_comments
                ? builder->CreateVectorOfStrings(doc_comment)
                : 0;
  return reflection::CreateField(*builder, name__, type__, id, value.offset,
      // Is uint64>max(int64) tested?
      IsInteger(value.type.base_type) ? StringToInt(value.constant.c_str()) : 0,
      // result may be platform-dependent if underlying is float (not double)
      IsFloat(value.type.base_type) ? strtod(value.constant.c_str(), nullptr)
                                    : 0.0,
      deprecated, required, key, attr__, docs__);
  // TODO: value.constant is almost always "0", we could save quite a bit of
  // space by sharing it. Same for common values of value.type.
}

bool FieldDef::Deserialize(Parser &parser, const reflection::Field *field) {
  name = field->name()->str();
  defined_namespace = parser.current_namespace_;
  if (!value.type.Deserialize(parser, field->type()))
    return false;
  value.offset = field->offset();
  if (IsInteger(value.type.base_type)) {
    value.constant = NumToString(field->default_integer());
  } else if (IsFloat(value.type.base_type)) {
    value.constant = FloatToString(field->default_real(), 16);
    size_t last_zero = value.constant.find_last_not_of('0');
    if (last_zero != std::string::npos && last_zero != 0) {
      value.constant.erase(last_zero, std::string::npos);
    }
  }
  deprecated = field->deprecated();
  required = field->required();
  key = field->key();
  if (!DeserializeAttributes(parser, field->attributes()))
    return false;
  // TODO: this should probably be handled by a separate attribute
  if (attributes.Lookup("flexbuffer")) {
    flexbuffer = true;
    parser.uses_flexbuffers_ = true;
    if (value.type.base_type != BASE_TYPE_VECTOR ||
        value.type.element != BASE_TYPE_UCHAR)
      return false;
  }
  if (auto nested = attributes.Lookup("nested_flatbuffer")) {
    auto nested_qualified_name =
        parser.current_namespace_->GetFullyQualifiedName(nested->constant);
    nested_flatbuffer = parser.LookupStruct(nested_qualified_name);
    if (!nested_flatbuffer) return false;
  }
  DeserializeDoc(doc_comment, field->documentation());
  return true;
}

Offset<reflection::RPCCall> RPCCall::Serialize(FlatBufferBuilder *builder,
                                               const Parser &parser) const {
  auto name__ = builder->CreateString(name);
  auto attr__ = SerializeAttributes(builder, parser);
  auto docs__ = parser.opts.binary_schema_comments
                ? builder->CreateVectorOfStrings(doc_comment)
                : 0;
  return reflection::CreateRPCCall(*builder, name__,
                                   request->serialized_location,
                                   response->serialized_location,
                                   attr__, docs__);
}

bool RPCCall::Deserialize(Parser &parser, const reflection::RPCCall *call) {
  name = call->name()->str();
  if (!DeserializeAttributes(parser, call->attributes()))
    return false;
  DeserializeDoc(doc_comment, call->documentation());
  request = parser.structs_.Lookup(call->request()->name()->str());
  response = parser.structs_.Lookup(call->response()->name()->str());
  if (!request || !response) { return false; }
  return true;
}

Offset<reflection::Service> ServiceDef::Serialize(FlatBufferBuilder *builder,
                                                  const Parser &parser) const {
  std::vector<Offset<reflection::RPCCall>> servicecall_offsets;
  for (auto it = calls.vec.begin(); it != calls.vec.end(); ++it) {
    servicecall_offsets.push_back((*it)->Serialize(builder, parser));
  }
  auto qualified_name = defined_namespace->GetFullyQualifiedName(name);
  auto name__ = builder->CreateString(qualified_name);
  auto call__ = builder->CreateVector(servicecall_offsets);
  auto attr__ = SerializeAttributes(builder, parser);
  auto docs__ = parser.opts.binary_schema_comments
                ? builder->CreateVectorOfStrings(doc_comment)
                : 0;
  return reflection::CreateService(*builder, name__, call__, attr__, docs__);
}

bool ServiceDef::Deserialize(Parser &parser,
                             const reflection::Service *service) {
  name = parser.UnqualifiedName(service->name()->str());
  if (service->calls()) {
    for (uoffset_t i = 0; i < service->calls()->size(); ++i) {
      auto call = new RPCCall();
      if (!call->Deserialize(parser, service->calls()->Get(i)) ||
          calls.Add(call->name, call)) {
        delete call;
        return false;
      }
    }
  }
  if (!DeserializeAttributes(parser, service->attributes()))
    return false;
  DeserializeDoc(doc_comment, service->documentation());
  return true;
}

Offset<reflection::Enum> EnumDef::Serialize(FlatBufferBuilder *builder,
                                            const Parser &parser) const {
  std::vector<Offset<reflection::EnumVal>> enumval_offsets;
  for (auto it = vals.vec.begin(); it != vals.vec.end(); ++it) {
    enumval_offsets.push_back((*it)->Serialize(builder, parser));
  }
  auto qualified_name = defined_namespace->GetFullyQualifiedName(name);
  auto name__ = builder->CreateString(qualified_name);
  auto vals__ = builder->CreateVector(enumval_offsets);
  auto type__ = underlying_type.Serialize(builder);
  auto attr__ = SerializeAttributes(builder, parser);
  auto docs__ = parser.opts.binary_schema_comments
                ? builder->CreateVectorOfStrings(doc_comment)
                : 0;
  return reflection::CreateEnum(*builder, name__, vals__, is_union, type__,
                                attr__, docs__);
}

bool EnumDef::Deserialize(Parser &parser, const reflection::Enum *_enum) {
  name = parser.UnqualifiedName(_enum->name()->str());
  for (uoffset_t i = 0; i < _enum->values()->size(); ++i) {
    auto val = new EnumVal();
    if (!val->Deserialize(parser, _enum->values()->Get(i)) ||
        vals.Add(val->name, val)) {
      delete val;
      return false;
    }
  }
  is_union = _enum->is_union();
  if (!underlying_type.Deserialize(parser, _enum->underlying_type())) {
    return false;
  }
  if (!DeserializeAttributes(parser, _enum->attributes()))
    return false;
  DeserializeDoc(doc_comment, _enum->documentation());
  return true;
}

Offset<reflection::EnumVal> EnumVal::Serialize(FlatBufferBuilder *builder,
                                               const Parser &parser) const {
  auto name__ = builder->CreateString(name);
  auto type__ = union_type.Serialize(builder);
  auto docs__ = parser.opts.binary_schema_comments
                ? builder->CreateVectorOfStrings(doc_comment)
                : 0;
  return reflection::CreateEnumVal(*builder, name__, value,
      union_type.struct_def ? union_type.struct_def->serialized_location : 0,
      type__, docs__);
}

bool EnumVal::Deserialize(const Parser &parser,
                          const reflection::EnumVal *val) {
  name = val->name()->str();
  value = val->value();
  if (!union_type.Deserialize(parser, val->union_type()))
    return false;
  DeserializeDoc(doc_comment, val->documentation());
  return true;
}

Offset<reflection::Type> Type::Serialize(FlatBufferBuilder *builder) const {
  return reflection::CreateType(
      *builder, static_cast<reflection::BaseType>(base_type),
      static_cast<reflection::BaseType>(element),
      struct_def ? struct_def->index : (enum_def ? enum_def->index : -1),
      fixed_length);
}

bool Type::Deserialize(const Parser &parser, const reflection::Type *type) {
  if (type == nullptr) return true;
  base_type = static_cast<BaseType>(type->base_type());
  element = static_cast<BaseType>(type->element());
  fixed_length = type->fixed_length();
  if (type->index() >= 0) {
    bool is_series = type->base_type() == reflection::Vector ||
                     type->base_type() == reflection::Array;
    if (type->base_type() == reflection::Obj ||
        (is_series &&
         type->element() == reflection::Obj)) {
      if (static_cast<size_t>(type->index()) < parser.structs_.vec.size()) {
        struct_def = parser.structs_.vec[type->index()];
        struct_def->refcount++;
      } else {
        return false;
      }
    } else {
      if (static_cast<size_t>(type->index()) < parser.enums_.vec.size()) {
        enum_def = parser.enums_.vec[type->index()];
      } else {
        return false;
      }
    }
  }
  return true;
}

flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<reflection::KeyValue>>>
Definition::SerializeAttributes(FlatBufferBuilder *builder,
                                const Parser &parser) const {
  std::vector<flatbuffers::Offset<reflection::KeyValue>> attrs;
  for (auto kv = attributes.dict.begin(); kv != attributes.dict.end(); ++kv) {
    auto it = parser.known_attributes_.find(kv->first);
    FLATBUFFERS_ASSERT(it != parser.known_attributes_.end());
    if (parser.opts.binary_schema_builtins || !it->second) {
      auto key = builder->CreateString(kv->first);
      auto val = builder->CreateString(kv->second->constant);
      attrs.push_back(reflection::CreateKeyValue(*builder, key, val));
    }
  }
  if (attrs.size()) {
    return builder->CreateVectorOfSortedTables(&attrs);
  } else {
    return 0;
  }
}

bool Definition::DeserializeAttributes(
    Parser &parser, const Vector<Offset<reflection::KeyValue>> *attrs) {
  if (attrs == nullptr)
    return true;
  for (uoffset_t i = 0; i < attrs->size(); ++i) {
    auto kv = attrs->Get(i);
    auto value = new Value();
    if (kv->value()) { value->constant = kv->value()->str(); }
    if (attributes.Add(kv->key()->str(), value)) {
      delete value;
      return false;
    }
    parser.known_attributes_[kv->key()->str()];
  }
  return true;
}

/************************************************************************/
/* DESERIALIZATION                                                      */
/************************************************************************/
bool Parser::Deserialize(const uint8_t *buf, const size_t size) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(buf), size);
  bool size_prefixed = false;
  if(!reflection::SchemaBufferHasIdentifier(buf)) {
    if (!flatbuffers::BufferHasIdentifier(buf, reflection::SchemaIdentifier(),
                                          true))
      return false;
    else
      size_prefixed = true;
  }
  auto verify_fn = size_prefixed ? &reflection::VerifySizePrefixedSchemaBuffer
                                 : &reflection::VerifySchemaBuffer;
  if (!verify_fn(verifier)) {
    return false;
  }
  auto schema = size_prefixed ? reflection::GetSizePrefixedSchema(buf)
                              : reflection::GetSchema(buf);
  return Deserialize(schema);
}

bool Parser::Deserialize(const reflection::Schema *schema) {
  file_identifier_ = schema->file_ident() ? schema->file_ident()->str() : "";
  file_extension_ = schema->file_ext() ? schema->file_ext()->str() : "";
  std::map<std::string, Namespace *> namespaces_index;

  // Create defs without deserializing so references from fields to structs and
  // enums can be resolved.
  for (auto it = schema->objects()->begin(); it != schema->objects()->end();
       ++it) {
    auto struct_def = new StructDef();
    struct_def->bytesize = it->bytesize();
    struct_def->fixed = it->is_struct();
    struct_def->minalign = it->minalign();
    if (structs_.Add(it->name()->str(), struct_def)) {
      delete struct_def;
      return false;
    }
    auto type = new Type(BASE_TYPE_STRUCT, struct_def, nullptr);
    if (types_.Add(it->name()->str(), type)) {
      delete type;
      return false;
    }
  }
  for (auto it = schema->enums()->begin(); it != schema->enums()->end(); ++it) {
    auto enum_def = new EnumDef();
    if (enums_.Add(it->name()->str(), enum_def)) {
      delete enum_def;
      return false;
    }
    auto type = new Type(BASE_TYPE_UNION, nullptr, enum_def);
    if (types_.Add(it->name()->str(), type)) {
      delete type;
      return false;
    }
  }

  // Now fields can refer to structs and enums by index.
  for (auto it = schema->objects()->begin(); it != schema->objects()->end();
       ++it) {
    std::string qualified_name = it->name()->str();
    auto struct_def = structs_.Lookup(qualified_name);
    struct_def->defined_namespace =
        GetNamespace(qualified_name, namespaces_, namespaces_index);
    if (!struct_def->Deserialize(*this, * it)) { return false; }
    if (schema->root_table() == *it) { root_struct_def_ = struct_def; }
  }
  for (auto it = schema->enums()->begin(); it != schema->enums()->end(); ++it) {
    std::string qualified_name = it->name()->str();
    auto enum_def = enums_.Lookup(qualified_name);
    enum_def->defined_namespace =
        GetNamespace(qualified_name, namespaces_, namespaces_index);
    if (!enum_def->Deserialize(*this, *it)) { return false; }
  }

  if (schema->services()) {
    for (auto it = schema->services()->begin(); it != schema->services()->end();
         ++it) {
      std::string qualified_name = it->name()->str();
      auto service_def = new ServiceDef();
      service_def->defined_namespace =
          GetNamespace(qualified_name, namespaces_, namespaces_index);
      if (!service_def->Deserialize(*this, *it) ||
          services_.Add(qualified_name, service_def)) {
        delete service_def;
        return false;
      }
    }
  }

  return true;
}

std::string Parser::ConformTo(const Parser &base) {
  for (auto sit = structs_.vec.begin(); sit != structs_.vec.end(); ++sit) {
    auto &struct_def = **sit;
    auto qualified_name =
        struct_def.defined_namespace->GetFullyQualifiedName(struct_def.name);
    auto struct_def_base = base.LookupStruct(qualified_name);
    if (!struct_def_base) continue;
    for (auto fit = struct_def.fields.vec.begin();
         fit != struct_def.fields.vec.end(); ++fit) {
      auto &field = **fit;
      auto field_base = struct_def_base->fields.Lookup(field.name);
      if (field_base) {
        if (field.value.offset != field_base->value.offset)
          return "offsets differ for field: " + field.name;
        if (field.value.constant != field_base->value.constant)
          return "defaults differ for field: " + field.name;
        if (!EqualByName(field.value.type, field_base->value.type))
          return "types differ for field: " + field.name;
      } else {
        // Doesn't have to exist, deleting fields is fine.
        // But we should check if there is a field that has the same offset
        // but is incompatible (in the case of field renaming).
        for (auto fbit = struct_def_base->fields.vec.begin();
             fbit != struct_def_base->fields.vec.end(); ++fbit) {
          field_base = *fbit;
          if (field.value.offset == field_base->value.offset) {
            if (!EqualByName(field.value.type, field_base->value.type))
              return "field renamed to different type: " + field.name;
            break;
          }
        }
      }
    }
  }
  for (auto eit = enums_.vec.begin(); eit != enums_.vec.end(); ++eit) {
    auto &enum_def = **eit;
    auto qualified_name =
        enum_def.defined_namespace->GetFullyQualifiedName(enum_def.name);
    auto enum_def_base = base.enums_.Lookup(qualified_name);
    if (!enum_def_base) continue;
    for (auto evit = enum_def.Vals().begin(); evit != enum_def.Vals().end();
         ++evit) {
      auto &enum_val = **evit;
      auto enum_val_base = enum_def_base->Lookup(enum_val.name);
      if (enum_val_base) {
        if (enum_val != *enum_val_base)
          return "values differ for enum: " + enum_val.name;
      }
    }
  }
  return "";
}

}  // namespace flatbuffers
