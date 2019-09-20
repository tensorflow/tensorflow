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

// independent from idl_parser, since this code is not needed for most clients

#include "flatbuffers/code_generators.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#if defined(FLATBUFFERS_CPP98_STL)
#  include <cctype>
#endif  // defined(FLATBUFFERS_CPP98_STL)

namespace flatbuffers {

// These arrays need to correspond to the IDLOptions::k enum.

struct LanguageParameters {
  IDLOptions::Language language;
  // Whether function names in the language typically start with uppercase.
  bool first_camel_upper;
  std::string file_extension;
  std::string string_type;
  std::string bool_type;
  std::string open_curly;
  std::string accessor_type;
  std::string const_decl;
  std::string unsubclassable_decl;
  std::string enum_decl;
  std::string enum_separator;
  std::string getter_prefix;
  std::string getter_suffix;
  std::string inheritance_marker;
  std::string namespace_ident;
  std::string namespace_begin;
  std::string namespace_end;
  std::string set_bb_byteorder;
  std::string get_bb_position;
  std::string get_fbb_offset;
  std::string accessor_prefix;
  std::string accessor_prefix_static;
  std::string optional_suffix;
  std::string includes;
  std::string class_annotation;
  std::string generated_type_annotation;
  CommentConfig comment_config;
  const FloatConstantGenerator *float_gen;
};

const LanguageParameters &GetLangParams(IDLOptions::Language lang) {
  static TypedFloatConstantGenerator CSharpFloatGen(
      "Double.", "Single.", "NaN", "PositiveInfinity", "NegativeInfinity");

  static TypedFloatConstantGenerator JavaFloatGen(
      "Double.", "Float.", "NaN", "POSITIVE_INFINITY", "NEGATIVE_INFINITY");

  static const LanguageParameters language_parameters[] = {
    {
        IDLOptions::kJava,
        false,
        ".java",
        "String",
        "boolean ",
        " {\n",
        "class ",
        " final ",
        "final ",
        "final class ",
        ";\n",
        "()",
        "",
        " extends ",
        "package ",
        ";",
        "",
        "_bb.order(ByteOrder.LITTLE_ENDIAN); ",
        "position()",
        "offset()",
        "",
        "",
        "",
        "import java.nio.*;\nimport java.lang.*;\nimport "
        "java.util.*;\nimport com.google.flatbuffers.*;\n",
        "\n@SuppressWarnings(\"unused\")\n",
        "\n@javax.annotation.Generated(value=\"flatc\")\n",
        {
            "/**",
            " *",
            " */",
        },
        &JavaFloatGen
    },
    {
        IDLOptions::kCSharp,
        true,
        ".cs",
        "string",
        "bool ",
        "\n{\n",
        "struct ",
        " readonly ",
        "",
        "enum ",
        ",\n",
        " { get",
        "} ",
        " : ",
        "namespace ",
        "\n{",
        "\n}\n",
        "",
        "Position",
        "Offset",
        "__p.",
        "Table.",
        "?",
        "using global::System;\nusing global::FlatBuffers;\n\n",
        "",
        "",
        {
            nullptr,
            "///",
            nullptr,
        },
        &CSharpFloatGen
    },
  };

  if (lang == IDLOptions::kJava) {
    return language_parameters[0];
  } else {
    FLATBUFFERS_ASSERT(lang == IDLOptions::kCSharp);
    return language_parameters[1];
  }
}

namespace general {
class GeneralGenerator : public BaseGenerator {
 public:
  GeneralGenerator(const Parser &parser, const std::string &path,
                   const std::string &file_name)
      : BaseGenerator(parser, path, file_name, "", "."),
        lang_(GetLangParams(parser_.opts.lang)),
        cur_name_space_(nullptr) {}

  GeneralGenerator &operator=(const GeneralGenerator &);
  bool generate() {
    std::string one_file_code;
    cur_name_space_ = parser_.current_namespace_;

    for (auto it = parser_.enums_.vec.begin(); it != parser_.enums_.vec.end();
         ++it) {
      std::string enumcode;
      auto &enum_def = **it;
      if (!parser_.opts.one_file) cur_name_space_ = enum_def.defined_namespace;
      GenEnum(enum_def, &enumcode);
      if (parser_.opts.one_file) {
        one_file_code += enumcode;
      } else {
        if (!SaveType(enum_def.name, *enum_def.defined_namespace, enumcode,
                      false))
          return false;
      }
    }

    for (auto it = parser_.structs_.vec.begin();
         it != parser_.structs_.vec.end(); ++it) {
      std::string declcode;
      auto &struct_def = **it;
      if (!parser_.opts.one_file)
        cur_name_space_ = struct_def.defined_namespace;
      GenStruct(struct_def, &declcode);
      if (parser_.opts.one_file) {
        one_file_code += declcode;
      } else {
        if (!SaveType(struct_def.name, *struct_def.defined_namespace, declcode,
                      true))
          return false;
      }
    }

    if (parser_.opts.one_file) {
      return SaveType(file_name_, *parser_.current_namespace_, one_file_code,
                      true);
    }
    return true;
  }

  // Save out the generated code for a single class while adding
  // declaration boilerplate.
  bool SaveType(const std::string &defname, const Namespace &ns,
                const std::string &classcode, bool needs_includes) const {
    if (!classcode.length()) return true;

    std::string code;
    if (lang_.language == IDLOptions::kCSharp) {
      code =
          "// <auto-generated>\n"
          "//  " +
          std::string(FlatBuffersGeneratedWarning()) +
          "\n"
          "// </auto-generated>\n\n";
    } else {
      code = "// " + std::string(FlatBuffersGeneratedWarning()) + "\n\n";
    }

    std::string namespace_name = FullNamespace(".", ns);
    if (!namespace_name.empty()) {
      code += lang_.namespace_ident + namespace_name + lang_.namespace_begin;
      code += "\n\n";
    }
    if (needs_includes) {
      code += lang_.includes;
      if (parser_.opts.gen_nullable) {
        code += "\nimport javax.annotation.Nullable;\n";
      }
      if (parser_.opts.java_checkerframework) {
        code += "\nimport org.checkerframework.dataflow.qual.Pure;\n";
      }
      code += lang_.class_annotation;
    }
    if (parser_.opts.gen_generated) {
      code += lang_.generated_type_annotation;
    }
    code += classcode;
    if (!namespace_name.empty()) code += lang_.namespace_end;
    auto filename = NamespaceDir(ns) + defname + lang_.file_extension;
    return SaveFile(filename.c_str(), code, false);
  }

  const Namespace *CurrentNameSpace() const { return cur_name_space_; }

  std::string FunctionStart(char upper) const {
    return std::string() + (lang_.language == IDLOptions::kJava
                                ? static_cast<char>(tolower(upper))
                                : upper);
  }

  std::string GenNullableAnnotation(const Type &t) const {
    return lang_.language == IDLOptions::kJava && parser_.opts.gen_nullable &&
                   !IsScalar(DestinationType(t, true).base_type)
               ? " @Nullable "
               : "";
  }

  std::string GenPureAnnotation(const Type &t) const {
    return lang_.language == IDLOptions::kJava &&
                   parser_.opts.java_checkerframework &&
                   !IsScalar(DestinationType(t, true).base_type)
               ? " @Pure "
               : "";
  }

  std::string GenTypeBasic(const Type &type, bool enableLangOverrides) const {
    // clang-format off
  static const char * const java_typename[] = {
    #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
        CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
        #JTYPE,
      FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
    #undef FLATBUFFERS_TD
  };

  static const char * const csharp_typename[] = {
    #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
        CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
        #NTYPE,
      FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
    #undef FLATBUFFERS_TD
  };
    // clang-format on

    if (enableLangOverrides) {
      if (lang_.language == IDLOptions::kCSharp) {
        if (IsEnum(type)) return WrapInNameSpace(*type.enum_def);
        if (type.base_type == BASE_TYPE_STRUCT) {
          return "Offset<" + WrapInNameSpace(*type.struct_def) + ">";
        }
      }
    }

    if (lang_.language == IDLOptions::kJava) {
      return java_typename[type.base_type];
    } else {
      FLATBUFFERS_ASSERT(lang_.language == IDLOptions::kCSharp);
      return csharp_typename[type.base_type];
    }
  }

  std::string GenTypeBasic(const Type &type) const {
    return GenTypeBasic(type, true);
  }

  std::string GenTypePointer(const Type &type) const {
    switch (type.base_type) {
      case BASE_TYPE_STRING: return lang_.string_type;
      case BASE_TYPE_VECTOR: return GenTypeGet(type.VectorType());
      case BASE_TYPE_STRUCT: return WrapInNameSpace(*type.struct_def);
      case BASE_TYPE_UNION:
        // Unions in C# use a generic Table-derived type for better type safety
        if (lang_.language == IDLOptions::kCSharp) return "TTable";
        FLATBUFFERS_FALLTHROUGH();  // else fall thru
      default: return "Table";
    }
  }

  std::string GenTypeGet(const Type &type) const {
    return IsScalar(type.base_type)
               ? GenTypeBasic(type)
               : (IsArray(type) ? GenTypeGet(type.VectorType())
                                : GenTypePointer(type));
  }

  // Find the destination type the user wants to receive the value in (e.g.
  // one size higher signed types for unsigned serialized values in Java).
  Type DestinationType(const Type &type, bool vectorelem) const {
    if (lang_.language != IDLOptions::kJava) return type;
    switch (type.base_type) {
      // We use int for both uchar/ushort, since that generally means less
      // casting than using short for uchar.
      case BASE_TYPE_UCHAR: return Type(BASE_TYPE_INT);
      case BASE_TYPE_USHORT: return Type(BASE_TYPE_INT);
      case BASE_TYPE_UINT: return Type(BASE_TYPE_LONG);
      case BASE_TYPE_ARRAY:
      case BASE_TYPE_VECTOR:
        if (vectorelem) return DestinationType(type.VectorType(), vectorelem);
        FLATBUFFERS_FALLTHROUGH(); // else fall thru
      default: return type;
    }
  }

  std::string GenOffsetType(const StructDef &struct_def) const {
    if (lang_.language == IDLOptions::kCSharp) {
      return "Offset<" + WrapInNameSpace(struct_def) + ">";
    } else {
      return "int";
    }
  }

  std::string GenOffsetConstruct(const StructDef &struct_def,
                                 const std::string &variable_name) const {
    if (lang_.language == IDLOptions::kCSharp) {
      return "new Offset<" + WrapInNameSpace(struct_def) + ">(" +
             variable_name + ")";
    }
    return variable_name;
  }

  std::string GenVectorOffsetType() const {
    if (lang_.language == IDLOptions::kCSharp) {
      return "VectorOffset";
    } else {
      return "int";
    }
  }

  // Generate destination type name
  std::string GenTypeNameDest(const Type &type) const {
    return GenTypeGet(DestinationType(type, true));
  }

  // Mask to turn serialized value into destination type value.
  std::string DestinationMask(const Type &type, bool vectorelem) const {
    if (lang_.language != IDLOptions::kJava) return "";
    switch (type.base_type) {
      case BASE_TYPE_UCHAR: return " & 0xFF";
      case BASE_TYPE_USHORT: return " & 0xFFFF";
      case BASE_TYPE_UINT: return " & 0xFFFFFFFFL";
      case BASE_TYPE_VECTOR:
        if (vectorelem) return DestinationMask(type.VectorType(), vectorelem);
        FLATBUFFERS_FALLTHROUGH(); // else fall thru
      default: return "";
    }
  }

  // Casts necessary to correctly read serialized data
  std::string DestinationCast(const Type &type) const {
    if (IsSeries(type)) {
      return DestinationCast(type.VectorType());
    } else {
      switch (lang_.language) {
        case IDLOptions::kJava:
          // Cast necessary to correctly read serialized unsigned values.
          if (type.base_type == BASE_TYPE_UINT) return "(long)";
          break;

        case IDLOptions::kCSharp:
          // Cast from raw integral types to enum.
          if (IsEnum(type)) return "(" + WrapInNameSpace(*type.enum_def) + ")";
          break;

        default: break;
      }
    }
    return "";
  }

  // Cast statements for mutator method parameters.
  // In Java, parameters representing unsigned numbers need to be cast down to
  // their respective type. For example, a long holding an unsigned int value
  // would be cast down to int before being put onto the buffer. In C#, one cast
  // directly cast an Enum to its underlying type, which is essential before
  // putting it onto the buffer.
  std::string SourceCast(const Type &type, bool castFromDest) const {
    if (IsSeries(type)) {
      return SourceCast(type.VectorType(), castFromDest);
    } else {
      switch (lang_.language) {
        case IDLOptions::kJava:
          if (castFromDest) {
            if (type.base_type == BASE_TYPE_UINT)
              return "(int)";
            else if (type.base_type == BASE_TYPE_USHORT)
              return "(short)";
            else if (type.base_type == BASE_TYPE_UCHAR)
              return "(byte)";
          }
          break;
        case IDLOptions::kCSharp:
          if (IsEnum(type)) return "(" + GenTypeBasic(type, false) + ")";
          break;
        default: break;
      }
    }
    return "";
  }

  std::string SourceCast(const Type &type) const { return SourceCast(type, true); }

  std::string SourceCastBasic(const Type &type, bool castFromDest) const {
    return IsScalar(type.base_type) ? SourceCast(type, castFromDest) : "";
  }

  std::string SourceCastBasic(const Type &type) const {
    return SourceCastBasic(type, true);
  }

  std::string GenEnumDefaultValue(const FieldDef &field) const {
    auto &value = field.value;
    FLATBUFFERS_ASSERT(value.type.enum_def);
    auto &enum_def = *value.type.enum_def;
    auto enum_val = enum_def.FindByValue(value.constant);
    return enum_val ? (WrapInNameSpace(enum_def) + "." + enum_val->name)
                    : value.constant;
  }

  std::string GenDefaultValue(const FieldDef &field, bool enableLangOverrides) const {
    auto& value = field.value;
    if (enableLangOverrides) {
      // handles both enum case and vector of enum case
      if (lang_.language == IDLOptions::kCSharp &&
          value.type.enum_def != nullptr &&
          value.type.base_type != BASE_TYPE_UNION) {
        return GenEnumDefaultValue(field);
      }
    }

    auto longSuffix = lang_.language == IDLOptions::kJava ? "L" : "";
    switch (value.type.base_type) {
      case BASE_TYPE_BOOL: return value.constant == "0" ? "false" : "true";
      case BASE_TYPE_ULONG: {
        if (lang_.language != IDLOptions::kJava) return value.constant;
        // Converts the ulong into its bits signed equivalent
        uint64_t defaultValue = StringToUInt(value.constant.c_str());
        return NumToString(static_cast<int64_t>(defaultValue)) + longSuffix;
      }
      case BASE_TYPE_UINT:
      case BASE_TYPE_LONG: return value.constant + longSuffix;
      default:
        if(IsFloat(value.type.base_type))
          return lang_.float_gen->GenFloatConstant(field);
        else
          return value.constant;
    }
  }

  std::string GenDefaultValue(const FieldDef &field) const {
    return GenDefaultValue(field, true);
  }

  std::string GenDefaultValueBasic(const FieldDef &field,
                                   bool enableLangOverrides) const {
    auto& value = field.value;
    if (!IsScalar(value.type.base_type)) {
      if (enableLangOverrides) {
        if (lang_.language == IDLOptions::kCSharp) {
          switch (value.type.base_type) {
            case BASE_TYPE_STRING: return "default(StringOffset)";
            case BASE_TYPE_STRUCT:
              return "default(Offset<" +
                     WrapInNameSpace(*value.type.struct_def) + ">)";
            case BASE_TYPE_VECTOR: return "default(VectorOffset)";
            default: break;
          }
        }
      }
      return "0";
    }
    return GenDefaultValue(field, enableLangOverrides);
  }

  std::string GenDefaultValueBasic(const FieldDef &field) const {
    return GenDefaultValueBasic(field, true);
  }

  void GenEnum(EnumDef &enum_def, std::string *code_ptr) const {
    std::string &code = *code_ptr;
    if (enum_def.generated) return;

    // Generate enum definitions of the form:
    // public static (final) int name = value;
    // In Java, we use ints rather than the Enum feature, because we want them
    // to map directly to how they're used in C/C++ and file formats.
    // That, and Java Enums are expensive, and not universally liked.
    GenComment(enum_def.doc_comment, code_ptr, &lang_.comment_config);

    // In C# this indicates enumeration values can be treated as bit flags.
    if (lang_.language == IDLOptions::kCSharp && enum_def.attributes.Lookup("bit_flags")) {
      code += "[System.FlagsAttribute]\n";
    }
    if (enum_def.attributes.Lookup("private")) {
      // For Java, we leave the enum unmarked to indicate package-private
      // For C# we mark the enum as internal
      if (lang_.language == IDLOptions::kCSharp) {
        code += "internal ";
      }
    } else {
      code += "public ";
    }
    code += lang_.enum_decl + enum_def.name;
    if (lang_.language == IDLOptions::kCSharp) {
      code += lang_.inheritance_marker +
              GenTypeBasic(enum_def.underlying_type, false);
    }
    code += lang_.open_curly;
    if (lang_.language == IDLOptions::kJava) {
      code += "  private " + enum_def.name + "() { }\n";
    }
    for (auto it = enum_def.Vals().begin(); it != enum_def.Vals().end(); ++it) {
      auto &ev = **it;
      GenComment(ev.doc_comment, code_ptr, &lang_.comment_config, "  ");
      if (lang_.language != IDLOptions::kCSharp) {
        code += "  public static";
        code += lang_.const_decl;
        code += GenTypeBasic(enum_def.underlying_type, false);
      }
      code += (lang_.language == IDLOptions::kJava) ? " " : "  ";
      code += ev.name + " = ";
      code += enum_def.ToString(ev);
      code += lang_.enum_separator;
    }

    // Generate a generate string table for enum values.
    // We do not do that for C# where this functionality is native.
    if (lang_.language != IDLOptions::kCSharp) {
      // Problem is, if values are very sparse that could generate really big
      // tables. Ideally in that case we generate a map lookup instead, but for
      // the moment we simply don't output a table at all.
      auto range = enum_def.Distance();
      // Average distance between values above which we consider a table
      // "too sparse". Change at will.
      static const uint64_t kMaxSparseness = 5;
      if (range / static_cast<uint64_t>(enum_def.size()) < kMaxSparseness) {
        code += "\n  public static";
        code += lang_.const_decl;
        code += lang_.string_type;
        code += "[] names = { ";
        auto val = enum_def.Vals().front();
        for (auto it = enum_def.Vals().begin(); it != enum_def.Vals().end();
             ++it) {
          auto ev = *it;
          for (auto k = enum_def.Distance(val, ev); k > 1; --k)
            code += "\"\", ";
          val = ev;
          code += "\"" + (*it)->name + "\", ";
        }
        code += "};\n\n";
        code += "  public static ";
        code += lang_.string_type;
        code += " " + MakeCamel("name", lang_.first_camel_upper);
        code += "(int e) { return names[e";
        if (enum_def.MinValue()->IsNonZero())
          code += " - " + enum_def.MinValue()->name;
        code += "]; }\n";
      }
    }

    // Close the class
    code += "}";
    // Java does not need the closing semi-colon on class definitions.
    code += (lang_.language != IDLOptions::kJava) ? ";" : "";
    code += "\n\n";
  }

  // Returns the function name that is able to read a value of the given type.
  std::string GenGetter(const Type &type) const {
    switch (type.base_type) {
      case BASE_TYPE_STRING: return lang_.accessor_prefix + "__string";
      case BASE_TYPE_STRUCT: return lang_.accessor_prefix + "__struct";
      case BASE_TYPE_UNION: return lang_.accessor_prefix + "__union";
      case BASE_TYPE_VECTOR: return GenGetter(type.VectorType());
      case BASE_TYPE_ARRAY: return GenGetter(type.VectorType());
      default: {
        std::string getter =
            lang_.accessor_prefix + "bb." + FunctionStart('G') + "et";
        if (type.base_type == BASE_TYPE_BOOL) {
          getter = "0!=" + getter;
        } else if (GenTypeBasic(type, false) != "byte") {
          getter += MakeCamel(GenTypeBasic(type, false));
        }
        return getter;
      }
    }
  }

  // Returns the function name that is able to read a value of the given type.
  std::string GenGetterForLookupByKey(flatbuffers::FieldDef *key_field,
                                      const std::string &data_buffer,
                                      const char *num = nullptr) const {
    auto type = key_field->value.type;
    auto dest_mask = DestinationMask(type, true);
    auto dest_cast = DestinationCast(type);
    auto getter = data_buffer + "." + FunctionStart('G') + "et";
    if (GenTypeBasic(type, false) != "byte") {
      getter += MakeCamel(GenTypeBasic(type, false));
    }
    getter = dest_cast + getter + "(" + GenOffsetGetter(key_field, num) + ")" +
             dest_mask;
    return getter;
  }

  // Direct mutation is only allowed for scalar fields.
  // Hence a setter method will only be generated for such fields.
  std::string GenSetter(const Type &type) const {
    if (IsScalar(type.base_type)) {
      std::string setter =
          lang_.accessor_prefix + "bb." + FunctionStart('P') + "ut";
      if (GenTypeBasic(type, false) != "byte" &&
          type.base_type != BASE_TYPE_BOOL) {
        setter += MakeCamel(GenTypeBasic(type, false));
      }
      return setter;
    } else {
      return "";
    }
  }

  // Returns the method name for use with add/put calls.
  std::string GenMethod(const Type &type) const {
    return IsScalar(type.base_type) ? MakeCamel(GenTypeBasic(type, false))
                                    : (IsStruct(type) ? "Struct" : "Offset");
  }

  // Recursively generate arguments for a constructor, to deal with nested
  // structs.
  void GenStructArgs(const StructDef &struct_def, std::string *code_ptr,
                     const char *nameprefix, size_t array_count = 0) const {
    std::string &code = *code_ptr;
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      const auto &field_type = field.value.type;
      const auto array_field = IsArray(field_type);
      const auto &type = array_field ? field_type.VectorType()
                                     : DestinationType(field_type, false);
      const auto array_cnt = array_field ? (array_count + 1) : array_count;
      if (IsStruct(type)) {
        // Generate arguments for a struct inside a struct. To ensure names
        // don't clash, and to make it obvious these arguments are constructing
        // a nested struct, prefix the name with the field name.
        GenStructArgs(*field_type.struct_def, code_ptr,
                      (nameprefix + (field.name + "_")).c_str(), array_cnt);
      } else {
        code += ", ";
        code += GenTypeBasic(type);
        if (lang_.language == IDLOptions::kJava) {
          for (size_t i = 0; i < array_cnt; i++) code += "[]";
        } else if (lang_.language == IDLOptions::kCSharp) {
          if (array_cnt > 0) {
            code += "[";
            for (size_t i = 1; i < array_cnt; i++) code += ",";
            code += "]";
          }
        } else {
          FLATBUFFERS_ASSERT(0);
        }
        code += " ";
        code += nameprefix;
        code += MakeCamel(field.name, lang_.first_camel_upper);
      }
    }
  }

  // Recusively generate struct construction statements of the form:
  // builder.putType(name);
  // and insert manual padding.
  void GenStructBody(const StructDef &struct_def, std::string *code_ptr,
                     const char *nameprefix, size_t index = 0,
                     bool in_array = false) const {
    std::string &code = *code_ptr;
    std::string indent((index + 1) * 2, ' ');
    code += indent + "  builder." + FunctionStart('P') + "rep(";
    code += NumToString(struct_def.minalign) + ", ";
    code += NumToString(struct_def.bytesize) + ");\n";
    for (auto it = struct_def.fields.vec.rbegin();
         it != struct_def.fields.vec.rend(); ++it) {
      auto &field = **it;
      const auto &field_type = field.value.type;
      if (field.padding) {
        code += indent + "  builder." + FunctionStart('P') + "ad(";
        code += NumToString(field.padding) + ");\n";
      }
      if (IsStruct(field_type)) {
        GenStructBody(*field_type.struct_def, code_ptr,
                      (nameprefix + (field.name + "_")).c_str(), index,
                      in_array);
      } else {
        const auto &type =
            IsArray(field_type) ? field_type.VectorType() : field_type;
        const auto index_var = "_idx" + NumToString(index);
        if (IsArray(field_type)) {
          code += indent + "  for (int " + index_var + " = ";
          code += NumToString(field_type.fixed_length);
          code += "; " + index_var + " > 0; " + index_var + "--) {\n";
          in_array = true;
        }
        if (IsStruct(type)) {
          GenStructBody(*field_type.struct_def, code_ptr,
                        (nameprefix + (field.name + "_")).c_str(), index + 1,
                        in_array);
        } else {
          code += IsArray(field_type) ? "  " : "";
          code += indent + "  builder." + FunctionStart('P') + "ut";
          code += GenMethod(type) + "(";
          code += SourceCast(type);
          auto argname =
              nameprefix + MakeCamel(field.name, lang_.first_camel_upper);
          code += argname;
          size_t array_cnt = index + (IsArray(field_type) ? 1 : 0);
          if (lang_.language == IDLOptions::kJava) {
            for (size_t i = 0; in_array && i < array_cnt; i++) {
              code += "[_idx" + NumToString(i) + "-1]";
            }
          } else if (lang_.language == IDLOptions::kCSharp) {
            if (array_cnt > 0) {
              code += "[";
              for (size_t i = 0; in_array && i < array_cnt; i++) {
                code += "_idx" + NumToString(i) + "-1";
                if (i != (array_cnt - 1)) code += ",";
              }
              code += "]";
            }
          } else {
            FLATBUFFERS_ASSERT(0);
          }
          code += ");\n";
        }
        if (IsArray(field_type)) { code += indent + "  }\n"; }
      }
    }
  }

  std::string GenByteBufferLength(const char *bb_name) const {
    std::string bb_len = bb_name;
    if (lang_.language == IDLOptions::kCSharp)
      bb_len += ".Length";
    else
      bb_len += ".capacity()";
    return bb_len;
  }

  std::string GenOffsetGetter(flatbuffers::FieldDef *key_field,
                              const char *num = nullptr) const {
    std::string key_offset = "";
    key_offset += lang_.accessor_prefix_static + "__offset(" +
                  NumToString(key_field->value.offset) + ", ";
    if (num) {
      key_offset += num;
      key_offset +=
          (lang_.language == IDLOptions::kCSharp ? ".Value, builder.DataBuffer)"
                                                 : ", _bb)");
    } else {
      key_offset += GenByteBufferLength("bb");
      key_offset += " - tableOffset, bb)";
    }
    return key_offset;
  }

  std::string GenLookupKeyGetter(flatbuffers::FieldDef *key_field) const {
    std::string key_getter = "      ";
    key_getter += "int tableOffset = " + lang_.accessor_prefix_static;
    key_getter += "__indirect(vectorLocation + 4 * (start + middle)";
    key_getter += ", bb);\n      ";
    if (key_field->value.type.base_type == BASE_TYPE_STRING) {
      key_getter += "int comp = " + lang_.accessor_prefix_static;
      key_getter += FunctionStart('C') + "ompareStrings(";
      key_getter += GenOffsetGetter(key_field);
      key_getter += ", byteKey, bb);\n";
    } else {
      auto get_val = GenGetterForLookupByKey(key_field, "bb");
      if (lang_.language == IDLOptions::kCSharp) {
        key_getter += "int comp = " + get_val + ".CompareTo(key);\n";
      } else {
        key_getter += GenTypeNameDest(key_field->value.type) + " val = ";
        key_getter += get_val + ";\n";
        key_getter += "      int comp = val > key ? 1 : val < key ? -1 : 0;\n";
      }
    }
    return key_getter;
  }

  std::string GenKeyGetter(flatbuffers::FieldDef *key_field) const {
    std::string key_getter = "";
    auto data_buffer =
        (lang_.language == IDLOptions::kCSharp) ? "builder.DataBuffer" : "_bb";
    if (key_field->value.type.base_type == BASE_TYPE_STRING) {
      if (lang_.language == IDLOptions::kJava) key_getter += " return ";
      key_getter += lang_.accessor_prefix_static;
      key_getter += FunctionStart('C') + "ompareStrings(";
      key_getter += GenOffsetGetter(key_field, "o1") + ", ";
      key_getter += GenOffsetGetter(key_field, "o2") + ", " + data_buffer + ")";
      if (lang_.language == IDLOptions::kJava) key_getter += ";";
    } else {
      auto field_getter = GenGetterForLookupByKey(key_field, data_buffer, "o1");
      if (lang_.language == IDLOptions::kCSharp) {
        key_getter += field_getter;
        field_getter = GenGetterForLookupByKey(key_field, data_buffer, "o2");
        key_getter += ".CompareTo(" + field_getter + ")";
      } else {
        key_getter +=
            "\n    " + GenTypeNameDest(key_field->value.type) + " val_1 = ";
        key_getter +=
            field_getter + ";\n    " + GenTypeNameDest(key_field->value.type);
        key_getter += " val_2 = ";
        field_getter = GenGetterForLookupByKey(key_field, data_buffer, "o2");
        key_getter += field_getter + ";\n";
        key_getter +=
            "    return val_1 > val_2 ? 1 : val_1 < val_2 ? -1 : 0;\n ";
      }
    }
    return key_getter;
  }

  void GenStruct(StructDef &struct_def, std::string *code_ptr) const {
    if (struct_def.generated) return;
    std::string &code = *code_ptr;

    // Generate a struct accessor class, with methods of the form:
    // public type name() { return bb.getType(i + offset); }
    // or for tables of the form:
    // public type name() {
    //   int o = __offset(offset); return o != 0 ? bb.getType(o + i) : default;
    // }
    GenComment(struct_def.doc_comment, code_ptr, &lang_.comment_config);
    if (struct_def.attributes.Lookup("private")) {
      // For Java, we leave the struct unmarked to indicate package-private
      // For C# we mark the struct as internal
      if (lang_.language == IDLOptions::kCSharp) {
        code += "internal ";
      }
    } else {
      code += "public ";
    }
    if (lang_.language == IDLOptions::kCSharp &&
        struct_def.attributes.Lookup("csharp_partial")) {
      // generate a partial class for this C# struct/table
      code += "partial ";
    } else {
      code += lang_.unsubclassable_decl;
    }
    code += lang_.accessor_type + struct_def.name;
    if (lang_.language == IDLOptions::kCSharp) {
      code += " : IFlatbufferObject";
      code += lang_.open_curly;
      code += "  private ";
      code += struct_def.fixed ? "Struct" : "Table";
      code += " __p;\n";

      if (lang_.language == IDLOptions::kCSharp) {
        code += "  public ByteBuffer ByteBuffer { get { return __p.bb; } }\n";
      }

    } else {
      code += lang_.inheritance_marker;
      code += struct_def.fixed ? "Struct" : "Table";
      code += lang_.open_curly;
    }

    if (!struct_def.fixed) {
      // Generate verson check method.
      // Force compile time error if not using the same version runtime.
      code += "  public static void ValidateVersion() {";
      if (lang_.language == IDLOptions::kCSharp)
        code += " FlatBufferConstants.";
      else
        code += " Constants.";
      code += "FLATBUFFERS_1_11_1(); ";
      code += "}\n";

      // Generate a special accessor for the table that when used as the root
      // of a FlatBuffer
      std::string method_name =
          FunctionStart('G') + "etRootAs" + struct_def.name;
      std::string method_signature =
          "  public static " + struct_def.name + " " + method_name;

      // create convenience method that doesn't require an existing object
      code += method_signature + "(ByteBuffer _bb) ";
      code += "{ return " + method_name + "(_bb, new " + struct_def.name +
              "()); }\n";

      // create method that allows object reuse
      code +=
          method_signature + "(ByteBuffer _bb, " + struct_def.name + " obj) { ";
      code += lang_.set_bb_byteorder;
      code += "return (obj.__assign(_bb." + FunctionStart('G') + "etInt(_bb.";
      code += lang_.get_bb_position;
      code += ") + _bb.";
      code += lang_.get_bb_position;
      code += ", _bb)); }\n";
      if (parser_.root_struct_def_ == &struct_def) {
        if (parser_.file_identifier_.length()) {
          // Check if a buffer has the identifier.
          code += "  public static ";
          code += lang_.bool_type + struct_def.name;
          code += "BufferHasIdentifier(ByteBuffer _bb) { return ";
          code += lang_.accessor_prefix_static + "__has_identifier(_bb, \"";
          code += parser_.file_identifier_;
          code += "\"); }\n";
        }
      }
    }
    // Generate the __init method that sets the field in a pre-existing
    // accessor object. This is to allow object reuse.
    code += "  public void __init(int _i, ByteBuffer _bb) ";
    code += "{ ";
    if (lang_.language == IDLOptions::kCSharp) {
      code += "__p = new ";
      code += struct_def.fixed ? "Struct" : "Table";
      code += "(_i, _bb); ";
    } else {
      code += "__reset(_i, _bb); ";
    }
    code += "}\n";
    code +=
        "  public " + struct_def.name + " __assign(int _i, ByteBuffer _bb) ";
    code += "{ __init(_i, _bb); return this; }\n\n";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      GenComment(field.doc_comment, code_ptr, &lang_.comment_config, "  ");
      std::string type_name = GenTypeGet(field.value.type);
      std::string type_name_dest = GenTypeNameDest(field.value.type);
      std::string conditional_cast = "";
      std::string optional = "";
      if (lang_.language == IDLOptions::kCSharp && !struct_def.fixed &&
          (field.value.type.base_type == BASE_TYPE_STRUCT ||
           field.value.type.base_type == BASE_TYPE_UNION ||
           (field.value.type.base_type == BASE_TYPE_VECTOR &&
            (field.value.type.element == BASE_TYPE_STRUCT ||
             field.value.type.element == BASE_TYPE_UNION)))) {
        optional = lang_.optional_suffix;
        conditional_cast = "(" + type_name_dest + optional + ")";
      }
      std::string dest_mask = DestinationMask(field.value.type, true);
      std::string dest_cast = DestinationCast(field.value.type);
      std::string src_cast = SourceCast(field.value.type);
      std::string method_start =
          "  public " +
          (field.required ? "" : GenNullableAnnotation(field.value.type)) +
          GenPureAnnotation(field.value.type) + type_name_dest + optional +
          " " + MakeCamel(field.name, lang_.first_camel_upper);
      std::string obj = lang_.language == IDLOptions::kCSharp
                            ? "(new " + type_name + "())"
                            : "obj";

      // Most field accessors need to retrieve and test the field offset first,
      // this is the prefix code for that:
      auto offset_prefix =
          IsArray(field.value.type)
              ? " { return "
              : (" { int o = " + lang_.accessor_prefix + "__offset(" +
                 NumToString(field.value.offset) + "); return o != 0 ? ");
      // Generate the accessors that don't do object reuse.
      if (field.value.type.base_type == BASE_TYPE_STRUCT) {
        // Calls the accessor that takes an accessor object with a new object.
        if (lang_.language != IDLOptions::kCSharp) {
          code += method_start + "() { return ";
          code += MakeCamel(field.name, lang_.first_camel_upper);
          code += "(new ";
          code += type_name + "()); }\n";
        }
      } else if (field.value.type.base_type == BASE_TYPE_VECTOR &&
                 field.value.type.element == BASE_TYPE_STRUCT) {
        // Accessors for vectors of structs also take accessor objects, this
        // generates a variant without that argument.
        if (lang_.language != IDLOptions::kCSharp) {
          code += method_start + "(int j) { return ";
          code += MakeCamel(field.name, lang_.first_camel_upper);
          code += "(new " + type_name + "(), j); }\n";
        }
      } else if (field.value.type.base_type == BASE_TYPE_UNION ||
          (field.value.type.base_type == BASE_TYPE_VECTOR &&
           field.value.type.VectorType().base_type == BASE_TYPE_UNION)) {
        if (lang_.language == IDLOptions::kCSharp) {
          // Union types in C# use generic Table-derived type for better type
          // safety.
          method_start += "<TTable>";
          type_name = type_name_dest;
        }
      }
      std::string getter = dest_cast + GenGetter(field.value.type);
      code += method_start;
      std::string default_cast = "";
      // only create default casts for c# scalars or vectors of scalars
      if (lang_.language == IDLOptions::kCSharp &&
          (IsScalar(field.value.type.base_type) ||
           (field.value.type.base_type == BASE_TYPE_VECTOR &&
            IsScalar(field.value.type.element)))) {
        // For scalars, default value will be returned by GetDefaultValue().
        // If the scalar is an enum, GetDefaultValue() returns an actual c# enum
        // that doesn't need to be casted. However, default values for enum
        // elements of vectors are integer literals ("0") and are still casted
        // for clarity.
        if (field.value.type.enum_def == nullptr ||
            field.value.type.base_type == BASE_TYPE_VECTOR) {
          default_cast = "(" + type_name_dest + ")";
        }
      }
      std::string member_suffix = "; ";
      if (IsScalar(field.value.type.base_type)) {
        code += lang_.getter_prefix;
        member_suffix += lang_.getter_suffix;
        if (struct_def.fixed) {
          code += " { return " + getter;
          code += "(" + lang_.accessor_prefix + "bb_pos + ";
          code += NumToString(field.value.offset) + ")";
          code += dest_mask;
        } else {
          code += offset_prefix + getter;
          code += "(o + " + lang_.accessor_prefix + "bb_pos)" + dest_mask;
          code += " : " + default_cast;
          code += GenDefaultValue(field);
        }
      } else {
        switch (field.value.type.base_type) {
          case BASE_TYPE_STRUCT:
            if (lang_.language != IDLOptions::kCSharp) {
              code += "(" + type_name + " obj" + ")";
            } else {
              code += lang_.getter_prefix;
              member_suffix += lang_.getter_suffix;
            }
            if (struct_def.fixed) {
              code += " { return " + obj + ".__assign(" + lang_.accessor_prefix;
              code += "bb_pos + " + NumToString(field.value.offset) + ", ";
              code += lang_.accessor_prefix + "bb)";
            } else {
              code += offset_prefix + conditional_cast;
              code += obj + ".__assign(";
              code += field.value.type.struct_def->fixed
                          ? "o + " + lang_.accessor_prefix + "bb_pos"
                          : lang_.accessor_prefix + "__indirect(o + " +
                                lang_.accessor_prefix + "bb_pos)";
              code += ", " + lang_.accessor_prefix + "bb) : null";
            }
            break;
          case BASE_TYPE_STRING:
            code += lang_.getter_prefix;
            member_suffix += lang_.getter_suffix;
            code += offset_prefix + getter + "(o + " + lang_.accessor_prefix;
            code += "bb_pos) : null";
            break;
          case BASE_TYPE_ARRAY: FLATBUFFERS_FALLTHROUGH();  // fall thru
          case BASE_TYPE_VECTOR: {
            auto vectortype = field.value.type.VectorType();
            if (vectortype.base_type == BASE_TYPE_UNION &&
                lang_.language == IDLOptions::kCSharp) {
                  conditional_cast = "(TTable?)";
                  getter += "<TTable>";
            }
            code += "(";
            if (vectortype.base_type == BASE_TYPE_STRUCT) {
              if (lang_.language != IDLOptions::kCSharp)
                code += type_name + " obj, ";
              getter = obj + ".__assign";
            } else if (vectortype.base_type == BASE_TYPE_UNION) {
              if (lang_.language != IDLOptions::kCSharp)
                code += type_name + " obj, ";
            }
            code += "int j)";
            const auto body = offset_prefix + conditional_cast + getter + "(";
            if (vectortype.base_type == BASE_TYPE_UNION) {
              if (lang_.language != IDLOptions::kCSharp)
                code += body + "obj, ";
              else
                code += " where TTable : struct, IFlatbufferObject" + body;
            } else {
              code += body;
            }
            auto index = lang_.accessor_prefix;
            if (IsArray(field.value.type)) {
              index += "bb_pos + " + NumToString(field.value.offset) + " + ";
            } else {
              index += "__vector(o) + ";
            }
            index += "j * " + NumToString(InlineSize(vectortype));
            if (vectortype.base_type == BASE_TYPE_STRUCT) {
              code += vectortype.struct_def->fixed
                          ? index
                          : lang_.accessor_prefix + "__indirect(" + index + ")";
              code += ", " + lang_.accessor_prefix + "bb";
            } else if (vectortype.base_type == BASE_TYPE_UNION) {
              code += index + " - " + lang_.accessor_prefix +  "bb_pos";
            } else {
              code += index;
            }
            code += ")" + dest_mask;
            if (!IsArray(field.value.type)) {
              code += " : ";
              code +=
                  field.value.type.element == BASE_TYPE_BOOL
                      ? "false"
                      : (IsScalar(field.value.type.element) ? default_cast + "0"
                                                            : "null");
            }

            break;
          }
          case BASE_TYPE_UNION:
            if (lang_.language == IDLOptions::kCSharp) {
              code += "() where TTable : struct, IFlatbufferObject";
              code += offset_prefix + "(TTable?)" + getter;
              code += "<TTable>(o) : null";
            } else {
              code += "(" + type_name + " obj)" + offset_prefix + getter;
              code += "(obj, o) : null";
            }
            break;
          default: FLATBUFFERS_ASSERT(0);
        }
      }
      code += member_suffix;
      code += "}\n";
      if (field.value.type.base_type == BASE_TYPE_VECTOR) {
        code +=
            "  public int " + MakeCamel(field.name, lang_.first_camel_upper);
        code += "Length";
        code += lang_.getter_prefix;
        code += offset_prefix;
        code += lang_.accessor_prefix + "__vector_len(o) : 0; ";
        code += lang_.getter_suffix;
        code += "}\n";
        // See if we should generate a by-key accessor.
        if (field.value.type.element == BASE_TYPE_STRUCT &&
            !field.value.type.struct_def->fixed) {
          auto &sd = *field.value.type.struct_def;
          auto &fields = sd.fields.vec;
          for (auto kit = fields.begin(); kit != fields.end(); ++kit) {
            auto &key_field = **kit;
            if (key_field.key) {
              auto qualified_name = WrapInNameSpace(sd);
              code += "  public " + qualified_name + lang_.optional_suffix + " ";
              code += MakeCamel(field.name, lang_.first_camel_upper) + "ByKey(";
              code += GenTypeNameDest(key_field.value.type) + " key)";
              code += offset_prefix;
              code += qualified_name + ".__lookup_by_key(";
              if (lang_.language == IDLOptions::kJava)
                code += "null, ";
              code += lang_.accessor_prefix + "__vector(o), key, ";
              code += lang_.accessor_prefix + "bb) : null; ";
              code += "}\n";
              if (lang_.language == IDLOptions::kJava) {
                code += "  public " + qualified_name + lang_.optional_suffix + " ";
                code += MakeCamel(field.name, lang_.first_camel_upper) + "ByKey(";
                code += qualified_name + lang_.optional_suffix + " obj, ";
                code += GenTypeNameDest(key_field.value.type) + " key)";
                code += offset_prefix;
                code += qualified_name + ".__lookup_by_key(obj, ";
                code += lang_.accessor_prefix + "__vector(o), key, ";
                code += lang_.accessor_prefix + "bb) : null; ";
                code += "}\n";
              }
              break;
            }
          }
        }
      }
      // Generate a ByteBuffer accessor for strings & vectors of scalars.
      if ((field.value.type.base_type == BASE_TYPE_VECTOR &&
           IsScalar(field.value.type.VectorType().base_type)) ||
          field.value.type.base_type == BASE_TYPE_STRING) {
        switch (lang_.language) {
          case IDLOptions::kJava:
            code += "  public ByteBuffer ";
            code += MakeCamel(field.name, lang_.first_camel_upper);
            code += "AsByteBuffer() { return ";
            code += lang_.accessor_prefix + "__vector_as_bytebuffer(";
            code += NumToString(field.value.offset) + ", ";
            code +=
                NumToString(field.value.type.base_type == BASE_TYPE_STRING
                                ? 1
                                : InlineSize(field.value.type.VectorType()));
            code += "); }\n";
            code += "  public ByteBuffer ";
            code += MakeCamel(field.name, lang_.first_camel_upper);
            code += "InByteBuffer(ByteBuffer _bb) { return ";
            code += lang_.accessor_prefix + "__vector_in_bytebuffer(_bb, ";
            code += NumToString(field.value.offset) + ", ";
            code +=
                NumToString(field.value.type.base_type == BASE_TYPE_STRING
                                ? 1
                                : InlineSize(field.value.type.VectorType()));
            code += "); }\n";
            break;
          case IDLOptions::kCSharp:
            code += "#if ENABLE_SPAN_T\n";
            code += "  public Span<" + GenTypeBasic(field.value.type.VectorType()) + "> Get";
            code += MakeCamel(field.name, lang_.first_camel_upper);
            code += "Bytes() { return ";
            code += lang_.accessor_prefix + "__vector_as_span<"+ GenTypeBasic(field.value.type.VectorType()) +">(";
            code += NumToString(field.value.offset);
            code += ", " + NumToString(SizeOf(field.value.type.VectorType().base_type));
            code += "); }\n";
            code += "#else\n";
            code += "  public ArraySegment<byte>? Get";
            code += MakeCamel(field.name, lang_.first_camel_upper);
            code += "Bytes() { return ";
            code += lang_.accessor_prefix + "__vector_as_arraysegment(";
            code += NumToString(field.value.offset);           
            code += "); }\n";
            code += "#endif\n";

            // For direct blockcopying the data into a typed array
            code += "  public ";
            code += GenTypeBasic(field.value.type.VectorType());
            code += "[] Get";
            code += MakeCamel(field.name, lang_.first_camel_upper);
            code += "Array() { ";
            if (IsEnum(field.value.type.VectorType())) {
              // Since __vector_as_array does not work for enum types,
              // fill array using an explicit loop.
              code += "int o = " + lang_.accessor_prefix + "__offset(";
              code += NumToString(field.value.offset);
              code += "); if (o == 0) return null; int p = ";
              code += lang_.accessor_prefix + "__vector(o); int l = ";
              code += lang_.accessor_prefix + "__vector_len(o); ";
              code += GenTypeBasic(field.value.type.VectorType());
              code += "[] a = new ";
              code += GenTypeBasic(field.value.type.VectorType());
              code += "[l]; for (int i = 0; i < l; i++) { a[i] = " + getter;
              code += "(p + i * ";
              code += NumToString(InlineSize(field.value.type.VectorType()));
              code += "); } return a;";
            } else {
              code += "return ";
              code += lang_.accessor_prefix + "__vector_as_array<";
              code += GenTypeBasic(field.value.type.VectorType());
              code += ">(";
              code += NumToString(field.value.offset);
              code += ");";
            }
            code += " }\n";
            break;
          default: break;
        }
      }
      // generate object accessors if is nested_flatbuffer
      if (field.nested_flatbuffer) {
        auto nested_type_name = WrapInNameSpace(*field.nested_flatbuffer);
        auto nested_method_name =
            MakeCamel(field.name, lang_.first_camel_upper) + "As" +
            field.nested_flatbuffer->name;
        auto get_nested_method_name = nested_method_name;
        if (lang_.language == IDLOptions::kCSharp) {
          get_nested_method_name = "Get" + nested_method_name;
          conditional_cast =
              "(" + nested_type_name + lang_.optional_suffix + ")";
        }
        if (lang_.language != IDLOptions::kCSharp) {
          code += "  public " + nested_type_name + lang_.optional_suffix + " ";
          code += nested_method_name + "() { return ";
          code +=
              get_nested_method_name + "(new " + nested_type_name + "()); }\n";
        } else {
          obj = "(new " + nested_type_name + "())";
        }
        code += "  public " + nested_type_name + lang_.optional_suffix + " ";
        code += get_nested_method_name + "(";
        if (lang_.language != IDLOptions::kCSharp)
          code += nested_type_name + " obj";
        code += ") { int o = " + lang_.accessor_prefix + "__offset(";
        code += NumToString(field.value.offset) + "); ";
        code += "return o != 0 ? " + conditional_cast + obj + ".__assign(";
        code += lang_.accessor_prefix;
        code += "__indirect(" + lang_.accessor_prefix + "__vector(o)), ";
        code += lang_.accessor_prefix + "bb) : null; }\n";
      }
      // Generate mutators for scalar fields or vectors of scalars.
      if (parser_.opts.mutable_buffer) {
        auto is_series = (IsSeries(field.value.type));
        const auto &underlying_type =
            is_series ? field.value.type.VectorType() : field.value.type;
        // Boolean parameters have to be explicitly converted to byte
        // representation.
        auto setter_parameter = underlying_type.base_type == BASE_TYPE_BOOL
                                    ? "(byte)(" + field.name + " ? 1 : 0)"
                                    : field.name;
        auto mutator_prefix = MakeCamel("mutate", lang_.first_camel_upper);
        // A vector mutator also needs the index of the vector element it should
        // mutate.
        auto mutator_params = (is_series ? "(int j, " : "(") +
                              GenTypeNameDest(underlying_type) + " " +
                              field.name + ") { ";
        auto setter_index =
            is_series
                ? lang_.accessor_prefix +
                      (IsArray(field.value.type)
                           ? "bb_pos + " + NumToString(field.value.offset)
                           : "__vector(o)") +
                      +" + j * " + NumToString(InlineSize(underlying_type))
                : (struct_def.fixed
                       ? lang_.accessor_prefix + "bb_pos + " +
                             NumToString(field.value.offset)
                       : "o + " + lang_.accessor_prefix + "bb_pos");
        if (IsScalar(underlying_type.base_type)) {
          code += "  public ";
          code += struct_def.fixed ? "void " : lang_.bool_type;
          code += mutator_prefix + MakeCamel(field.name, true);
          code += mutator_params;
          if (struct_def.fixed) {
            code += GenSetter(underlying_type) + "(" + setter_index + ", ";
            code += src_cast + setter_parameter + "); }\n";
          } else {
            code += "int o = " + lang_.accessor_prefix + "__offset(";
            code += NumToString(field.value.offset) + ");";
            code += " if (o != 0) { " + GenSetter(underlying_type);
            code += "(" + setter_index + ", " + src_cast + setter_parameter +
                    "); return true; } else { return false; } }\n";
          }
        }
      }
      if (parser_.opts.java_primitive_has_method &&
          IsScalar(field.value.type.base_type) && !struct_def.fixed) {
        auto vt_offset_constant = "  public static final int VT_" +
                                  MakeScreamingCamel(field.name) + " = " +
                                  NumToString(field.value.offset) + ";";

        code += vt_offset_constant;
        code += "\n";
      }
    }
    code += "\n";
    flatbuffers::FieldDef *key_field = nullptr;
    if (struct_def.fixed) {
      // create a struct constructor function
      code += "  public static " + GenOffsetType(struct_def) + " ";
      code += FunctionStart('C') + "reate";
      code += struct_def.name + "(FlatBufferBuilder builder";
      GenStructArgs(struct_def, code_ptr, "");
      code += ") {\n";
      GenStructBody(struct_def, code_ptr, "");
      code += "    return ";
      code += GenOffsetConstruct(
          struct_def, "builder." + std::string(lang_.get_fbb_offset));
      code += ";\n  }\n";
    } else {
      // Generate a method that creates a table in one go. This is only possible
      // when the table has no struct fields, since those have to be created
      // inline, and there's no way to do so in Java.
      bool has_no_struct_fields = true;
      int num_fields = 0;
      for (auto it = struct_def.fields.vec.begin();
           it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (field.deprecated) continue;
        if (IsStruct(field.value.type)) {
          has_no_struct_fields = false;
        } else {
          num_fields++;
        }
      }
      // JVM specifications restrict default constructor params to be < 255.
      // Longs and doubles take up 2 units, so we set the limit to be < 127.
      if (has_no_struct_fields && num_fields && num_fields < 127) {
        // Generate a table constructor of the form:
        // public static int createName(FlatBufferBuilder builder, args...)
        code += "  public static " + GenOffsetType(struct_def) + " ";
        code += FunctionStart('C') + "reate" + struct_def.name;
        code += "(FlatBufferBuilder builder";
        for (auto it = struct_def.fields.vec.begin();
             it != struct_def.fields.vec.end(); ++it) {
          auto &field = **it;
          if (field.deprecated) continue;
          code += ",\n      ";
          code += GenTypeBasic(DestinationType(field.value.type, false));
          code += " ";
          code += field.name;
          if (!IsScalar(field.value.type.base_type)) code += "Offset";

          // Java doesn't have defaults, which means this method must always
          // supply all arguments, and thus won't compile when fields are added.
          if (lang_.language != IDLOptions::kJava) {
            code += " = ";
            code += GenDefaultValueBasic(field);
          }
        }
        code += ") {\n    builder.";
        code += FunctionStart('S') + "tartTable(";
        code += NumToString(struct_def.fields.vec.size()) + ");\n";
        for (size_t size = struct_def.sortbysize ? sizeof(largest_scalar_t) : 1;
             size; size /= 2) {
          for (auto it = struct_def.fields.vec.rbegin();
               it != struct_def.fields.vec.rend(); ++it) {
            auto &field = **it;
            if (!field.deprecated &&
                (!struct_def.sortbysize ||
                 size == SizeOf(field.value.type.base_type))) {
              code += "    " + struct_def.name + ".";
              code += FunctionStart('A') + "dd";
              code += MakeCamel(field.name) + "(builder, " + field.name;
              if (!IsScalar(field.value.type.base_type)) code += "Offset";
              code += ");\n";
            }
          }
        }
        code += "    return " + struct_def.name + ".";
        code += FunctionStart('E') + "nd" + struct_def.name;
        code += "(builder);\n  }\n\n";
      }
      // Generate a set of static methods that allow table construction,
      // of the form:
      // public static void addName(FlatBufferBuilder builder, short name)
      // { builder.addShort(id, name, default); }
      // Unlike the Create function, these always work.
      code += "  public static void " + FunctionStart('S') + "tart";
      code += struct_def.name;
      code += "(FlatBufferBuilder builder) { builder.";
      code += FunctionStart('S') + "tartTable(";
      code += NumToString(struct_def.fields.vec.size()) + "); }\n";
      for (auto it = struct_def.fields.vec.begin();
           it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (field.deprecated) continue;
        if (field.key) key_field = &field;
        code += "  public static void " + FunctionStart('A') + "dd";
        code += MakeCamel(field.name);
        code += "(FlatBufferBuilder builder, ";
        code += GenTypeBasic(DestinationType(field.value.type, false));
        auto argname = MakeCamel(field.name, false);
        if (!IsScalar(field.value.type.base_type)) argname += "Offset";
        code += " " + argname + ") { builder." + FunctionStart('A') + "dd";
        code += GenMethod(field.value.type) + "(";
        code += NumToString(it - struct_def.fields.vec.begin()) + ", ";
        code += SourceCastBasic(field.value.type);
        code += argname;
        if (!IsScalar(field.value.type.base_type) &&
            field.value.type.base_type != BASE_TYPE_UNION &&
            lang_.language == IDLOptions::kCSharp) {
          code += ".Value";
        }
        code += ", ";
        if (lang_.language == IDLOptions::kJava)
          code += SourceCastBasic(field.value.type);
        code += GenDefaultValue(field, false);
        code += "); }\n";
        if (field.value.type.base_type == BASE_TYPE_VECTOR) {
          auto vector_type = field.value.type.VectorType();
          auto alignment = InlineAlignment(vector_type);
          auto elem_size = InlineSize(vector_type);
          if (!IsStruct(vector_type)) {
            // Generate a method to create a vector from a Java array.
            code += "  public static " + GenVectorOffsetType() + " ";
            code += FunctionStart('C') + "reate";
            code += MakeCamel(field.name);
            code += "Vector(FlatBufferBuilder builder, ";
            code += GenTypeBasic(vector_type) + "[] data) ";
            code += "{ builder." + FunctionStart('S') + "tartVector(";
            code += NumToString(elem_size);
            code += ", data." + FunctionStart('L') + "ength, ";
            code += NumToString(alignment);
            code += "); for (int i = data.";
            code += FunctionStart('L') + "ength - 1; i >= 0; i--) builder.";
            code += FunctionStart('A') + "dd";
            code += GenMethod(vector_type);
            code += "(";
            code += SourceCastBasic(vector_type, false);
            code += "data[i]";
            if (lang_.language == IDLOptions::kCSharp &&
                (vector_type.base_type == BASE_TYPE_STRUCT ||
                 vector_type.base_type == BASE_TYPE_STRING))
              code += ".Value";
            code += "); return ";
            code += "builder." + FunctionStart('E') + "ndVector(); }\n";
            // For C#, include a block copy method signature.
            // Skip if the vector is of enums, because builder.Add
            // throws an exception when supplied an enum array.
            if (lang_.language == IDLOptions::kCSharp &&
                !IsEnum(vector_type)) {
              code += "  public static " + GenVectorOffsetType() + " ";
              code += FunctionStart('C') + "reate";
              code += MakeCamel(field.name);
              code += "VectorBlock(FlatBufferBuilder builder, ";
              code += GenTypeBasic(vector_type) + "[] data) ";
              code += "{ builder." + FunctionStart('S') + "tartVector(";
              code += NumToString(elem_size);
              code += ", data." + FunctionStart('L') + "ength, ";
              code += NumToString(alignment);
              code += "); builder.Add(data); return builder.EndVector(); }\n";
            }
          }
          // Generate a method to start a vector, data to be added manually
          // after.
          code += "  public static void " + FunctionStart('S') + "tart";
          code += MakeCamel(field.name);
          code += "Vector(FlatBufferBuilder builder, int numElems) ";
          code += "{ builder." + FunctionStart('S') + "tartVector(";
          code += NumToString(elem_size);
          code += ", numElems, " + NumToString(alignment);
          code += "); }\n";
        }
      }
      code += "  public static " + GenOffsetType(struct_def) + " ";
      code += FunctionStart('E') + "nd" + struct_def.name;
      code += "(FlatBufferBuilder builder) {\n    int o = builder.";
      code += FunctionStart('E') + "ndTable();\n";
      for (auto it = struct_def.fields.vec.begin();
           it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (!field.deprecated && field.required) {
          code += "    builder." + FunctionStart('R') + "equired(o, ";
          code += NumToString(field.value.offset);
          code += ");  // " + field.name + "\n";
        }
      }
      code += "    return " + GenOffsetConstruct(struct_def, "o") + ";\n  }\n";
      if (parser_.root_struct_def_ == &struct_def) {
        std::string size_prefix[] = { "", "SizePrefixed" };
        for (int i = 0; i < 2; ++i) {
          code += "  public static void ";
          code += FunctionStart('F') + "inish" + size_prefix[i] +
                  struct_def.name;
          code += "Buffer(FlatBufferBuilder builder, " +
                  GenOffsetType(struct_def);
          code += " offset) {";
          code += " builder." + FunctionStart('F') + "inish" + size_prefix[i] +
                  "(offset";
          if (lang_.language == IDLOptions::kCSharp) { code += ".Value"; }

          if (parser_.file_identifier_.length())
            code += ", \"" + parser_.file_identifier_ + "\"";
          code += "); }\n";
        }
      }
    }
    // Only generate key compare function for table,
    // because `key_field` is not set for struct
    if (struct_def.has_key && !struct_def.fixed) {
      FLATBUFFERS_ASSERT(key_field);
      if (lang_.language == IDLOptions::kJava) {
        code += "\n  @Override\n  protected int keysCompare(";
        code += "Integer o1, Integer o2, ByteBuffer _bb) {";
        code += GenKeyGetter(key_field);
        code += " }\n";
      } else {
        code += "\n  public static VectorOffset ";
        code += "CreateSortedVectorOf" + struct_def.name;
        code += "(FlatBufferBuilder builder, ";
        code += "Offset<" + struct_def.name + ">";
        code += "[] offsets) {\n";
        code += "    Array.Sort(offsets, (Offset<" + struct_def.name +
                "> o1, Offset<" + struct_def.name + "> o2) => " +
                GenKeyGetter(key_field);
        code += ");\n";
        code += "    return builder.CreateVectorOfTables(offsets);\n  }\n";
      }

      code += "\n  public static " + struct_def.name + lang_.optional_suffix;
      code += " __lookup_by_key(";
      if (lang_.language == IDLOptions::kJava)
        code +=  struct_def.name + " obj, ";
      code += "int vectorLocation, ";
      code += GenTypeNameDest(key_field->value.type);
      code += " key, ByteBuffer bb) {\n";
      if (key_field->value.type.base_type == BASE_TYPE_STRING) {
        code += "    byte[] byteKey = ";
        if (lang_.language == IDLOptions::kJava)
          code += "key.getBytes(Table.UTF8_CHARSET.get());\n";
        else
          code += "System.Text.Encoding.UTF8.GetBytes(key);\n";
      }
      code += "    int span = ";
      code += "bb." + FunctionStart('G') + "etInt(vectorLocation - 4);\n";
      code += "    int start = 0;\n";
      code += "    while (span != 0) {\n";
      code += "      int middle = span / 2;\n";
      code += GenLookupKeyGetter(key_field);
      code += "      if (comp > 0) {\n";
      code += "        span = middle;\n";
      code += "      } else if (comp < 0) {\n";
      code += "        middle++;\n";
      code += "        start += middle;\n";
      code += "        span -= middle;\n";
      code += "      } else {\n";
      code += "        return ";
      if (lang_.language == IDLOptions::kJava)
        code += "(obj == null ? new " + struct_def.name + "() : obj)";
      else
        code += "new " + struct_def.name + "()";
      code += ".__assign(tableOffset, bb);\n";
      code += "      }\n    }\n";
      code += "    return null;\n";
      code += "  }\n";
    }
    code += "}";
    // Java does not need the closing semi-colon on class definitions.
    code += (lang_.language != IDLOptions::kJava) ? ";" : "";
    code += "\n\n";
  }
  const LanguageParameters &lang_;
  // This tracks the current namespace used to determine if a type need to be
  // prefixed by its namespace
  const Namespace *cur_name_space_;
};
}  // namespace general

bool GenerateGeneral(const Parser &parser, const std::string &path,
                     const std::string &file_name) {
  general::GeneralGenerator generator(parser, path, file_name);
  return generator.generate();
}

std::string GeneralMakeRule(const Parser &parser, const std::string &path,
                            const std::string &file_name) {
  FLATBUFFERS_ASSERT(parser.opts.lang <= IDLOptions::kMAX);
  const auto &lang = GetLangParams(parser.opts.lang);

  std::string make_rule;

  for (auto it = parser.enums_.vec.begin(); it != parser.enums_.vec.end();
       ++it) {
    auto &enum_def = **it;
    if (!make_rule.empty()) make_rule += " ";
    std::string directory =
        BaseGenerator::NamespaceDir(parser, path, *enum_def.defined_namespace);
    make_rule += directory + enum_def.name + lang.file_extension;
  }

  for (auto it = parser.structs_.vec.begin(); it != parser.structs_.vec.end();
       ++it) {
    auto &struct_def = **it;
    if (!make_rule.empty()) make_rule += " ";
    std::string directory = BaseGenerator::NamespaceDir(
        parser, path, *struct_def.defined_namespace);
    make_rule += directory + struct_def.name + lang.file_extension;
  }

  make_rule += ": ";
  auto included_files = parser.GetIncludedFilesRecursive(file_name);
  for (auto it = included_files.begin(); it != included_files.end(); ++it) {
    make_rule += " " + *it;
  }
  return make_rule;
}

std::string BinaryFileName(const Parser &parser, const std::string &path,
                           const std::string &file_name) {
  auto ext = parser.file_extension_.length() ? parser.file_extension_ : "bin";
  return path + file_name + "." + ext;
}

bool GenerateBinary(const Parser &parser, const std::string &path,
                    const std::string &file_name) {
  return !parser.builder_.GetSize() ||
         flatbuffers::SaveFile(
             BinaryFileName(parser, path, file_name).c_str(),
             reinterpret_cast<char *>(parser.builder_.GetBufferPointer()),
             parser.builder_.GetSize(), true);
}

std::string BinaryMakeRule(const Parser &parser, const std::string &path,
                           const std::string &file_name) {
  if (!parser.builder_.GetSize()) return "";
  std::string filebase =
      flatbuffers::StripPath(flatbuffers::StripExtension(file_name));
  std::string make_rule =
      BinaryFileName(parser, path, filebase) + ": " + file_name;
  auto included_files =
      parser.GetIncludedFilesRecursive(parser.root_struct_def_->file);
  for (auto it = included_files.begin(); it != included_files.end(); ++it) {
    make_rule += " " + *it;
  }
  return make_rule;
}

}  // namespace flatbuffers
