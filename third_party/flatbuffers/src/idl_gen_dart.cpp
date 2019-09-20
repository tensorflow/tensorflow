/*
 * Copyright 2018 Dan Field
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
#include <cassert>

#include "flatbuffers/code_generators.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

namespace flatbuffers {

static std::string GeneratedFileName(const std::string &path,
                                     const std::string &file_name) {
  return path + file_name + "_generated.dart";
}

namespace dart {

const std::string _kFb = "fb";
// see https://www.dartlang.org/guides/language/language-tour#keywords
// yeild*, async*, and sync* shouldn't be problems anyway but keeping them in
static const char *keywords[] = {
  "abstract",   "deferred", "if",       "super",   "as",       "do",
  "implements", "switch",   "assert",   "dynamic", "import",   "sync*",
  "async",      "else",     "in",       "this",    "async*",   "enum",
  "is",         "throw",    "await",    "export",  "library",  "true",
  "break",      "external", "new",      "try",     "case",     "extends",
  "null",       "typedef",  "catch",    "factory", "operator", "var",
  "class",      "false",    "part",     "void",    "const",    "final",
  "rethrow",    "while",    "continue", "finally", "return",   "with",
  "covariant",  "for",      "set",      "yield",   "default",  "get",
  "static",     "yield*"
};

// Iterate through all definitions we haven't generate code for (enums, structs,
// and tables) and output them to a single file.
class DartGenerator : public BaseGenerator {
 public:
  typedef std::map<std::string, std::string> namespace_code_map;

  DartGenerator(const Parser &parser, const std::string &path,
                const std::string &file_name)
      : BaseGenerator(parser, path, file_name, "", ".") {}
  // Iterate through all definitions we haven't generate code for (enums,
  // structs, and tables) and output them to a single file.
  bool generate() {
    std::string code;
    namespace_code_map namespace_code;
    GenerateEnums(&namespace_code);
    GenerateStructs(&namespace_code);

    for (auto kv = namespace_code.begin(); kv != namespace_code.end(); ++kv) {
      code.clear();
      code = code + "// " + FlatBuffersGeneratedWarning() + "\n";
      code = code +
             "// ignore_for_file: unused_import, unused_field, "
             "unused_local_variable\n\n";

      if (!kv->first.empty()) {
        code += "library " + kv->first + ";\n\n";
      }

      code += "import 'dart:typed_data' show Uint8List;\n";
      code += "import 'package:flat_buffers/flat_buffers.dart' as " + _kFb +
              ";\n\n";

      if (parser_.opts.include_dependence_headers) {
        GenIncludeDependencies(&code, kv->first);
      }

      for (auto kv2 = namespace_code.begin(); kv2 != namespace_code.end();
           ++kv2) {
        if (kv2->first != kv->first) {
          code += "import '" +
                  GeneratedFileName("./", file_name_ + (!kv2->first.empty() ? "_" + kv2->first : "")) +
                  "' as " + ImportAliasName(kv2->first) + ";\n";
        }
      }
      code += "\n";
      code += kv->second;

      if (!SaveFile(
              GeneratedFileName(path_, file_name_ + (!kv->first.empty() ? "_" + kv->first : "")).c_str(),
              code, false)) {
        return false;
      }
    }
    return true;
  }

 private:
  static std::string ImportAliasName(const std::string &ns) {
    std::string ret;
    ret.assign(ns);
    size_t pos = ret.find('.');
    while (pos != std::string::npos) {
      ret.replace(pos, 1, "_");
      pos = ret.find('.', pos + 1);
    }

    return ret;
  }

  static std::string BuildNamespaceName(const Namespace &ns) {
    if (ns.components.empty()) {
      return "";
    }
    std::stringstream sstream;
    std::copy(ns.components.begin(), ns.components.end() - 1,
              std::ostream_iterator<std::string>(sstream, "."));

    auto ret = sstream.str() + ns.components.back();
    for (size_t i = 0; i < ret.size(); i++) {
      auto lower = tolower(ret[i]);
      if (lower != ret[i]) {
        ret[i] = static_cast<char>(lower);
        if (i != 0 && ret[i - 1] != '.') {
          ret.insert(i, "_");
          i++;
        }
      }
    }
    // std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
    return ret;
  }

  void GenIncludeDependencies(std::string* code, const std::string& the_namespace) {
    for (auto it = parser_.included_files_.begin();
         it != parser_.included_files_.end(); ++it) {
      if (it->second.empty()) continue;

      auto noext = flatbuffers::StripExtension(it->second);
      auto basename = flatbuffers::StripPath(noext);

      *code += "import '" + GeneratedFileName("", basename + (the_namespace == "" ? "" : "_" + the_namespace)) + "';\n";
    }
  }

  static std::string EscapeKeyword(const std::string &name) {
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
      if (name == keywords[i]) { return MakeCamel(name + "_", false); }
    }

    return MakeCamel(name, false);
  }

  void GenerateEnums(namespace_code_map *namespace_code) {
    for (auto it = parser_.enums_.vec.begin(); it != parser_.enums_.vec.end();
         ++it) {
      auto &enum_def = **it;
      GenEnum(enum_def, namespace_code);  // enum_code_ptr);
    }
  }

  void GenerateStructs(namespace_code_map *namespace_code) {
    for (auto it = parser_.structs_.vec.begin();
         it != parser_.structs_.vec.end(); ++it) {
      auto &struct_def = **it;
      GenStruct(struct_def, namespace_code);
    }
  }

  // Generate a documentation comment, if available.
  static void GenDocComment(const std::vector<std::string> &dc,
                            std::string *code_ptr,
                            const std::string &extra_lines,
                            const char *indent = nullptr) {
    if (dc.empty() && extra_lines.empty()) {
      // Don't output empty comment blocks with 0 lines of comment content.
      return;
    }

    auto &code = *code_ptr;

    for (auto it = dc.begin(); it != dc.end(); ++it) {
      if (indent) code += indent;
      code += "/// " + *it + "\n";
    }
    if (!extra_lines.empty()) {
      if (!dc.empty()) {
        if (indent) code += indent;
        code += "///\n";
      }
      if (indent) code += indent;
      std::string::size_type start = 0;
      for (;;) {
        auto end = extra_lines.find('\n', start);
        if (end != std::string::npos) {
          code += "/// " + extra_lines.substr(start, end - start) + "\n";
          start = end + 1;
        } else {
          code += "/// " + extra_lines.substr(start) + "\n";
          break;
        }
      }
    }
  }

  static void GenDocComment(std::string *code_ptr,
                            const std::string &extra_lines) {
    GenDocComment(std::vector<std::string>(), code_ptr, extra_lines);
  }

  // Generate an enum declaration and an enum string lookup table.
  void GenEnum(EnumDef &enum_def, namespace_code_map *namespace_code) {
    if (enum_def.generated) return;
    auto ns = BuildNamespaceName(*enum_def.defined_namespace);
    std::string code;
    GenDocComment(enum_def.doc_comment, &code, "");

    auto name = enum_def.is_union ? enum_def.name + "TypeId" : enum_def.name;
    auto is_bit_flags = enum_def.attributes.Lookup("bit_flags");

    code += "class " + name + " {\n";
    code += "  final int value;\n";
    code += "  const " + name + "._(this.value);\n\n";
    code += "  factory " + name + ".fromValue(int value) {\n";
    code += "    if (value == null) value = 0;\n";

    code += "    if (!values.containsKey(value)) {\n";
    code +=
        "      throw new StateError('Invalid value $value for bit flag enum ";
    code += name + "');\n";
    code += "    }\n";

    code += "    return values[value];\n";
    code += "  }\n\n";

    // this is meaningless for bit_flags
    // however, note that unlike "regular" dart enums this enum can still have
    // holes.
    if (!is_bit_flags) {
      code += "  static const int minValue = " +
              enum_def.ToString(*enum_def.MinValue()) + ";\n";
      code += "  static const int maxValue = " +
              enum_def.ToString(*enum_def.MaxValue()) + ";\n";
    }

    code +=
        "  static bool containsValue(int value) =>"
        " values.containsKey(value);\n\n";

    for (auto it = enum_def.Vals().begin(); it != enum_def.Vals().end(); ++it) {
      auto &ev = **it;

      if (!ev.doc_comment.empty()) {
        if (it != enum_def.Vals().begin()) { code += '\n'; }
        GenDocComment(ev.doc_comment, &code, "", "  ");
      }
      code += "  static const " + name + " " + ev.name + " = ";
      code += "const " + name + "._(" + enum_def.ToString(ev) + ");\n";
    }

    code += "  static get values => {";
    for (auto it = enum_def.Vals().begin(); it != enum_def.Vals().end(); ++it) {
      auto &ev = **it;
      code += enum_def.ToString(ev) + ": " + ev.name + ",";
    }
    code += "};\n\n";

    code += "  static const " + _kFb + ".Reader<" + name +
            "> reader = const _" + name + "Reader();\n\n";
    code += "  @override\n";
    code += "  String toString() {\n";
    code += "    return '" + name + "{value: $value}';\n";
    code += "  }\n";
    code += "}\n\n";

    GenEnumReader(enum_def, name, &code);
    (*namespace_code)[ns] += code;
  }

  void GenEnumReader(EnumDef &enum_def, const std::string &name,
                     std::string *code_ptr) {
    auto &code = *code_ptr;

    code += "class _" + name + "Reader extends " + _kFb + ".Reader<" + name +
            "> {\n";
    code += "  const _" + name + "Reader();\n\n";
    code += "  @override\n";
    code += "  int get size => 1;\n\n";
    code += "  @override\n";
    code +=
        "  " + name + " read(" + _kFb + ".BufferContext bc, int offset) =>\n";
    code += "      new " + name + ".fromValue(const " + _kFb + "." +
            GenType(enum_def.underlying_type) + "Reader().read(bc, offset));\n";
    code += "}\n\n";
  }

  static std::string GenType(const Type &type) {
    switch (type.base_type) {
      case BASE_TYPE_BOOL: return "Bool";
      case BASE_TYPE_CHAR: return "Int8";
      case BASE_TYPE_UTYPE:
      case BASE_TYPE_UCHAR: return "Uint8";
      case BASE_TYPE_SHORT: return "Int16";
      case BASE_TYPE_USHORT: return "Uint16";
      case BASE_TYPE_INT: return "Int32";
      case BASE_TYPE_UINT: return "Uint32";
      case BASE_TYPE_LONG: return "Int64";
      case BASE_TYPE_ULONG: return "Uint64";
      case BASE_TYPE_FLOAT: return "Float32";
      case BASE_TYPE_DOUBLE: return "Float64";
      case BASE_TYPE_STRING: return "String";
      case BASE_TYPE_VECTOR: return GenType(type.VectorType());
      case BASE_TYPE_STRUCT: return type.struct_def->name;
      case BASE_TYPE_UNION: return type.enum_def->name + "TypeId";
      default: return "Table";
    }
  }

  std::string GenReaderTypeName(const Type &type, Namespace *current_namespace,
                                const FieldDef &def,
                                bool parent_is_vector = false) {
    if (type.base_type == BASE_TYPE_BOOL) {
      return "const " + _kFb + ".BoolReader()";
    } else if (type.base_type == BASE_TYPE_VECTOR) {
      return "const " + _kFb + ".ListReader<" +
             GenDartTypeName(type.VectorType(), current_namespace, def) + ">(" +
             GenReaderTypeName(type.VectorType(), current_namespace, def,
                               true) +
             ")";
    } else if (type.base_type == BASE_TYPE_STRING) {
      return "const " + _kFb + ".StringReader()";
    }
    if (IsScalar(type.base_type)) {
      if (type.enum_def && parent_is_vector) {
        return GenDartTypeName(type, current_namespace, def) + ".reader";
      }
      return "const " + _kFb + "." + GenType(type) + "Reader()";
    } else {
      return GenDartTypeName(type, current_namespace, def) + ".reader";
    }
  }

  std::string GenDartTypeName(const Type &type, Namespace *current_namespace,
                              const FieldDef &def, bool addBuilder = false) {
    if (type.enum_def) {
      if (type.enum_def->is_union && type.base_type != BASE_TYPE_UNION) {
        return type.enum_def->name + "TypeId";
      } else if (type.enum_def->is_union) {
        return "dynamic";
      } else if (type.base_type != BASE_TYPE_VECTOR) {
        return type.enum_def->name;
      }
    }

    switch (type.base_type) {
      case BASE_TYPE_BOOL: return "bool";
      case BASE_TYPE_LONG:
      case BASE_TYPE_ULONG:
      case BASE_TYPE_INT:
      case BASE_TYPE_UINT:
      case BASE_TYPE_SHORT:
      case BASE_TYPE_USHORT:
      case BASE_TYPE_CHAR:
      case BASE_TYPE_UCHAR: return "int";
      case BASE_TYPE_FLOAT:
      case BASE_TYPE_DOUBLE: return "double";
      case BASE_TYPE_STRING: return "String";
      case BASE_TYPE_STRUCT:
        return MaybeWrapNamespace(
            type.struct_def->name + (addBuilder ? "ObjectBuilder" : ""),
            current_namespace, def);
      case BASE_TYPE_VECTOR:
        return "List<" +
               GenDartTypeName(type.VectorType(), current_namespace, def,
                               addBuilder) +
               ">";
      default: assert(0); return "dynamic";
    }
  }

  static const std::string MaybeWrapNamespace(const std::string &type_name,
                                              Namespace *current_ns,
                                              const FieldDef &field) {
    auto curr_ns_str = BuildNamespaceName(*current_ns);
    std::string field_ns_str = "";
    if (field.value.type.struct_def) {
      field_ns_str +=
          BuildNamespaceName(*field.value.type.struct_def->defined_namespace);
    } else if (field.value.type.enum_def) {
      field_ns_str +=
          BuildNamespaceName(*field.value.type.enum_def->defined_namespace);
    }

    if (field_ns_str != "" && field_ns_str != curr_ns_str) {
      return ImportAliasName(field_ns_str) + "." + type_name;
    } else {
      return type_name;
    }
  }

  // Generate an accessor struct with constructor for a flatbuffers struct.
  void GenStruct(const StructDef &struct_def,
                 namespace_code_map *namespace_code) {
    if (struct_def.generated) return;

    auto object_namespace = BuildNamespaceName(*struct_def.defined_namespace);
    std::string code;

    const auto &object_name = struct_def.name;

    // Emit constructor

    GenDocComment(struct_def.doc_comment, &code, "");

    auto reader_name = "_" + object_name + "Reader";
    auto builder_name = object_name + "Builder";
    auto object_builder_name = object_name + "ObjectBuilder";

    std::string reader_code, builder_code;

    code += "class " + object_name + " {\n";

    code += "  " + object_name + "._(this._bc, this._bcOffset);\n";
    if (!struct_def.fixed) {
      code += "  factory " + object_name + "(List<int> bytes) {\n";
      code += "    " + _kFb + ".BufferContext rootRef = new " + _kFb +
              ".BufferContext.fromBytes(bytes);\n";
      code += "    return reader.read(rootRef, 0);\n";
      code += "  }\n";
    }

    code += "\n";
    code += "  static const " + _kFb + ".Reader<" + object_name +
            "> reader = const " + reader_name + "();\n\n";

    code += "  final " + _kFb + ".BufferContext _bc;\n";
    code += "  final int _bcOffset;\n\n";

    GenImplementationGetters(struct_def, &code);

    code += "}\n\n";

    GenReader(struct_def, &reader_name, &reader_code);
    GenBuilder(struct_def, &builder_name, &builder_code);
    GenObjectBuilder(struct_def, &object_builder_name, &builder_code);

    code += reader_code;
    code += builder_code;

    (*namespace_code)[object_namespace] += code;
  }

  std::string NamespaceAliasFromUnionType(const std::string &in) {
    if (in.find('_') == std::string::npos) { return in; }

    std::stringstream ss(in);
    std::string item;
    std::vector<std::string> parts;
    std::string ns;

    while (std::getline(ss, item, '_')) { parts.push_back(item); }

    for (auto it = parts.begin(); it != parts.end() - 1; ++it) {
      auto &part = *it;

      for (size_t i = 0; i < part.length(); i++) {
        if (i && !isdigit(part[i]) &&
            part[i] == static_cast<char>(toupper(part[i]))) {
          ns += "_";
          ns += static_cast<char>(tolower(part[i]));
        } else {
          ns += static_cast<char>(tolower(part[i]));
        }
      }
      if (it != parts.end() - 2) { ns += "_"; }
    }

    return ns + "." + parts.back();
  }

  void GenImplementationGetters(const StructDef &struct_def,
                                std::string *code_ptr) {
    auto &code = *code_ptr;

    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;

      std::string field_name = MakeCamel(field.name, false);
      std::string type_name = GenDartTypeName(
          field.value.type, struct_def.defined_namespace, field, false);

      GenDocComment(field.doc_comment, &code, "", "  ");

      code += "  " + type_name + " get " + field_name;
      if (field.value.type.base_type == BASE_TYPE_UNION) {
        code += " {\n";
        code += "    switch (" + field_name + "Type?.value) {\n";
        auto &enum_def = *field.value.type.enum_def;
        for (auto en_it = enum_def.Vals().begin() + 1;
             en_it != enum_def.Vals().end(); ++en_it) {
          auto &ev = **en_it;

          auto enum_name = NamespaceAliasFromUnionType(ev.name);
          code += "      case " + enum_def.ToString(ev) + ": return " +
                  enum_name + ".reader.vTableGet(_bc, _bcOffset, " +
                  NumToString(field.value.offset) + ", null);\n";
        }
        code += "      default: return null;\n";
        code += "    }\n";
        code += "  }\n";
      } else {
        code += " => ";
        if (field.value.type.enum_def &&
            field.value.type.base_type != BASE_TYPE_VECTOR) {
          code += "new " +
                  GenDartTypeName(field.value.type,
                                  struct_def.defined_namespace, field) +
                  ".fromValue(";
        }

        code += GenReaderTypeName(field.value.type,
                                  struct_def.defined_namespace, field);
        if (struct_def.fixed) {
          code +=
              ".read(_bc, _bcOffset + " + NumToString(field.value.offset) + ")";
        } else {
          code += ".vTableGet(_bc, _bcOffset, " +
                  NumToString(field.value.offset) + ", ";
          if (!field.value.constant.empty() && field.value.constant != "0") {
            if (IsBool(field.value.type.base_type)) {
              code += "true";
            } else {
              code += field.value.constant;
            }
          } else {
            if (IsBool(field.value.type.base_type)) {
              code += "false";
            } else if (IsScalar(field.value.type.base_type)) {
              code += "0";
            } else {
              code += "null";
            }
          }
          code += ")";
        }
        if (field.value.type.enum_def &&
            field.value.type.base_type != BASE_TYPE_VECTOR) {
          code += ")";
        }
        code += ";\n";
      }
    }

    code += "\n";

    code += "  @override\n";
    code += "  String toString() {\n";
    code += "    return '" + struct_def.name + "{";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      code +=
          MakeCamel(field.name, false) + ": $" + MakeCamel(field.name, false);
      if (it != struct_def.fields.vec.end() - 1) { code += ", "; }
    }
    code += "}';\n";
    code += "  }\n";
  }

  void GenReader(const StructDef &struct_def, std::string *reader_name_ptr,
                 std::string *code_ptr) {
    auto &code = *code_ptr;
    auto &reader_name = *reader_name_ptr;
    auto &impl_name = struct_def.name;

    code += "class " + reader_name + " extends " + _kFb;
    if (struct_def.fixed) {
      code += ".StructReader<";
    } else {
      code += ".TableReader<";
    }
    code += impl_name + "> {\n";
    code += "  const " + reader_name + "();\n\n";

    if (struct_def.fixed) {
      code += "  @override\n";
      code += "  int get size => " + NumToString(struct_def.bytesize) + ";\n\n";
    }
    code += "  @override\n";
    code += "  " + impl_name +
            " createObject(fb.BufferContext bc, int offset) => \n    new " +
            impl_name + "._(bc, offset);\n";
    code += "}\n\n";
  }

  void GenBuilder(const StructDef &struct_def, std::string *builder_name_ptr,
                  std::string *code_ptr) {
    if (struct_def.fields.vec.size() == 0) { return; }
    auto &code = *code_ptr;
    auto &builder_name = *builder_name_ptr;

    code += "class " + builder_name + " {\n";
    code += "  " + builder_name + "(this.fbBuilder) {\n";
    code += "    assert(fbBuilder != null);\n";
    code += "  }\n\n";
    code += "  final " + _kFb + ".Builder fbBuilder;\n\n";

    if (struct_def.fixed) {
      StructBuilderBody(struct_def, code_ptr);
    } else {
      TableBuilderBody(struct_def, code_ptr);
    }

    code += "}\n\n";
  }

  void StructBuilderBody(const StructDef &struct_def, std::string *code_ptr) {
    auto &code = *code_ptr;

    code += "  int finish(";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;

      if (IsStruct(field.value.type)) {
        code += "fb.StructBuilder";
      } else {
        code += GenDartTypeName(field.value.type, struct_def.defined_namespace,
                                field);
      }
      code += " " + field.name;
      if (it != struct_def.fields.vec.end() - 1) { code += ", "; }
    }
    code += ") {\n";

    for (auto it = struct_def.fields.vec.rbegin();
         it != struct_def.fields.vec.rend(); ++it) {
      auto &field = **it;

      if (field.deprecated) continue;

      if (field.padding) {
        code += "    fbBuilder.pad(" + NumToString(field.padding) + ");\n";
      }

      if (IsStruct(field.value.type)) {
        code += "    " + field.name + "();\n";
      } else {
        code += "    fbBuilder.put" + GenType(field.value.type) + "(";
        code += field.name;
        if (field.value.type.enum_def) { code += "?.value"; }
        code += ");\n";
      }
    }
    code += "    return fbBuilder.offset;\n";
    code += "  }\n\n";
  }

  void TableBuilderBody(const StructDef &struct_def, std::string *code_ptr) {
    auto &code = *code_ptr;

    code += "  void begin() {\n";
    code += "    fbBuilder.startTable();\n";
    code += "  }\n\n";

    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;

      auto offset = it - struct_def.fields.vec.begin();

      if (IsScalar(field.value.type.base_type)) {
        code += "  int add" + MakeCamel(field.name) + "(";
        code += GenDartTypeName(field.value.type, struct_def.defined_namespace,
                                field);
        code += " " + MakeCamel(field.name, false) + ") {\n";
        code += "    fbBuilder.add" + GenType(field.value.type) + "(" +
                NumToString(offset) + ", ";
        code += MakeCamel(field.name, false);
        if (field.value.type.enum_def) { code += "?.value"; }
        code += ");\n";
      } else if (IsStruct(field.value.type)) {
        code += "  int add" + MakeCamel(field.name) + "(int offset) {\n";
        code +=
            "    fbBuilder.addStruct(" + NumToString(offset) + ", offset);\n";
      } else {
        code += "  int add" + MakeCamel(field.name) + "Offset(int offset) {\n";
        code +=
            "    fbBuilder.addOffset(" + NumToString(offset) + ", offset);\n";
      }
      code += "    return fbBuilder.offset;\n";
      code += "  }\n";
    }

    code += "\n";
    code += "  int finish() {\n";
    code += "    return fbBuilder.endTable();\n";
    code += "  }\n";
  }

  void GenObjectBuilder(const StructDef &struct_def,
                        std::string *builder_name_ptr, std::string *code_ptr) {
    auto &code = *code_ptr;
    auto &builder_name = *builder_name_ptr;

    code += "class " + builder_name + " extends " + _kFb + ".ObjectBuilder {\n";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      code += "  final " +
              GenDartTypeName(field.value.type, struct_def.defined_namespace,
                              field, true) +
              " _" + MakeCamel(field.name, false) + ";\n";
    }
    code += "\n";
    code += "  " + builder_name + "(";
    if (struct_def.fields.vec.size() != 0) {
      code +=

          "{\n";
      for (auto it = struct_def.fields.vec.begin();
           it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (field.deprecated) continue;
        code += "    " +
                GenDartTypeName(field.value.type, struct_def.defined_namespace,
                                field, true) +
                " " + MakeCamel(field.name, false) + ",\n";
      }
      code += "  })\n";
      code += "      : ";
      for (auto it = struct_def.fields.vec.begin();
           it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (field.deprecated) continue;
        code += "_" + MakeCamel(field.name, false) + " = " +
                MakeCamel(field.name, false);
        if (it == struct_def.fields.vec.end() - 1) {
          code += ";\n\n";
        } else {
          code += ",\n        ";
        }
      }
    } else {
      code += ");\n\n";
    }

    code += "  /// Finish building, and store into the [fbBuilder].\n";
    code += "  @override\n";
    code += "  int finish(\n";
    code += "    " + _kFb + ".Builder fbBuilder) {\n";
    code += "    assert(fbBuilder != null);\n";

    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      if (IsScalar(field.value.type.base_type) || IsStruct(field.value.type))
        continue;

      code += "    final int " + MakeCamel(field.name, false) + "Offset";
      if (field.value.type.base_type == BASE_TYPE_VECTOR) {
        code +=
            " = _" + MakeCamel(field.name, false) + "?.isNotEmpty == true\n";
        code += "        ? fbBuilder.writeList";
        switch (field.value.type.VectorType().base_type) {
          case BASE_TYPE_STRING:
            code += "(_" + MakeCamel(field.name, false) +
                    ".map((b) => fbBuilder.writeString(b)).toList())";
            break;
          case BASE_TYPE_STRUCT:
            if (field.value.type.struct_def->fixed) {
              code += "OfStructs(_" + MakeCamel(field.name, false) + ")";
            } else {
              code += "(_" + MakeCamel(field.name, false) +
                      ".map((b) => b.getOrCreateOffset(fbBuilder)).toList())";
            }
            break;
          default:
            code += GenType(field.value.type.VectorType()) + "(_" +
                    MakeCamel(field.name, false);
            if (field.value.type.enum_def) { code += ".map((f) => f.value)"; }
            code += ")";
        }
        code += "\n        : null;\n";
      } else if (field.value.type.base_type == BASE_TYPE_STRING) {
        code += " = fbBuilder.writeString(_" + MakeCamel(field.name, false) + ");\n";
      } else {
        code += " = _" + MakeCamel(field.name, false) +
                "?.getOrCreateOffset(fbBuilder);\n";
      }
    }

    code += "\n";
    if (struct_def.fixed) {
      StructObjectBuilderBody(struct_def, code_ptr);
    } else {
      TableObjectBuilderBody(struct_def, code_ptr);
    }
    code += "  }\n\n";

    code += "  /// Convenience method to serialize to byte list.\n";
    code += "  @override\n";
    code += "  Uint8List toBytes([String fileIdentifier]) {\n";
    code += "    " + _kFb + ".Builder fbBuilder = new ";
    code += _kFb + ".Builder();\n";
    code += "    int offset = finish(fbBuilder);\n";
    code += "    return fbBuilder.finish(offset, fileIdentifier);\n";
    code += "  }\n";
    code += "}\n";
  }

  void StructObjectBuilderBody(const StructDef &struct_def,
                               std::string *code_ptr,
                               bool prependUnderscore = true) {
    auto &code = *code_ptr;

    for (auto it = struct_def.fields.vec.rbegin();
         it != struct_def.fields.vec.rend(); ++it) {
      auto &field = **it;

      if (field.deprecated) continue;

      if (field.padding) {
        code += "    fbBuilder.pad(" + NumToString(field.padding) + ");\n";
      }

      if (IsStruct(field.value.type)) {
        code += "    ";
        if (prependUnderscore) { code += "_"; }
        code += field.name + ".finish(fbBuilder);\n";
      } else {
        code += "    fbBuilder.put" + GenType(field.value.type) + "(";
        if (prependUnderscore) { code += "_"; }
        code += field.name;
        if (field.value.type.enum_def) { code += "?.value"; }
        code += ");\n";
      }
    }

    code += "    return fbBuilder.offset;\n";
  }

  void TableObjectBuilderBody(const StructDef &struct_def,
                              std::string *code_ptr,
                              bool prependUnderscore = true) {
    std::string &code = *code_ptr;
    code += "    fbBuilder.startTable();\n";

    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;

      if (field.deprecated) continue;

      auto offset = it - struct_def.fields.vec.begin();
      if (IsScalar(field.value.type.base_type)) {
        code += "    fbBuilder.add" + GenType(field.value.type) + "(" +
                NumToString(offset) + ", ";
        if (prependUnderscore) { code += "_"; }
        code += MakeCamel(field.name, false);
        if (field.value.type.enum_def) { code += "?.value"; }
        code += ");\n";
      } else if (IsStruct(field.value.type)) {
        code += "    if (";
        if (prependUnderscore) { code += "_"; }
        code += MakeCamel(field.name, false) + " != null) {\n";
        code += "      fbBuilder.addStruct(" + NumToString(offset) + ", ";
        code += "_" + MakeCamel(field.name, false) + ".finish(fbBuilder));\n";
        code += "    }\n";
      } else {
        code +=
            "    if (" + MakeCamel(field.name, false) + "Offset != null) {\n";
        code += "      fbBuilder.addOffset(" + NumToString(offset) + ", " +
                MakeCamel(field.name, false) + "Offset);\n";
        code += "    }\n";
      }
    }
    code += "    return fbBuilder.endTable();\n";
  }
};
}  // namespace dart

bool GenerateDart(const Parser &parser, const std::string &path,
                  const std::string &file_name) {
  dart::DartGenerator generator(parser, path, file_name);
  return generator.generate();
}

std::string DartMakeRule(const Parser &parser, const std::string &path,
                         const std::string &file_name) {
  assert(parser.opts.lang <= IDLOptions::kMAX);

  auto filebase =
      flatbuffers::StripPath(flatbuffers::StripExtension(file_name));
  auto make_rule = GeneratedFileName(path, filebase) + ": ";

  auto included_files = parser.GetIncludedFilesRecursive(file_name);
  for (auto it = included_files.begin(); it != included_files.end(); ++it) {
    make_rule += " " + *it;
  }
  return make_rule;
}

}  // namespace flatbuffers
