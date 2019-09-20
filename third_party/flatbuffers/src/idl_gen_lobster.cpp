/*
 * Copyright 2018 Google Inc. All rights reserved.
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

#include <string>
#include <unordered_set>

#include "flatbuffers/code_generators.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

namespace flatbuffers {
namespace lobster {

class LobsterGenerator : public BaseGenerator {
 public:
 LobsterGenerator(const Parser &parser, const std::string &path,
                  const std::string &file_name)
      : BaseGenerator(parser, path, file_name, "" /* not used */, "_") {
    static const char * const keywords[] = {
      "nil", "true", "false", "return", "struct", "class", "import", "int",
      "float", "string", "any", "def", "is", "from", "program", "private",
      "coroutine", "resource", "enum", "typeof", "var", "let", "pakfile",
      "switch", "case", "default", "namespace", "not", "and", "or", "bool",
    };
    keywords_.insert(std::begin(keywords), std::end(keywords));
  }

  std::string EscapeKeyword(const std::string &name) const {
    return keywords_.find(name) == keywords_.end() ? name : name + "_";
  }

  std::string NormalizedName(const Definition &definition) const {
    return EscapeKeyword(definition.name);
  }

  std::string NormalizedName(const EnumVal &ev) const {
    return EscapeKeyword(ev.name);
  }

  std::string NamespacedName(const Definition &def) {
    return WrapInNameSpace(def.defined_namespace, NormalizedName(def));
  }

  std::string GenTypeName(const Type &type) {
    auto bits = NumToString(SizeOf(type.base_type) * 8);
    if (IsInteger(type.base_type)) return "int" + bits;
    if (IsFloat(type.base_type)) return "float" + bits;
    if (type.base_type == BASE_TYPE_STRING) return "string";
    if (type.base_type == BASE_TYPE_STRUCT) return "table";
    return "none";
  }

  std::string LobsterType(const Type &type) {
    if (IsFloat(type.base_type)) return "float";
    if (IsScalar(type.base_type) && type.enum_def) return NormalizedName(*type.enum_def);
    if (!IsScalar(type.base_type)) return "flatbuffers_offset";
    return "int";
  }

  // Returns the method name for use with add/put calls.
  std::string GenMethod(const Type &type) {
    return IsScalar(type.base_type)
      ? MakeCamel(GenTypeBasic(type))
      : (IsStruct(type) ? "Struct" : "UOffsetTRelative");
  }

  // This uses Python names for now..
  std::string GenTypeBasic(const Type &type) {
    static const char *ctypename[] = {
      // clang-format off
      #define FLATBUFFERS_TD(ENUM, IDLTYPE, \
        CTYPE, JTYPE, GTYPE, NTYPE, PTYPE, RTYPE, KTYPE) \
        #PTYPE,
      FLATBUFFERS_GEN_TYPES(FLATBUFFERS_TD)
      #undef FLATBUFFERS_TD
      // clang-format on
    };
    return ctypename[type.base_type];
  }

  // Generate a struct field, conditioned on its child type(s).
  void GenStructAccessor(const StructDef &struct_def,
                         const FieldDef &field, std::string *code_ptr) {
    GenComment(field.doc_comment, code_ptr, nullptr, "    ");
    std::string &code = *code_ptr;
    auto offsets = NumToString(field.value.offset);
    auto def = "    def " + NormalizedName(field);
    if (IsScalar(field.value.type.base_type)) {
      std::string acc;
      if (struct_def.fixed) {
        acc = "buf_.read_" + GenTypeName(field.value.type) +
              "_le(pos_ + " + offsets + ")";

      } else {
        acc = "buf_.flatbuffers_field_" +
              GenTypeName(field.value.type) + "(pos_, " + offsets + ", " +
              field.value.constant + ")";
      }
      if (field.value.type.enum_def)
        acc = NormalizedName(*field.value.type.enum_def) + "(" + acc + ")";
      code += def + "():\n        return " + acc + "\n";
      return;
    }
    switch (field.value.type.base_type) {
      case BASE_TYPE_STRUCT: {
        auto name = NamespacedName(*field.value.type.struct_def);
        code += def + "():\n        ";
        if (struct_def.fixed) {
          code += "return " + name + "{ buf_, pos_ + " + offsets + " }\n";
        } else {
          code += std::string("let o = buf_.flatbuffers_field_") +
                  (field.value.type.struct_def->fixed ? "struct" : "table") +
                  "(pos_, " + offsets + ")\n        return if o: " + name +
                  " { buf_, o } else: nil\n";
        }
        break;
      }
      case BASE_TYPE_STRING:
        code += def + "():\n        return buf_.flatbuffers_field_string(pos_, " +
                offsets + ")\n";
        break;
      case BASE_TYPE_VECTOR: {
        auto vectortype = field.value.type.VectorType();
        code += def + "(i:int):\n        return ";
        if (vectortype.base_type == BASE_TYPE_STRUCT) {
          auto start = "buf_.flatbuffers_field_vector(pos_, " + offsets +
                       ") + i * " + NumToString(InlineSize(vectortype));
          if (!(vectortype.struct_def->fixed)) {
            start = "buf_.flatbuffers_indirect(" + start + ")";
          }
          code += NamespacedName(*field.value.type.struct_def) + " { buf_, " +
                  start + " }\n";
        } else {
          if (vectortype.base_type == BASE_TYPE_STRING)
            code += "buf_.flatbuffers_string";
          else
            code += "buf_.read_" + GenTypeName(vectortype) + "_le";
          code += "(buf_.flatbuffers_field_vector(pos_, " + offsets +
                  ") + i * " + NumToString(InlineSize(vectortype)) + ")\n";
        }
        break;
      }
      case BASE_TYPE_UNION: {
        for (auto it = field.value.type.enum_def->Vals().begin();
             it != field.value.type.enum_def->Vals().end(); ++it) {
          auto &ev = **it;
          if (ev.IsNonZero()) {
            code += def + "_as_" + ev.name + "():\n        return " +
                    NamespacedName(*ev.union_type.struct_def) +
                    " { buf_, buf_.flatbuffers_field_table(pos_, " + offsets +
                    ") }\n";
          }
        }
        break;
      }
      default: FLATBUFFERS_ASSERT(0);
    }
    if (field.value.type.base_type == BASE_TYPE_VECTOR) {
      code += def +
              "_length():\n        return buf_.flatbuffers_field_vector_len(pos_, " +
              offsets + ")\n";
    }
  }

  // Generate table constructors, conditioned on its members' types.
  void GenTableBuilders(const StructDef &struct_def,
                        std::string *code_ptr) {
    std::string &code = *code_ptr;
    code += "struct " + NormalizedName(struct_def) +
            "Builder:\n    b_:flatbuffers_builder\n";
    code += "    def start():\n        b_.StartObject(" +
            NumToString(struct_def.fields.vec.size()) + ")\n        return this\n";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      auto offset = it - struct_def.fields.vec.begin();
      code += "    def add_" + NormalizedName(field) + "(" +
              NormalizedName(field) + ":" + LobsterType(field.value.type) +
              "):\n        b_.Prepend" + GenMethod(field.value.type) + "Slot(" +
              NumToString(offset) + ", " + NormalizedName(field);
      if (IsScalar(field.value.type.base_type))
        code += ", " + field.value.constant;
      code += ")\n        return this\n";
    }
    code += "    def end():\n        return b_.EndObject()\n\n";
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      if (field.value.type.base_type == BASE_TYPE_VECTOR) {
        code += "def " + NormalizedName(struct_def) + "Start" +
                MakeCamel(NormalizedName(field)) +
                "Vector(b_:flatbuffers_builder, n_:int):\n    b_.StartVector(";
        auto vector_type = field.value.type.VectorType();
        auto alignment = InlineAlignment(vector_type);
        auto elem_size = InlineSize(vector_type);
        code += NumToString(elem_size) + ", n_, " + NumToString(alignment) +
                ")\n";
        if (vector_type.base_type != BASE_TYPE_STRUCT ||
            !vector_type.struct_def->fixed) {
          code += "def " + NormalizedName(struct_def) + "Create" +
                  MakeCamel(NormalizedName(field)) +
                  "Vector(b_:flatbuffers_builder, v_:[" +
                  LobsterType(vector_type) + "]):\n    b_.StartVector(" +
                  NumToString(elem_size) + ", v_.length, " +
                  NumToString(alignment) +
                  ")\n    reverse(v_) e_: b_.Prepend" +
                  GenMethod(vector_type) +
                  "(e_)\n    return b_.EndVector(v_.length)\n";
        }
        code += "\n";
      }
    }
  }

  void GenStructPreDecl(const StructDef &struct_def, std::string *code_ptr) {
    if (struct_def.generated) return;
    std::string &code = *code_ptr;
    CheckNameSpace(struct_def, &code);
    code += "class " + NormalizedName(struct_def) + "\n\n";
  }

  // Generate struct or table methods.
  void GenStruct(const StructDef &struct_def, std::string *code_ptr) {
    if (struct_def.generated) return;
    std::string &code = *code_ptr;
    CheckNameSpace(struct_def, &code);
    GenComment(struct_def.doc_comment, code_ptr, nullptr, "");
    code += "class " + NormalizedName(struct_def) + " : flatbuffers_handle\n";
    for (auto it = struct_def.fields.vec.begin();
        it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (field.deprecated) continue;
      GenStructAccessor(struct_def, field, code_ptr);
    }
    code += "\n";
    if (!struct_def.fixed) {
      // Generate a special accessor for the table that has been declared as
      // the root type.
      code += "def GetRootAs" + NormalizedName(struct_def) + "(buf:string): return " +
              NormalizedName(struct_def) +
              " { buf, buf.flatbuffers_indirect(0) }\n\n";
    }
    if (struct_def.fixed) {
      // create a struct constructor function
      GenStructBuilder(struct_def, code_ptr);
    } else {
      // Create a set of functions that allow table construction.
      GenTableBuilders(struct_def, code_ptr);
    }
  }

  // Generate enum declarations.
  void GenEnum(const EnumDef &enum_def, std::string *code_ptr) {
    if (enum_def.generated) return;
    std::string &code = *code_ptr;
    CheckNameSpace(enum_def, &code);
    GenComment(enum_def.doc_comment, code_ptr, nullptr, "");
    code += "enum " + NormalizedName(enum_def) + ":\n";
    for (auto it = enum_def.Vals().begin(); it != enum_def.Vals().end(); ++it) {
      auto &ev = **it;
      GenComment(ev.doc_comment, code_ptr, nullptr, "    ");
      code += "    " + enum_def.name + "_" + NormalizedName(ev) + " = " +
              enum_def.ToString(ev) + "\n";
    }
    code += "\n";
  }

  // Recursively generate arguments for a constructor, to deal with nested
  // structs.
  void StructBuilderArgs(const StructDef &struct_def,
                         const char *nameprefix, std::string *code_ptr) {
    for (auto it = struct_def.fields.vec.begin();
         it != struct_def.fields.vec.end(); ++it) {
      auto &field = **it;
      if (IsStruct(field.value.type)) {
        // Generate arguments for a struct inside a struct. To ensure names
        // don't clash, and to make it obvious these arguments are constructing
        // a nested struct, prefix the name with the field name.
        StructBuilderArgs(*field.value.type.struct_def,
          (nameprefix + (NormalizedName(field) + "_")).c_str(), code_ptr);
      } else {
        std::string &code = *code_ptr;
        code += ", " + (nameprefix + NormalizedName(field)) + ":" +
                LobsterType(field.value.type);
      }
    }
  }

  // Recursively generate struct construction statements and instert manual
  // padding.
  void StructBuilderBody(const StructDef &struct_def,
                         const char *nameprefix, std::string *code_ptr) {
    std::string &code = *code_ptr;
    code += "    b_.Prep(" + NumToString(struct_def.minalign) + ", " +
            NumToString(struct_def.bytesize) + ")\n";
    for (auto it = struct_def.fields.vec.rbegin();
         it != struct_def.fields.vec.rend(); ++it) {
      auto &field = **it;
      if (field.padding)
        code += "    b_.Pad(" + NumToString(field.padding) + ")\n";
      if (IsStruct(field.value.type)) {
        StructBuilderBody(*field.value.type.struct_def,
          (nameprefix + (NormalizedName(field) + "_")).c_str(), code_ptr);
      } else {
        code += "    b_.Prepend" + GenMethod(field.value.type) + "(" +
                nameprefix + NormalizedName(field) + ")\n";
      }
    }
  }

  // Create a struct with a builder and the struct's arguments.
  void GenStructBuilder(const StructDef &struct_def,
                              std::string *code_ptr) {
    std::string &code = *code_ptr;
    code += "def Create" + NormalizedName(struct_def) +
            "(b_:flatbuffers_builder";
    StructBuilderArgs(struct_def, "", code_ptr);
    code += "):\n";
    StructBuilderBody(struct_def, "", code_ptr);
    code += "    return b_.Offset()\n\n";
  }

  void CheckNameSpace(const Definition &def, std::string *code_ptr) {
    auto ns = GetNameSpace(def);
    if (ns == current_namespace_) return;
    current_namespace_ = ns;
    std::string &code = *code_ptr;
    code += "namespace " + ns + "\n\n";
  }

  bool generate() {
    std::string code;
    code += std::string("// ") + FlatBuffersGeneratedWarning() +
            "\nimport flatbuffers\n\n";
    for (auto it = parser_.enums_.vec.begin(); it != parser_.enums_.vec.end();
         ++it) {
      auto &enum_def = **it;
      GenEnum(enum_def, &code);
    }
    for (auto it = parser_.structs_.vec.begin();
         it != parser_.structs_.vec.end(); ++it) {
      auto &struct_def = **it;
      GenStructPreDecl(struct_def, &code);
    }
    for (auto it = parser_.structs_.vec.begin();
         it != parser_.structs_.vec.end(); ++it) {
      auto &struct_def = **it;
      GenStruct(struct_def, &code);
    }
    return SaveFile((path_ + file_name_ + "_generated.lobster").c_str(),
                    code, false);
  }

 private:
  std::unordered_set<std::string> keywords_;
  std::string current_namespace_;
};

}  // namespace lobster

bool GenerateLobster(const Parser &parser, const std::string &path,
                    const std::string &file_name) {
  lobster::LobsterGenerator generator(parser, path, file_name);
  return generator.generate();
}

}  // namespace flatbuffers
