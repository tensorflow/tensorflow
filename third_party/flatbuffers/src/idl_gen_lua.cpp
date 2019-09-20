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

#include <string>

#include "flatbuffers/code_generators.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include <unordered_set>

namespace flatbuffers {
namespace lua {

  // Hardcode spaces per indentation.
  const CommentConfig def_comment = { nullptr, "--", nullptr };
  const char * Indent = "    ";
  const char * Comment = "-- ";
  const char * End = "end\n";
  const char * EndFunc = "end\n";
  const char * SelfData = "self.view";
  const char * SelfDataPos = "self.view.pos";
  const char * SelfDataBytes = "self.view.bytes";

  class LuaGenerator : public BaseGenerator {
  public:
    LuaGenerator(const Parser &parser, const std::string &path,
      const std::string &file_name)
      : BaseGenerator(parser, path, file_name, "" /* not used */,
        "" /* not used */) {
      static const char * const keywords[] = {
        "and",
        "break",
        "do",
        "else",
        "elseif",
        "end",
        "false",
        "for",
        "function",
        "goto",
        "if",
        "in",
        "local",
        "nil",
        "not",
        "or",
        "repeat",
        "return",
        "then",
        "true",
        "until",
        "while"
      };
      keywords_.insert(std::begin(keywords), std::end(keywords));
    }

    // Most field accessors need to retrieve and test the field offset first,
    // this is the prefix code for that.
    std::string OffsetPrefix(const FieldDef &field) {
      return std::string(Indent) +
        "local o = " + SelfData + ":Offset(" + NumToString(field.value.offset) + ")\n" +
        Indent + "if o ~= 0 then\n";
    }

    // Begin a class declaration.
    void BeginClass(const StructDef &struct_def, std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "local " + NormalizedName(struct_def) + " = {} -- the module\n";
      code += "local " + NormalizedMetaName(struct_def) + " = {} -- the class metatable\n";
      code += "\n";
    }

    // Begin enum code with a class declaration.
    void BeginEnum(const std::string &class_name, std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "local " + class_name + " = {\n";
    }

    std::string EscapeKeyword(const std::string &name) const {
      return keywords_.find(name) == keywords_.end() ? name : "_" + name;
    }

    std::string NormalizedName(const Definition &definition) const {
      return EscapeKeyword(definition.name);
    }

    std::string NormalizedName(const EnumVal &ev) const {
      return EscapeKeyword(ev.name);
    }

    std::string NormalizedMetaName(const Definition &definition) const {
      return EscapeKeyword(definition.name) + "_mt";
    }

    // A single enum member.
    void EnumMember(const EnumDef &enum_def, const EnumVal &ev, std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += std::string(Indent) + NormalizedName(ev) + " = " +
              enum_def.ToString(ev) + ",\n";
    }

    // End enum code.
    void EndEnum(std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "}\n";
    }

    void GenerateNewObjectPrototype(const StructDef &struct_def,
      std::string *code_ptr) {
      std::string &code = *code_ptr;

      code += "function " + NormalizedName(struct_def) + ".New()\n";
      code += std::string(Indent) + "local o = {}\n";
      code += std::string(Indent) + "setmetatable(o, {__index = " + NormalizedMetaName(struct_def) + "})\n";
      code += std::string(Indent) + "return o\n";
      code += EndFunc;
    }

    // Initialize a new struct or table from existing data.
    void NewRootTypeFromBuffer(const StructDef &struct_def,
      std::string *code_ptr) {
      std::string &code = *code_ptr;

      code += "function " + NormalizedName(struct_def) + ".GetRootAs" + NormalizedName(struct_def) + "(buf, offset)\n";
      code += std::string(Indent) + "local n = flatbuffers.N.UOffsetT:Unpack(buf, offset)\n";
      code += std::string(Indent) + "local o = " + NormalizedName(struct_def) + ".New()\n";
      code += std::string(Indent) + "o:Init(buf, n + offset)\n";
      code += std::string(Indent) + "return o\n";
      code += EndFunc;
    }

    // Initialize an existing object with other data, to avoid an allocation.
    void InitializeExisting(const StructDef &struct_def,
      std::string *code_ptr) {
      std::string &code = *code_ptr;

      GenReceiver(struct_def, code_ptr);
      code += "Init(buf, pos)\n";
      code += std::string(Indent) + SelfData + " = flatbuffers.view.New(buf, pos)\n";
      code += EndFunc;
    }

    // Get the length of a vector.
    void GetVectorLen(const StructDef &struct_def, const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;

      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field)) + "Length()\n";
      code += OffsetPrefix(field);
      code += std::string(Indent) + Indent + "return " + SelfData + ":VectorLen(o)\n";
      code += std::string(Indent) + End;
      code += std::string(Indent) + "return 0\n";
      code += EndFunc;
    }

    // Get the value of a struct's scalar.
    void GetScalarFieldOfStruct(const StructDef &struct_def,
      const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      std::string getter = GenGetter(field.value.type);
      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "()\n";
      code += std::string(Indent) + "return " + getter;
      code += std::string(SelfDataPos) + " + " + NumToString(field.value.offset) + ")\n";
      code += EndFunc;
    }

    // Get the value of a table's scalar.
    void GetScalarFieldOfTable(const StructDef &struct_def,
      const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      std::string getter = GenGetter(field.value.type);
      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "()\n";
      code += OffsetPrefix(field);
      getter += std::string("o + ") + SelfDataPos + ")";
      auto is_bool = field.value.type.base_type == BASE_TYPE_BOOL;
      if (is_bool) {
        getter = "(" + getter + " ~= 0)";
      }
      code += std::string(Indent) + Indent + "return " + getter + "\n";
      code += std::string(Indent) + End;
      std::string default_value;
      if (is_bool) {
        default_value = field.value.constant == "0" ? "false" : "true";
      }
      else {
        default_value = field.value.constant;
      }
      code += std::string(Indent) + "return " + default_value + "\n";
      code += EndFunc;
    }

    // Get a struct by initializing an existing struct.
    // Specific to Struct.
    void GetStructFieldOfStruct(const StructDef &struct_def,
      const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "(obj)\n";
      code += std::string(Indent) + "obj:Init(" + SelfDataBytes + ", " + SelfDataPos + " + ";
      code += NumToString(field.value.offset) + ")\n";
      code += std::string(Indent) + "return obj\n";
      code += EndFunc;
    }

    // Get a struct by initializing an existing struct.
    // Specific to Table.
    void GetStructFieldOfTable(const StructDef &struct_def,
      const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "()\n";
      code += OffsetPrefix(field);
      if (field.value.type.struct_def->fixed) {
        code += std::string(Indent) + Indent + "local x = o + " + SelfDataPos + "\n";
      }
      else {
        code += std::string(Indent) + Indent + "local x = " + SelfData + ":Indirect(o + " + SelfDataPos + ")\n";
      }
      code += std::string(Indent) + Indent + "local obj = require('" + TypeNameWithNamespace(field) + "').New()\n";
      code += std::string(Indent) + Indent + "obj:Init(" + SelfDataBytes + ", x)\n";
      code += std::string(Indent) + Indent + "return obj\n";
      code += std::string(Indent) + End;
      code += EndFunc;
    }

    // Get the value of a string.
    void GetStringField(const StructDef &struct_def, const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "()\n";
      code += OffsetPrefix(field);
      code += std::string(Indent) + Indent + "return " + GenGetter(field.value.type);
      code += std::string("o + ") + SelfDataPos + ")\n";
      code += std::string(Indent) + End;
      code += EndFunc;
    }

    // Get the value of a union from an object.
    void GetUnionField(const StructDef &struct_def, const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field)) + "()\n";
      code += OffsetPrefix(field);

      // TODO(rw): this works and is not the good way to it:
      //bool is_native_table = TypeName(field) == "*flatbuffers.Table";
      //if (is_native_table) {
      //  code += std::string(Indent) + Indent + "from flatbuffers.table import Table\n";
      //} else {
      //  code += std::string(Indent) + Indent +
      //  code += "from ." + TypeName(field) + " import " + TypeName(field) + "\n";
      //}
      code += std::string(Indent) + Indent + "local obj = flatbuffers.view.New(require('flatbuffers.binaryarray').New(0), 0)\n";
      code += std::string(Indent) + Indent + GenGetter(field.value.type) + "obj, o)\n";
      code += std::string(Indent) + Indent + "return obj\n";
      code += std::string(Indent) + End;
      code += EndFunc;
    }

    // Get the value of a vector's struct member.
    void GetMemberOfVectorOfStruct(const StructDef &struct_def,
      const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      auto vectortype = field.value.type.VectorType();

      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "(j)\n";
      code += OffsetPrefix(field);
      code += std::string(Indent) + Indent + "local x = " + SelfData + ":Vector(o)\n";
      code += std::string(Indent) + Indent + "x = x + ((j-1) * ";
      code += NumToString(InlineSize(vectortype)) + ")\n";
      if (!(vectortype.struct_def->fixed)) {
        code += std::string(Indent) + Indent + "x = " + SelfData + ":Indirect(x)\n";
      }
      code += std::string(Indent) + Indent + "local obj = require('" + TypeNameWithNamespace(field) + "').New()\n";
      code += std::string(Indent) + Indent + "obj:Init(" + SelfDataBytes + ", x)\n";
      code += std::string(Indent) + Indent + "return obj\n";
      code += std::string(Indent) + End;
      code += EndFunc;
    }

    // Get the value of a vector's non-struct member. Uses a named return
    // argument to conveniently set the zero value for the result.
    void GetMemberOfVectorOfNonStruct(const StructDef &struct_def,
      const FieldDef &field,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      auto vectortype = field.value.type.VectorType();

      GenReceiver(struct_def, code_ptr);
      code += MakeCamel(NormalizedName(field));
      code += "(j)\n";
      code += OffsetPrefix(field);
      code += std::string(Indent) + Indent + "local a = " + SelfData + ":Vector(o)\n";
      code += std::string(Indent) + Indent;
      code += "return " + GenGetter(field.value.type);
      code += "a + ((j-1) * ";
      code += NumToString(InlineSize(vectortype)) + "))\n";
      code += std::string(Indent) + End;
      if (vectortype.base_type == BASE_TYPE_STRING) {
        code += std::string(Indent) + "return ''\n";
      }
      else {
        code += std::string(Indent) + "return 0\n";
      }
      code += EndFunc;
    }

    // Begin the creator function signature.
    void BeginBuilderArgs(const StructDef &struct_def,
      std::string *code_ptr) {
      std::string &code = *code_ptr;

      code += "function " + NormalizedName(struct_def) + ".Create" + NormalizedName(struct_def);
      code += "(builder";
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
        }
        else {
          std::string &code = *code_ptr;
          code += std::string(", ") + nameprefix;
          code += MakeCamel(NormalizedName(field), false);
        }
      }
    }

    // End the creator function signature.
    void EndBuilderArgs(std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += ")\n";
    }

    // Recursively generate struct construction statements and instert manual
    // padding.
    void StructBuilderBody(const StructDef &struct_def,
      const char *nameprefix, std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += std::string(Indent) + "builder:Prep(" + NumToString(struct_def.minalign) + ", ";
      code += NumToString(struct_def.bytesize) + ")\n";
      for (auto it = struct_def.fields.vec.rbegin();
        it != struct_def.fields.vec.rend(); ++it) {
        auto &field = **it;
        if (field.padding)
          code += std::string(Indent) + "builder:Pad(" + NumToString(field.padding) + ")\n";
        if (IsStruct(field.value.type)) {
          StructBuilderBody(*field.value.type.struct_def,
            (nameprefix + (NormalizedName(field) + "_")).c_str(), code_ptr);
        }
        else {
          code += std::string(Indent) + "builder:Prepend" + GenMethod(field) + "(";
          code += nameprefix + MakeCamel(NormalizedName(field), false) + ")\n";
        }
      }
    }

    void EndBuilderBody(std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += std::string(Indent) + "return builder:Offset()\n";
      code += EndFunc;
    }

    // Get the value of a table's starting offset.
    void GetStartOfTable(const StructDef &struct_def,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "function " + NormalizedName(struct_def) + ".Start";
      code += "(builder) ";
      code += "builder:StartObject(";
      code += NumToString(struct_def.fields.vec.size());
      code += ") end\n";
    }

    // Set the value of a table's field.
    void BuildFieldOfTable(const StructDef &struct_def,
      const FieldDef &field, const size_t offset,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "function " + NormalizedName(struct_def) + ".Add" + MakeCamel(NormalizedName(field));
      code += "(builder, ";
      code += MakeCamel(NormalizedName(field), false);
      code += ") ";
      code += "builder:Prepend";
      code += GenMethod(field) + "Slot(";
      code += NumToString(offset) + ", ";
      // todo: i don't need to cast in Lua, but am I missing something?
    //    if (!IsScalar(field.value.type.base_type) && (!struct_def.fixed)) {
    //      code += "flatbuffers.N.UOffsetTFlags.py_type";
    //      code += "(";
    //      code += MakeCamel(NormalizedName(field), false) + ")";
    //    } else {
      code += MakeCamel(NormalizedName(field), false);
      //    }
      code += ", " + field.value.constant;
      code += ") end\n";
    }

    // Set the value of one of the members of a table's vector.
    void BuildVectorOfTable(const StructDef &struct_def,
      const FieldDef &field, std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "function " + NormalizedName(struct_def) + ".Start";
      code += MakeCamel(NormalizedName(field));
      code += "Vector(builder, numElems) return builder:StartVector(";
      auto vector_type = field.value.type.VectorType();
      auto alignment = InlineAlignment(vector_type);
      auto elem_size = InlineSize(vector_type);
      code += NumToString(elem_size);
      code += ", numElems, " + NumToString(alignment);
      code += ") end\n";
    }

    // Get the offset of the end of a table.
    void GetEndOffsetOnTable(const StructDef &struct_def,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "function " + NormalizedName(struct_def) + ".End";
      code += "(builder) ";
      code += "return builder:EndObject() end\n";
    }

    // Generate the receiver for function signatures.
    void GenReceiver(const StructDef &struct_def, std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += "function " + NormalizedMetaName(struct_def) + ":";
    }

    // Generate a struct field, conditioned on its child type(s).
    void GenStructAccessor(const StructDef &struct_def,
      const FieldDef &field, std::string *code_ptr) {
      GenComment(field.doc_comment, code_ptr, &def_comment);
      if (IsScalar(field.value.type.base_type)) {
        if (struct_def.fixed) {
          GetScalarFieldOfStruct(struct_def, field, code_ptr);
        }
        else {
          GetScalarFieldOfTable(struct_def, field, code_ptr);
        }
      }
      else {
        switch (field.value.type.base_type) {
        case BASE_TYPE_STRUCT:
          if (struct_def.fixed) {
            GetStructFieldOfStruct(struct_def, field, code_ptr);
          }
          else {
            GetStructFieldOfTable(struct_def, field, code_ptr);
          }
          break;
        case BASE_TYPE_STRING: GetStringField(struct_def, field, code_ptr); break;
        case BASE_TYPE_VECTOR: {
          auto vectortype = field.value.type.VectorType();
          if (vectortype.base_type == BASE_TYPE_STRUCT) {
            GetMemberOfVectorOfStruct(struct_def, field, code_ptr);
          }
          else {
            GetMemberOfVectorOfNonStruct(struct_def, field, code_ptr);
          }
          break;
        }
        case BASE_TYPE_UNION: GetUnionField(struct_def, field, code_ptr); break;
        default: FLATBUFFERS_ASSERT(0);
        }
      }
      if (field.value.type.base_type == BASE_TYPE_VECTOR) {
        GetVectorLen(struct_def, field, code_ptr);
      }
    }

    // Generate table constructors, conditioned on its members' types.
    void GenTableBuilders(const StructDef &struct_def,
      std::string *code_ptr) {
      GetStartOfTable(struct_def, code_ptr);

      for (auto it = struct_def.fields.vec.begin();
        it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (field.deprecated) continue;

        auto offset = it - struct_def.fields.vec.begin();
        BuildFieldOfTable(struct_def, field, offset, code_ptr);
        if (field.value.type.base_type == BASE_TYPE_VECTOR) {
          BuildVectorOfTable(struct_def, field, code_ptr);
        }
      }

      GetEndOffsetOnTable(struct_def, code_ptr);
    }

    // Generate struct or table methods.
    void GenStruct(const StructDef &struct_def, std::string *code_ptr) {
      if (struct_def.generated) return;

      GenComment(struct_def.doc_comment, code_ptr, &def_comment);
      BeginClass(struct_def, code_ptr);

      GenerateNewObjectPrototype(struct_def, code_ptr);

      if (!struct_def.fixed) {
        // Generate a special accessor for the table that has been declared as
        // the root type.
        NewRootTypeFromBuffer(struct_def, code_ptr);
      }

      // Generate the Init method that sets the field in a pre-existing
      // accessor object. This is to allow object reuse.
      InitializeExisting(struct_def, code_ptr);
      for (auto it = struct_def.fields.vec.begin();
        it != struct_def.fields.vec.end(); ++it) {
        auto &field = **it;
        if (field.deprecated) continue;

        GenStructAccessor(struct_def, field, code_ptr);
      }

      if (struct_def.fixed) {
        // create a struct constructor function
        GenStructBuilder(struct_def, code_ptr);
      }
      else {
        // Create a set of functions that allow table construction.
        GenTableBuilders(struct_def, code_ptr);
      }
    }

    // Generate enum declarations.
    void GenEnum(const EnumDef &enum_def, std::string *code_ptr) {
      if (enum_def.generated) return;

      GenComment(enum_def.doc_comment, code_ptr, &def_comment);
      BeginEnum(NormalizedName(enum_def), code_ptr);
      for (auto it = enum_def.Vals().begin(); it != enum_def.Vals().end();
           ++it) {
        auto &ev = **it;
        GenComment(ev.doc_comment, code_ptr, &def_comment, Indent);
        EnumMember(enum_def, ev, code_ptr);
      }
      EndEnum(code_ptr);
    }

    // Returns the function name that is able to read a value of the given type.
    std::string GenGetter(const Type &type) {
      switch (type.base_type) {
      case BASE_TYPE_STRING: return std::string(SelfData) + ":String(";
      case BASE_TYPE_UNION: return  std::string(SelfData) + ":Union(";
      case BASE_TYPE_VECTOR: return GenGetter(type.VectorType());
      default:
        return std::string(SelfData) + ":Get(flatbuffers.N." +
          MakeCamel(GenTypeGet(type)) + ", ";
      }
    }

    // Returns the method name for use with add/put calls.
    std::string GenMethod(const FieldDef &field) {
      return IsScalar(field.value.type.base_type)
        ? MakeCamel(GenTypeBasic(field.value.type))
        : (IsStruct(field.value.type) ? "Struct" : "UOffsetTRelative");
    }

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

    std::string GenTypePointer(const Type &type) {
      switch (type.base_type) {
      case BASE_TYPE_STRING: return "string";
      case BASE_TYPE_VECTOR: return GenTypeGet(type.VectorType());
      case BASE_TYPE_STRUCT: return type.struct_def->name;
      case BASE_TYPE_UNION:
        // fall through
      default: return "*flatbuffers.Table";
      }
    }

    std::string GenTypeGet(const Type &type) {
      return IsScalar(type.base_type) ? GenTypeBasic(type) : GenTypePointer(type);
    }

    std::string GetNamespace(const Type &type) {
      return type.struct_def->defined_namespace->GetFullyQualifiedName(type.struct_def->name);
    }

    std::string TypeName(const FieldDef &field) {
      return GenTypeGet(field.value.type);
    }

    std::string TypeNameWithNamespace(const FieldDef &field) {
      return GetNamespace(field.value.type);
    }

    // Create a struct with a builder and the struct's arguments.
    void GenStructBuilder(const StructDef &struct_def,
      std::string *code_ptr) {
      BeginBuilderArgs(struct_def, code_ptr);
      StructBuilderArgs(struct_def, "", code_ptr);
      EndBuilderArgs(code_ptr);

      StructBuilderBody(struct_def, "", code_ptr);
      EndBuilderBody(code_ptr);
    }

    bool generate() {
      if (!generateEnums()) return false;
      if (!generateStructs()) return false;
      return true;
    }

  private:
    bool generateEnums() {
      for (auto it = parser_.enums_.vec.begin(); it != parser_.enums_.vec.end();
        ++it) {
        auto &enum_def = **it;
        std::string enumcode;
        GenEnum(enum_def, &enumcode);
        if (!SaveType(enum_def, enumcode, false)) return false;
      }
      return true;
    }

    bool generateStructs() {
      for (auto it = parser_.structs_.vec.begin();
        it != parser_.structs_.vec.end(); ++it) {
        auto &struct_def = **it;
        std::string declcode;
        GenStruct(struct_def, &declcode);
        if (!SaveType(struct_def, declcode, true)) return false;
      }
      return true;
    }

    // Begin by declaring namespace and imports.
    void BeginFile(const std::string &name_space_name, const bool needs_imports,
      std::string *code_ptr) {
      std::string &code = *code_ptr;
      code += std::string(Comment) + FlatBuffersGeneratedWarning() + "\n\n";
      code += std::string(Comment) + "namespace: " + name_space_name + "\n\n";
      if (needs_imports) {
        code += "local flatbuffers = require('flatbuffers')\n\n";
      }
    }

    // Save out the generated code for a Lua Table type.
    bool SaveType(const Definition &def, const std::string &classcode,
      bool needs_imports) {
      if (!classcode.length()) return true;

      std::string namespace_dir = path_;
      auto &namespaces = def.defined_namespace->components;
      for (auto it = namespaces.begin(); it != namespaces.end(); ++it) {
        if (it != namespaces.begin()) namespace_dir += kPathSeparator;
        namespace_dir += *it;
        //std::string init_py_filename = namespace_dir + "/__init__.py";
        //SaveFile(init_py_filename.c_str(), "", false);
      }

      std::string code = "";
      BeginFile(LastNamespacePart(*def.defined_namespace), needs_imports, &code);
      code += classcode;
      code += "\n";
      code += "return " + NormalizedName(def) + " " + Comment + "return the module";
      std::string filename =
        NamespaceDir(*def.defined_namespace) + NormalizedName(def) + ".lua";
      return SaveFile(filename.c_str(), code, false);
    }
  private:
    std::unordered_set<std::string> keywords_;
  };

}  // namespace lua

bool GenerateLua(const Parser &parser, const std::string &path,
  const std::string &file_name) {
  lua::LuaGenerator generator(parser, path, file_name);
  return generator.generate();
}

}  // namespace flatbuffers
