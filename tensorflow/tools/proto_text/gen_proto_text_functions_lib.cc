/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/proto_text/gen_proto_text_functions_lib.h"

#include <algorithm>
#include <set>
#include <unordered_set>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

using ::tensorflow::protobuf::Descriptor;
using ::tensorflow::protobuf::EnumDescriptor;
using ::tensorflow::protobuf::FieldDescriptor;
using ::tensorflow::protobuf::FieldOptions;
using ::tensorflow::protobuf::FileDescriptor;

namespace tensorflow {

namespace {

template <typename... Args>
string StrCat(const Args&... args) {
  std::ostringstream s;
  std::vector<int> give_me_a_name{((s << args), 0)...};
  return s.str();
}

template <typename... Args>
string StrAppend(string* to_append, const Args&... args) {
  *to_append += StrCat(args...);
  return *to_append;
}

// Class used to generate the code for proto text functions. One of these should
// be created for each FileDescriptor whose code should be generated.
//
// This class has a notion of the current output Section.  The Print, Nested,
// and Unnest functions apply their operations to the current output section,
// which can be toggled with SetOutput.
//
// Note that on the generated code, various pieces are not optimized - for
// example: map input and output, Cord input and output, comparisons against
// the field names (it's a loop over all names), and tracking of has_seen.
class Generator {
 public:
  explicit Generator(const string& tf_header_prefix)
      : tf_header_prefix_(tf_header_prefix),
        header_(&code_.header),
        header_impl_(&code_.header_impl),
        cc_(&code_.cc) {}

  void Generate(const FileDescriptor& fd);

  // The generated code; valid after Generate has been called.
  ProtoTextFunctionCode code() const { return code_; }

 private:
  struct Section {
    explicit Section(string* str) : str(str) {}
    string* str;
    string indent;
  };

  // Switches the currently active section to <section>.
  Generator& SetOutput(Section* section) {
    cur_ = section;
    return *this;
  }

  // Increases indent level.  Returns <*this>, to allow chaining.
  Generator& Nest() {
    StrAppend(&cur_->indent, "  ");
    return *this;
  }

  // Decreases indent level.  Returns <*this>, to allow chaining.
  Generator& Unnest() {
    cur_->indent = cur_->indent.substr(0, cur_->indent.size() - 2);
    return *this;
  }

  // Appends the concatenated args, with a trailing newline. Returns <*this>, to
  // allow chaining.
  template <typename... Args>
  Generator& Print(Args... args) {
    StrAppend(cur_->str, cur_->indent, args..., "\n");
    return *this;
  }

  // Appends the print code for a single field's value.
  // If <omit_default> is true, then the emitted code will not print zero-valued
  // values.
  // <field_expr> is code that when emitted yields the field's value.
  void AppendFieldValueAppend(const FieldDescriptor& field,
                              const bool omit_default,
                              const string& field_expr);

  // Appends the print code for as single field.
  void AppendFieldAppend(const FieldDescriptor& field);

  // Appends the print code for a message. May change which section is currently
  // active.
  void AppendDebugStringFunctions(const Descriptor& md);

  // Appends the print and parse functions for an enum. May change which
  // section is currently active.
  void AppendEnumFunctions(const EnumDescriptor& enum_d);

  // Appends the parse functions for a message. May change which section is
  // currently active.
  void AppendParseMessageFunction(const Descriptor& md);

  // Appends all functions for a message and its nested message and enum types.
  // May change which section is currently active.
  void AppendMessageFunctions(const Descriptor& md);

  // Appends lines to open or close namespace declarations.
  void AddNamespaceToCurrentSection(const string& package, bool open);

  // Appends the given headers as sorted #include lines.
  void AddHeadersToCurrentSection(const std::vector<string>& headers);

  // When adding #includes for tensorflow headers, prefix them with this.
  const string tf_header_prefix_;
  ProtoTextFunctionCode code_;
  Section* cur_ = nullptr;
  Section header_;
  Section header_impl_;
  Section cc_;

  std::unordered_set<string> map_append_signatures_included_;

  TF_DISALLOW_COPY_AND_ASSIGN(Generator);
};

// Returns the prefix needed to reference objects defined in <fd>. E.g.
// "::tensorflow::test".
string GetPackageReferencePrefix(const FileDescriptor* fd) {
  string result = "::";
  const string& package = fd->package();
  for (size_t i = 0; i < package.size(); ++i) {
    if (package[i] == '.') {
      result += "::";
    } else {
      result += package[i];
    }
  }
  result += "::";
  return result;
}

// Returns the name of the class generated by proto to represent <d>.
string GetClassName(const Descriptor& d) {
  if (d.containing_type() == nullptr) return d.name();
  return StrCat(GetClassName(*d.containing_type()), "_", d.name());
}

// Returns the name of the class generated by proto to represent <ed>.
string GetClassName(const EnumDescriptor& ed) {
  if (ed.containing_type() == nullptr) return ed.name();
  return StrCat(GetClassName(*ed.containing_type()), "_", ed.name());
}

// Returns the qualified name that refers to the class generated by proto to
// represent <d>.
string GetQualifiedName(const Descriptor& d) {
  return StrCat(GetPackageReferencePrefix(d.file()), GetClassName(d));
}

// Returns the qualified name that refers to the class generated by proto to
// represent <ed>.
string GetQualifiedName(const EnumDescriptor& d) {
  return StrCat(GetPackageReferencePrefix(d.file()), GetClassName(d));
}

// Returns the qualified name that refers to the generated
// AppendProtoDebugString function for <d>.
string GetQualifiedAppendFn(const Descriptor& d) {
  return StrCat(GetPackageReferencePrefix(d.file()),
                "internal::AppendProtoDebugString");
}

// Returns the name of the generated function that returns an enum value's
// string value.
string GetEnumNameFn(const EnumDescriptor& enum_d) {
  return StrCat("EnumName_", GetClassName(enum_d));
}

// Returns the qualified name of the function returned by GetEnumNameFn().
string GetQualifiedEnumNameFn(const EnumDescriptor& enum_d) {
  return StrCat(GetPackageReferencePrefix(enum_d.file()),
                GetEnumNameFn(enum_d));
}

// Returns the name of a generated header file, either the public api (if impl
// is false) or the internal implementation header (if impl is true).
string GetProtoTextHeaderName(const FileDescriptor& fd, bool impl) {
  const int dot_index = fd.name().find_last_of('.');
  return fd.name().substr(0, dot_index) +
         (impl ? ".pb_text-impl.h" : ".pb_text.h");
}

// Returns the name of the header generated by the proto library for <fd>.
string GetProtoHeaderName(const FileDescriptor& fd) {
  const int dot_index = fd.name().find_last_of('.');
  return fd.name().substr(0, dot_index) + ".pb.h";
}

// Returns the C++ class name for the given proto field.
string GetCppClass(const FieldDescriptor& d) {
  string cpp_class = d.cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE
                         ? GetQualifiedName(*d.message_type())
                         : d.cpp_type_name();

  // In open-source TensorFlow, the definition of int64 varies across
  // platforms. The following line, which is manipulated during internal-
  // external sync'ing, takes care of the variability.
  if (cpp_class == "int64") {
    cpp_class = kProtobufInt64Typename;
  }

  return cpp_class;
}

// Returns the string that can be used for a header guard for the generated
// headers for <fd>, either for the public api (if impl is false) or the
// internal implementation header (if impl is true).
string GetHeaderGuard(const FileDescriptor& fd, bool impl) {
  string s = fd.name();
  std::replace(s.begin(), s.end(), '/', '_');
  std::replace(s.begin(), s.end(), '.', '_');
  return s + (impl ? "_IMPL_H_" : "_H_");
}

void Generator::AppendFieldValueAppend(const FieldDescriptor& field,
                                       const bool omit_default,
                                       const string& field_expr) {
  // This does not emit code with proper presence semantics (e.g. it doesn't
  // check 'has' fields on non-messages).
  CHECK(!field.has_presence() || field.containing_oneof() != nullptr ||
        field.cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE)
      << field.file()->name();

  SetOutput(&cc_);

  switch (field.cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32:
    case FieldDescriptor::CPPTYPE_INT64:
    case FieldDescriptor::CPPTYPE_UINT32:
    case FieldDescriptor::CPPTYPE_UINT64:
    case FieldDescriptor::CPPTYPE_DOUBLE:
    case FieldDescriptor::CPPTYPE_FLOAT:
      Print("o->", omit_default ? "AppendNumericIfNotZero" : "AppendNumeric",
            "(\"", field.name(), "\", ", field_expr, ");");
      break;
    case FieldDescriptor::CPPTYPE_BOOL:
      Print("o->", omit_default ? "AppendBoolIfTrue" : "AppendBool", "(\"",
            field.name(), "\", ", field_expr, ");");
      break;
    case FieldDescriptor::CPPTYPE_STRING: {
      const auto ctype = field.options().ctype();
      CHECK(ctype == FieldOptions::CORD || ctype == FieldOptions::STRING)
          << "Unsupported ctype " << ctype;

      Print("o->", omit_default ? "AppendStringIfNotEmpty" : "AppendString",
            "(\"", field.name(), "\", ProtobufStringToString(", field_expr,
            "));");
      break;
    }
    case FieldDescriptor::CPPTYPE_ENUM:
      if (omit_default) {
        Print("if (", field_expr, " != 0) {").Nest();
      }
      Print("const char* enum_name = ",
            GetQualifiedEnumNameFn(*field.enum_type()), "(", field_expr, ");");
      Print("if (enum_name[0]) {").Nest();
      Print("o->AppendEnumName(\"", field.name(), "\", enum_name);");
      Unnest().Print("} else {").Nest();
      Print("o->AppendNumeric(\"", field.name(), "\", ", field_expr, ");");
      Unnest().Print("}");
      if (omit_default) {
        Unnest().Print("}");
      }
      break;
    case FieldDescriptor::CPPTYPE_MESSAGE:
      CHECK(!field.message_type()->options().map_entry());
      if (omit_default) {
        Print("if (msg.has_", field.name(), "()) {").Nest();
      }
      Print("o->OpenNestedMessage(\"", field.name(), "\");");
      Print(GetQualifiedAppendFn(*field.message_type()), "(o, ", field_expr,
            ");");
      Print("o->CloseNestedMessage();");
      if (omit_default) {
        Unnest().Print("}");
      }
      break;
  }
}

void Generator::AppendFieldAppend(const FieldDescriptor& field) {
  const string& name = field.name();

  if (field.is_map()) {
    Print("{").Nest();
    const auto& key_type = *field.message_type()->FindFieldByName("key");
    const auto& value_type = *field.message_type()->FindFieldByName("value");

    Print("std::vector<", key_type.cpp_type_name(), "> keys;");
    Print("for (const auto& e : msg.", name, "()) keys.push_back(e.first);");
    Print("std::stable_sort(keys.begin(), keys.end());");
    Print("for (const auto& key : keys) {").Nest();
    Print("o->OpenNestedMessage(\"", name, "\");");
    AppendFieldValueAppend(key_type, false /* omit_default */, "key");
    AppendFieldValueAppend(value_type, false /* omit_default */,
                           StrCat("msg.", name, "().at(key)"));
    Print("o->CloseNestedMessage();");
    Unnest().Print("}");

    Unnest().Print("}");
  } else if (field.is_repeated()) {
    Print("for (int i = 0; i < msg.", name, "_size(); ++i) {");
    Nest();
    AppendFieldValueAppend(field, false /* omit_default */,
                           "msg." + name + "(i)");
    Unnest().Print("}");
  } else {
    const auto* oneof = field.containing_oneof();
    if (oneof != nullptr) {
      string camel_name = field.camelcase_name();
      camel_name[0] = toupper(camel_name[0]);
      Print("if (msg.", oneof->name(), "_case() == ",
            GetQualifiedName(*oneof->containing_type()), "::k", camel_name,
            ") {");
      Nest();
      AppendFieldValueAppend(field, false /* omit_default */,
                             "msg." + name + "()");
      Unnest();
      Print("}");
    } else {
      AppendFieldValueAppend(field, true /* omit_default */,
                             "msg." + name + "()");
    }
  }
}

void Generator::AppendEnumFunctions(const EnumDescriptor& enum_d) {
  const string sig = StrCat("const char* ", GetEnumNameFn(enum_d), "(\n    ",
                            GetQualifiedName(enum_d), " value)");
  SetOutput(&header_);
  Print().Print("// Enum text output for ", string(enum_d.full_name()));
  Print(sig, ";");

  SetOutput(&cc_);
  Print().Print(sig, " {");
  Nest().Print("switch (value) {").Nest();
  for (int i = 0; i < enum_d.value_count(); ++i) {
    const auto& value = *enum_d.value(i);
    Print("case ", value.number(), ": return \"", value.name(), "\";");
  }
  Print("default: return \"\";");
  Unnest().Print("}");
  Unnest().Print("}");
}

void Generator::AppendParseMessageFunction(const Descriptor& md) {
  const bool map_append = (md.options().map_entry());
  string sig;
  if (!map_append) {
    sig = StrCat("bool ProtoParseFromString(\n    const string& s,\n    ",
                 GetQualifiedName(md), "* msg)");
    SetOutput(&header_).Print(sig, "\n        TF_MUST_USE_RESULT;");

    SetOutput(&cc_);
    Print().Print(sig, " {").Nest();
    Print("msg->Clear();");
    Print("Scanner scanner(s);");
    Print("if (!internal::ProtoParseFromScanner(",
          "&scanner, false, false, msg)) return false;");
    Print("scanner.Eos();");
    Print("return scanner.GetResult();");
    Unnest().Print("}");
  }

  // Parse from scanner - the real work here.
  sig = StrCat("bool ProtoParseFromScanner(",
               "\n    ::tensorflow::strings::Scanner* scanner, bool nested, "
               "bool close_curly,\n    ");
  const FieldDescriptor* key_type = nullptr;
  const FieldDescriptor* value_type = nullptr;
  if (map_append) {
    key_type = md.FindFieldByName("key");
    value_type = md.FindFieldByName("value");
    StrAppend(&sig, "::tensorflow::protobuf::Map<", GetCppClass(*key_type),
              ", ", GetCppClass(*value_type), ">* map)");
  } else {
    StrAppend(&sig, GetQualifiedName(md), "* msg)");
  }

  if (!map_append_signatures_included_.insert(sig).second) {
    // signature for function to append to a map of this type has
    // already been defined in this .cc file. Don't define it again.
    return;
  }

  if (!map_append) {
    SetOutput(&header_impl_).Print(sig, ";");
  }

  SetOutput(&cc_);
  Print().Print("namespace internal {");
  if (map_append) {
    Print("namespace {");
  }
  Print().Print(sig, " {").Nest();
  if (map_append) {
    Print(GetCppClass(*key_type), " map_key;");
    Print("bool set_map_key = false;");
    Print(GetCppClass(*value_type), " map_value;");
    Print("bool set_map_value = false;");
  }
  Print("std::vector<bool> has_seen(", md.field_count(), ", false);");
  Print("while(true) {").Nest();
  Print("ProtoSpaceAndComments(scanner);");

  // Emit success case
  Print("if (nested && (scanner->Peek() == (close_curly ? '}' : '>'))) {")
      .Nest();
  Print("scanner->One(Scanner::ALL);");
  Print("ProtoSpaceAndComments(scanner);");
  if (map_append) {
    Print("if (!set_map_key || !set_map_value) return false;");
    Print("(*map)[map_key] = map_value;");
  }
  Print("return true;");
  Unnest().Print("}");

  Print("if (!nested && scanner->empty()) { return true; }");
  Print("scanner->RestartCapture()");
  Print("    .Many(Scanner::LETTER_DIGIT_UNDERSCORE)");
  Print("    .StopCapture();");
  Print("StringPiece identifier;");
  Print("if (!scanner->GetResult(nullptr, &identifier)) return false;");
  Print("bool parsed_colon = false;");
  Print("(void)parsed_colon;"); // Avoid "set but not used" compiler warning
  Print("ProtoSpaceAndComments(scanner);");
  Print("if (scanner->Peek() == ':') {");
  Nest().Print("parsed_colon = true;");
  Print("scanner->One(Scanner::ALL);");
  Print("ProtoSpaceAndComments(scanner);");
  Unnest().Print("}");
  for (int i = 0; i < md.field_count(); ++i) {
    const FieldDescriptor* field = md.field(i);
    const string& field_name = field->name();
    string mutable_value_expr;
    string set_value_prefix;
    if (map_append) {
      mutable_value_expr = StrCat("&map_", field_name);
      set_value_prefix = StrCat("map_", field_name, " = ");
    } else if (field->is_repeated()) {
      if (field->is_map()) {
        mutable_value_expr = StrCat("msg->mutable_", field_name, "()");
        set_value_prefix =
            "UNREACHABLE";  // generator will never use this value.
      } else {
        mutable_value_expr = StrCat("msg->add_", field_name, "()");
        set_value_prefix = StrCat("msg->add_", field_name);
      }
    } else {
      mutable_value_expr = StrCat("msg->mutable_", field_name, "()");
      set_value_prefix = StrCat("msg->set_", field_name);
    }

    Print(i == 0 ? "" : "else ", "if (identifier == \"", field_name, "\") {");
    Nest();

    if (field->is_repeated()) {
      CHECK(!map_append);

      // Check to see if this is an array assignment, like a: [1, 2, 3]
      Print("const bool is_list = (scanner->Peek() == '[');");
      Print("do {");
      // [ or , // skip
      Nest().Print("if (is_list) {");
      Nest().Print("scanner->One(Scanner::ALL);");
      Print("ProtoSpaceAndComments(scanner);");
      Unnest().Print("}");
    } else if (field->containing_oneof() != nullptr) {
      CHECK(!map_append);

      // Detect duplicate oneof value.
      const string oneof_name = field->containing_oneof()->name();
      Print("if (msg->", oneof_name, "_case() != 0) return false;");
    }

    if (!field->is_repeated() && !map_append) {
      // Detect duplicate nested repeated message.
      Print("if (has_seen[", i, "]) return false;");
      Print("has_seen[", i, "] = true;");
    }
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      Print("const char open_char = scanner->Peek();");
      Print("if (open_char != '{' && open_char != '<') return false;");
      Print("scanner->One(Scanner::ALL);");
      Print("ProtoSpaceAndComments(scanner);");
      if (field->is_map()) {
        Print("if (!ProtoParseFromScanner(");
      } else {
        Print("if (!", GetPackageReferencePrefix(field->message_type()->file()),
              "internal::ProtoParseFromScanner(");
      }
      Print("    scanner, true, open_char == '{', ", mutable_value_expr,
            ")) return false;");
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_STRING) {
      Print("string str_value;");
      Print(
          "if (!parsed_colon || "
          "!::tensorflow::strings::ProtoParseStringLiteralFromScanner(");
      Print("    scanner, &str_value)) return false;");
      Print("SetProtobufStringSwapAllowed(&str_value, ", mutable_value_expr,
            ");");
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_ENUM) {
      Print("StringPiece value;");
      Print(
          "if (!parsed_colon || "
          "!scanner->RestartCapture().Many("
          "Scanner::LETTER_DIGIT_DASH_UNDERSCORE)."
          "GetResult(nullptr, &value)) return false;");
      const auto* enum_d = field->enum_type();
      string value_prefix;
      if (enum_d->containing_type() == nullptr) {
        value_prefix = GetPackageReferencePrefix(enum_d->file());
      } else {
        value_prefix = StrCat(GetQualifiedName(*enum_d), "_");
      }

      for (int enum_i = 0; enum_i < enum_d->value_count(); ++enum_i) {
        const auto* value_d = enum_d->value(enum_i);
        const string& value_name = value_d->name();
        string condition = StrCat("value == \"", value_name, "\"");

        Print(enum_i == 0 ? "" : "} else ", "if (", condition, ") {");
        Nest();
        Print(set_value_prefix, "(", value_prefix, value_name, ");");
        Unnest();
      }
      Print("} else {");
      Nest();
      // Proto3 allows all numeric values.
      Print("int32 int_value;");
      Print("if (strings::SafeStringToNumeric(value, &int_value)) {");
      Nest();
      Print(set_value_prefix, "(static_cast<", GetQualifiedName(*enum_d),
            ">(int_value));");
      Unnest();
      Print("} else {").Nest().Print("return false;").Unnest().Print("}");
      Unnest().Print("}");
    } else {
      Print(field->cpp_type_name(), " value;");
      switch (field->cpp_type()) {
        case FieldDescriptor::CPPTYPE_INT32:
        case FieldDescriptor::CPPTYPE_INT64:
        case FieldDescriptor::CPPTYPE_UINT32:
        case FieldDescriptor::CPPTYPE_UINT64:
        case FieldDescriptor::CPPTYPE_DOUBLE:
        case FieldDescriptor::CPPTYPE_FLOAT:
          Print(
              "if (!parsed_colon || "
              "!::tensorflow::strings::ProtoParseNumericFromScanner(",
              "scanner, &value)) return false;");
          break;
        case FieldDescriptor::CPPTYPE_BOOL:
          Print(
              "if (!parsed_colon || "
              "!::tensorflow::strings::ProtoParseBoolFromScanner(",
              "scanner, &value)) return false;");
          break;
        default:
          LOG(FATAL) << "handled earlier";
      }
      Print(set_value_prefix, "(value);");
    }

    if (field->is_repeated()) {
      Unnest().Print("} while (is_list && scanner->Peek() == ',');");
      Print(
          "if (is_list && "
          "!scanner->OneLiteral(\"]\").GetResult()) return false;");
    }
    if (map_append) {
      Print("set_map_", field_name, " = true;");
    }
    Unnest().Print("}");
  }
  Unnest().Print("}");
  Unnest().Print("}");
  Unnest().Print();
  if (map_append) {
    Print("}  // namespace");
  }
  Print("}  // namespace internal");
}

void Generator::AppendDebugStringFunctions(const Descriptor& md) {
  SetOutput(&header_impl_).Print();
  SetOutput(&header_).Print().Print("// Message-text conversion for ",
                                    string(md.full_name()));

  // Append the two debug string functions for <md>.
  for (int short_pass = 0; short_pass < 2; ++short_pass) {
    const bool short_debug = (short_pass == 1);

    // Make the Get functions.
    const string sig = StrCat(
        "string ", short_debug ? "ProtoShortDebugString" : "ProtoDebugString",
        "(\n    const ", GetQualifiedName(md), "& msg)");
    SetOutput(&header_).Print(sig, ";");

    SetOutput(&cc_);
    Print().Print(sig, " {").Nest();
    Print("string s;");
    Print("::tensorflow::strings::ProtoTextOutput o(&s, ",
          short_debug ? "true" : "false", ");");
    Print("internal::AppendProtoDebugString(&o, msg);");
    Print("o.CloseTopMessage();");
    Print("return s;");
    Unnest().Print("}");
  }

  // Make the Append function.
  const string sig =
      StrCat("void AppendProtoDebugString(\n",
             "    ::tensorflow::strings::ProtoTextOutput* o,\n    const ",
             GetQualifiedName(md), "& msg)");
  SetOutput(&header_impl_).Print(sig, ";");
  SetOutput(&cc_);
  Print().Print("namespace internal {").Print();
  Print(sig, " {").Nest();
  std::vector<const FieldDescriptor*> fields;
  fields.reserve(md.field_count());
  for (int i = 0; i < md.field_count(); ++i) {
    fields.push_back(md.field(i));
  }
  std::sort(fields.begin(), fields.end(),
            [](const FieldDescriptor* left, const FieldDescriptor* right) {
              return left->number() < right->number();
            });

  for (const FieldDescriptor* field : fields) {
    SetOutput(&cc_);
    AppendFieldAppend(*field);
  }
  Unnest().Print("}").Print().Print("}  // namespace internal");
}

void Generator::AppendMessageFunctions(const Descriptor& md) {
  if (md.options().map_entry()) {
    // The 'map entry' Message is not a user-visible message type.  Only its
    // parse function is created (and that actually parsed the whole Map, not
    // just the map entry). Printing of a map is done in the code generated for
    // the containing message.
    AppendParseMessageFunction(md);
    return;
  }

  // Recurse before adding the main message function, so that internal
  // map_append functions are available before they are needed.
  for (int i = 0; i < md.enum_type_count(); ++i) {
    AppendEnumFunctions(*md.enum_type(i));
  }
  for (int i = 0; i < md.nested_type_count(); ++i) {
    AppendMessageFunctions(*md.nested_type(i));
  }

  AppendDebugStringFunctions(md);
  AppendParseMessageFunction(md);
}

void Generator::AddNamespaceToCurrentSection(const string& package, bool open) {
  Print();
  std::vector<string> parts = {""};
  for (size_t i = 0; i < package.size(); ++i) {
    if (package[i] == '.') {
      parts.resize(parts.size() + 1);
    } else {
      parts.back() += package[i];
    }
  }
  if (open) {
    for (const auto& p : parts) {
      Print("namespace ", p, " {");
    }
  } else {
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
      Print("}  // namespace ", *it);
    }
  }
}

void Generator::AddHeadersToCurrentSection(const std::vector<string>& headers) {
  std::vector<string> sorted = headers;
  std::sort(sorted.begin(), sorted.end());
  for (const auto& h : sorted) {
    Print("#include \"", h, "\"");
  }
}

// Adds to <all_fd> and <all_d> with all descriptors recursively
// reachable from the given descriptor.
void GetAllFileDescriptorsFromFile(const FileDescriptor* fd,
                                   std::set<const FileDescriptor*>* all_fd,
                                   std::set<const Descriptor*>* all_d);

// Adds to <all_fd> and <all_d> with all descriptors recursively
// reachable from the given descriptor.
void GetAllFileDescriptorsFromMessage(const Descriptor* d,
                                      std::set<const FileDescriptor*>* all_fd,
                                      std::set<const Descriptor*>* all_d) {
  if (!all_d->insert(d).second) return;
  GetAllFileDescriptorsFromFile(d->file(), all_fd, all_d);
  for (int i = 0; i < d->field_count(); ++i) {
    auto* f = d->field(i);
    switch (f->cpp_type()) {
      case FieldDescriptor::CPPTYPE_INT32:
      case FieldDescriptor::CPPTYPE_INT64:
      case FieldDescriptor::CPPTYPE_UINT32:
      case FieldDescriptor::CPPTYPE_UINT64:
      case FieldDescriptor::CPPTYPE_DOUBLE:
      case FieldDescriptor::CPPTYPE_FLOAT:
      case FieldDescriptor::CPPTYPE_BOOL:
      case FieldDescriptor::CPPTYPE_STRING:
        break;
      case FieldDescriptor::CPPTYPE_MESSAGE:
        GetAllFileDescriptorsFromMessage(f->message_type(), all_fd, all_d);
        break;
      case FieldDescriptor::CPPTYPE_ENUM:
        GetAllFileDescriptorsFromFile(f->enum_type()->file(), all_fd, all_d);
        break;
    }
  }
  for (int i = 0; i < d->nested_type_count(); ++i) {
    GetAllFileDescriptorsFromMessage(d->nested_type(i), all_fd, all_d);
  }
}

void GetAllFileDescriptorsFromFile(const FileDescriptor* fd,
                                   std::set<const FileDescriptor*>* all_fd,
                                   std::set<const Descriptor*>* all_d) {
  if (!all_fd->insert(fd).second) return;
  for (int i = 0; i < fd->message_type_count(); ++i) {
    GetAllFileDescriptorsFromMessage(fd->message_type(i), all_fd, all_d);
  }
}

void Generator::Generate(const FileDescriptor& fd) {
  const string package = fd.package();
  std::set<const FileDescriptor*> all_fd;
  std::set<const Descriptor*> all_d;
  GetAllFileDescriptorsFromFile(&fd, &all_fd, &all_d);

  std::vector<string> headers;

  // Add header to header file.
  SetOutput(&header_);
  Print("// GENERATED FILE - DO NOT MODIFY");
  Print("#ifndef ", GetHeaderGuard(fd, false /* impl */));
  Print("#define ", GetHeaderGuard(fd, false /* impl */));
  Print();
  headers = {
      GetProtoHeaderName(fd),
      StrCat(tf_header_prefix_, "tensorflow/core/platform/macros.h"),
      StrCat(tf_header_prefix_, "tensorflow/core/platform/protobuf.h"),
      StrCat(tf_header_prefix_, "tensorflow/core/platform/types.h"),
  };
  for (const auto& h : headers) {
    Print("#include \"", h, "\"");
  }
  AddNamespaceToCurrentSection(package, true /* is_open */);

  // Add header to impl file.
  SetOutput(&header_impl_);
  Print("// GENERATED FILE - DO NOT MODIFY");
  Print("#ifndef ", GetHeaderGuard(fd, true /* impl */));
  Print("#define ", GetHeaderGuard(fd, true /* impl */));
  Print();
  headers = {
      GetProtoTextHeaderName(fd, false /* impl */),
      StrCat(tf_header_prefix_,
             "tensorflow/core/lib/strings/proto_text_util.h"),
      StrCat(tf_header_prefix_, "tensorflow/core/lib/strings/scanner.h"),
  };
  for (const FileDescriptor* d : all_fd) {
    if (d != &fd) {
      headers.push_back(GetProtoTextHeaderName(*d, true /* impl */));
    }
    headers.push_back(GetProtoHeaderName(*d));
  }
  AddHeadersToCurrentSection(headers);
  AddNamespaceToCurrentSection(package, true /* is_open */);
  SetOutput(&header_impl_).Print().Print("namespace internal {");

  // Add header to cc file.
  SetOutput(&cc_);
  Print("// GENERATED FILE - DO NOT MODIFY");
  Print();
  Print("#include <algorithm>");  // for `std::stable_sort()`
  Print();
  headers = {GetProtoTextHeaderName(fd, true /* impl */)};
  AddHeadersToCurrentSection(headers);
  Print();
  Print("using ::tensorflow::strings::ProtoSpaceAndComments;");
  Print("using ::tensorflow::strings::Scanner;");
  Print("using ::tensorflow::strings::StrCat;");
  AddNamespaceToCurrentSection(package, true /* is_open */);

  // Add declarations and definitions.
  for (int i = 0; i < fd.enum_type_count(); ++i) {
    AppendEnumFunctions(*fd.enum_type(i));
  }
  for (int i = 0; i < fd.message_type_count(); ++i) {
    AppendMessageFunctions(*fd.message_type(i));
  }

  // Add footer to header file.
  SetOutput(&header_);
  AddNamespaceToCurrentSection(package, false /* is_open */);
  Print().Print("#endif  // ", GetHeaderGuard(fd, false /* impl */));

  // Add footer to header impl file.
  SetOutput(&header_impl_).Print().Print("}  // namespace internal");
  AddNamespaceToCurrentSection(package, false /* is_open */);
  Print().Print("#endif  // ", GetHeaderGuard(fd, true /* impl */));

  // Add footer to cc file.
  SetOutput(&cc_);
  AddNamespaceToCurrentSection(package, false /* is_open */);
}

}  // namespace

ProtoTextFunctionCode GetProtoTextFunctionCode(const FileDescriptor& fd,
                                               const string& tf_header_prefix) {
  Generator gen(tf_header_prefix);
  gen.Generate(fd);
  return gen.code();
}

}  // namespace tensorflow
