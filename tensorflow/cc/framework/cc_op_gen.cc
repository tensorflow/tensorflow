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

#include "tensorflow/cc/framework/cc_op_gen.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/cc/framework/cc_op_gen_util.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace cc_op {
namespace {

const int kRightMargin = 79;

std::string GetConstructorDecl(const OpInfo& op_info,
                               absl::string_view op_name_prefix,
                               bool include_attr) {
  const std::string prefix = absl::StrCat(op_name_prefix, op_info.op_name, "(");
  std::string c_decl;
  for (int i = 0; i < op_info.arg_types.size(); ++i) {
    if (i > 0) absl::StrAppend(&c_decl, ", ");
    absl::StrAppend(&c_decl, op_info.arg_types[i], " ", op_info.arg_names[i]);
  }
  if (include_attr && op_info.has_optional_attrs) {
    absl::StrAppend(&c_decl, ", const ", op_info.op_name, "::Attrs& attrs");
  }
  absl::StrAppend(&c_decl, ")");
  return WordWrap(prefix, c_decl, kRightMargin);
}

void WriteClassDecl(const OpInfo& op_info, WritableFile* h) {
  std::string class_decl = op_info.comment;
  absl::StrAppend(&class_decl, "class ", op_info.op_name, " {\n");
  absl::StrAppend(&class_decl, " public:\n");
  if (op_info.has_optional_attrs) {
    absl::StrAppend(&class_decl, op_info.GetOpAttrStruct());
  }
  absl::StrAppend(&class_decl, "  ",
                  GetConstructorDecl(op_info, "", /* include_attr */ false),
                  ";\n");
  if (op_info.has_optional_attrs) {
    absl::StrAppend(&class_decl, "  ",
                    GetConstructorDecl(op_info, "", /* include_attr */ true),
                    ";\n");
  }
  if (op_info.output_types.empty()) {
    // Allow casting this class to Operation.
    absl::StrAppend(&class_decl,
                    "  operator ::tensorflow::Operation() const { "
                    "return operation; }\n");
  } else if (op_info.output_types.size() == 1) {
    if (op_info.is_list_output[0]) {
      // Write the subscript operator, allowing out[i] for the list-typed
      // output.
      absl::StrAppend(&class_decl,
                      "  ::tensorflow::Output operator[](size_t index) "
                      "const { return ",
                      op_info.output_names[0], "[index]; }\n\n");

    } else {
      // Write type cast functions, allowing casting this class to Input and
      // Output.
      absl::StrAppend(&class_decl,
                      "  operator ::tensorflow::Output() const { return ",
                      op_info.output_names[0], "; }\n");
      absl::StrAppend(&class_decl,
                      "  operator ::tensorflow::Input() const { return ",
                      op_info.output_names[0], "; }\n");
      // Write node() to get the Node* directly.
      absl::StrAppend(&class_decl,
                      "  ::tensorflow::Node* node() const { return ",
                      op_info.output_names[0], ".node(); }\n");
    }
  }
  // Add the static functions to set optional attrs
  if (op_info.has_optional_attrs) {
    absl::StrAppend(&class_decl, "\n");
    for (int i = 0; i < op_info.graph_op_def.attr_size(); ++i) {
      const auto& attr(op_info.graph_op_def.attr(i));
      const auto& api_def_attr(op_info.api_def.attr(i));
      if ((op_info.inferred_input_attrs.find(attr.name()) !=
           op_info.inferred_input_attrs.end()) ||
          !api_def_attr.has_default_value()) {
        continue;
      }
      const auto entry = AttrTypeName(attr.type());
      const auto attr_type_name = entry.first;
      const bool use_const = entry.second;
      const std::string camel_case_name = ToCamelCase(api_def_attr.rename_to());
      const std::string suffix =
          (camel_case_name == op_info.op_name || camel_case_name == "Attrs")
              ? "_"
              : "";
      const std::string attr_func_def = strings::StrCat(
          camel_case_name, suffix, "(", use_const ? "const " : "",
          attr_type_name, use_const ? "&" : "");
      absl::StrAppend(&class_decl, "  static Attrs ", attr_func_def, " x) {\n");
      absl::StrAppend(&class_decl, "    return Attrs().", camel_case_name,
                      suffix, "(x);\n");
      absl::StrAppend(&class_decl, "  }\n");
    }
  }

  absl::StrAppend(&class_decl, "\n  Operation operation;\n");
  for (int i = 0; i < op_info.output_types.size(); ++i) {
    strings::StrAppend(&class_decl, "  ", op_info.output_types[i], " ",
                       op_info.output_names[i], ";\n");
  }

  absl::StrAppend(&class_decl, "};\n");
  if (!op_info.aliases.empty()) {
    for (const auto& alias : op_info.aliases) {
      strings::StrAppend(&class_decl, "typedef ", op_info.op_name, " ", alias,
                         ";\n");
    }
  }
  absl::StrAppend(&class_decl, "\n");
  TF_CHECK_OK(h->Append(class_decl));
}

void GetOutput(const OpInfo& op_info, std::string* out) {
  const std::string scope_str = op_info.arg_names[0];
  std::string return_on_error =
      absl::StrCat("if (!", scope_str, ".ok()) return;");

  absl::StrAppend(out, "  this->operation = Operation(ret);\n");

  // No outputs.
  if (op_info.graph_op_def.output_arg_size() == 0) {
    absl::StrAppend(out, "  return;\n");
    return;
  }
  if (op_info.graph_op_def.output_arg_size() == 1) {
    // One output, no need for NameRangeMap
    if (op_info.is_list_output[0]) {
      absl::StrAppend(out,
                      "  for (int32 i = 0; i < ret->num_outputs(); ++i)\n");
      absl::StrAppend(out, "    this->", op_info.output_names[0],
                      ".push_back(Output(ret, i));\n");
    } else {
      absl::StrAppend(out, "  this->", op_info.output_names[0],
                      " = Output(ret, 0);\n");
    }
    return;
  }
  absl::StrAppend(out, "  ::tensorflow::NameRangeMap _outputs_range;\n");
  absl::StrAppend(out,
                  "  ::tensorflow::Status _status_ = "
                  "::tensorflow::NameRangesForNode(*ret, ret->op_def(), "
                  "nullptr, &_outputs_range);\n");
  strings::StrAppend(out, "  if (!_status_.ok()) {\n", "    ", scope_str,
                     ".UpdateStatus(_status_);\n", "    return;\n");
  absl::StrAppend(out, "  }\n\n");

  for (int i = 0; i < op_info.graph_op_def.output_arg_size(); ++i) {
    const std::string arg_range = absl::StrCat(
        "_outputs_range[\"", op_info.graph_op_def.output_arg(i).name(), "\"]");
    if (op_info.is_list_output[i]) {
      strings::StrAppend(out, "  for (int32 i = ", arg_range, ".first; i < ",
                         arg_range, ".second; ++i)\n");
      absl::StrAppend(out, "    this->", op_info.output_names[i],
                      ".push_back(Output(ret, i));\n");
    } else {
      strings::StrAppend(out, "  this->", op_info.output_names[i],
                         " = Output(ret, ", arg_range, ".first);\n");
    }
  }
}

std::string GetConstructorBody(const OpInfo& op_info) {
  const std::string scope_str = op_info.arg_names[0];

  std::string body;
  std::string return_on_error =
      absl::StrCat("if (!", scope_str, ".ok()) return;");

  absl::StrAppend(&body, "  ", return_on_error, "\n");

  for (int i = 0; i < op_info.graph_op_def.input_arg_size(); ++i) {
    const auto& arg(op_info.graph_op_def.input_arg(i));
    const auto& api_def_arg(op_info.api_def.in_arg(i));
    strings::StrAppend(
        &body, "  auto _", api_def_arg.rename_to(), " = ::tensorflow::ops::",
        ArgIsList(arg) ? "AsNodeOutList" : "AsNodeOut", "(", scope_str, ", ",
        AvoidCPPKeywords(api_def_arg.rename_to()), ");\n");
    absl::StrAppend(&body, "  ", return_on_error, "\n");
  }

  absl::StrAppend(&body, "  ::tensorflow::Node* ret;\n");
  strings::StrAppend(&body, "  const auto unique_name = ", scope_str,
                     ".GetUniqueNameForOp(\"", op_info.op_name, "\");\n");
  absl::StrAppend(&body,
                  "  auto builder = ::tensorflow::NodeBuilder(unique_name, \"",
                  op_info.graph_op_def.name(), "\")\n");
  const std::string spaces = "                     ";
  for (int i = 0; i < op_info.api_def.in_arg_size(); ++i) {
    const auto& arg(op_info.api_def.in_arg(i));
    absl::StrAppend(&body, spaces, ".Input(_", arg.rename_to(), ")\n");
  }
  for (int i = 0; i < op_info.api_def.attr_size(); ++i) {
    const auto& graph_attr(op_info.graph_op_def.attr(i));
    const auto& api_def_attr(op_info.api_def.attr(i));
    if (op_info.inferred_input_attrs.find(api_def_attr.name()) !=
        op_info.inferred_input_attrs.end()) {
      continue;
    }
    const std::string attr_name =
        api_def_attr.has_default_value()
            ? absl::StrCat("attrs.", api_def_attr.rename_to(), "_")
            : AvoidCPPKeywords(api_def_attr.rename_to());
    strings::StrAppend(&body, spaces, ".Attr(\"", graph_attr.name(), "\", ",
                       attr_name, ")\n");
  }
  absl::StrAppend(&body, "  ;\n");
  absl::StrAppend(&body, "  ", scope_str, ".UpdateBuilder(&builder);\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(builder.Finalize(",
                     scope_str, ".graph(), &ret));\n");
  absl::StrAppend(&body, "  ", return_on_error, "\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(", scope_str,
                     ".DoShapeInference(ret));\n");

  GetOutput(op_info, &body);
  return body;
}

void WriteClassDef(const OpInfo& op_info, WritableFile* cc) {
  std::string class_def;
  absl::StrAppend(
      &class_def,
      GetConstructorDecl(op_info, strings::StrCat(op_info.op_name, "::"),
                         /* include_attr */ true),
      " {\n");
  absl::StrAppend(&class_def, GetConstructorBody(op_info));
  absl::StrAppend(&class_def, "}\n\n");

  if (op_info.has_optional_attrs) {
    absl::StrAppend(
        &class_def,
        GetConstructorDecl(op_info, strings::StrCat(op_info.op_name, "::"),
                           /* include_attr */ false));
    absl::StrAppend(&class_def, "\n  : ", op_info.op_name, "(");
    int i = 0;
    for (; i < op_info.arg_names.size(); ++i) {
      if (i > 0) absl::StrAppend(&class_def, ", ");
      absl::StrAppend(&class_def, op_info.arg_names[i]);
    }
    if (i > 0) absl::StrAppend(&class_def, ", ");
    absl::StrAppend(&class_def, op_info.op_name, "::Attrs()");
    absl::StrAppend(&class_def, ") {}\n\n");
  }
  TF_CHECK_OK(cc->Append(class_def));
}

void WriteCCOp(const OpDef& graph_op_def, const ApiDef& api_def,
               const std::vector<std::string>& aliases, WritableFile* h,
               WritableFile* cc) {
  OpInfo op_info(graph_op_def, api_def, aliases);

  WriteClassDecl(op_info, h);
  WriteClassDef(op_info, cc);
}

void StartFiles(bool internal, const std::string& dot_h_fname, WritableFile* h,
                WritableFile* cc, std::string* op_header_guard) {
  const std::string header =
      R"header(// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
)header";

  // TODO(keveman): Make namespaces configurable.
  const std::string namespace_begin = internal ? R"namespace(
namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

)namespace"
                                               : R"namespace(
namespace tensorflow {
namespace ops {

)namespace";

  const std::string op_header = GetPath(dot_h_fname);
  *op_header_guard = ToGuard(op_header);
  const std::string cc_header = strings::StrCat(
      R"include(// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
)include",
      "#include \"", op_header, "\"\n", namespace_begin);

  const std::string filename = GetFilename(dot_h_fname);
  const std::string doxygen = strings::StrCat(
      "/// @defgroup ", filename, " ", ToTitle(filename), "\n", "/// @{\n\n");

  TF_CHECK_OK(h->Append(
      strings::StrCat("// This file is MACHINE GENERATED! Do not edit.\n\n"
                      "#ifndef ",
                      *op_header_guard,
                      "\n"
                      "#define ",
                      *op_header_guard, "\n\n")));
  TF_CHECK_OK(h->Append(header));
  TF_CHECK_OK(h->Append(namespace_begin));
  TF_CHECK_OK(h->Append(doxygen));
  TF_CHECK_OK(cc->Append(cc_header));
}

void FinishFiles(bool internal, WritableFile* h, WritableFile* cc,
                 const std::string& op_header_guard) {
  const std::string footer = internal ? R"footer(}  // namespace internal
}  // namespace ops
}  // namespace tensorflow
)footer"
                                      :
                                      R"footer(/// @}

}  // namespace ops
}  // namespace tensorflow
)footer";

  TF_CHECK_OK(h->Append(footer));
  TF_CHECK_OK(
      h->Append(absl::StrCat("\n#endif  ", "// ", op_header_guard, "\n")));
  TF_CHECK_OK(cc->Append(footer));

  TF_CHECK_OK(cc->Close());
  TF_CHECK_OK(h->Close());
}

std::string MakeInternal(const std::string& fname) {
  auto dot_pos = fname.rfind('.');
  if (dot_pos == std::string::npos) {
    return absl::StrCat(fname, "_internal");
  } else {
    return absl::StrCat(fname.substr(0, dot_pos), "_internal",
                        fname.substr(dot_pos));
  }
}

}  // namespace

void WriteCCOps(const OpList& ops, const ApiDefMap& api_def_map,
                const std::string& dot_h_fname,
                const std::string& dot_cc_fname) {
  Env* env = Env::Default();

  // Write the initial boilerplate to the .h and .cc files.
  std::unique_ptr<WritableFile> h = nullptr;
  std::unique_ptr<WritableFile> cc = nullptr;
  TF_CHECK_OK(env->NewWritableFile(dot_h_fname, &h));
  TF_CHECK_OK(env->NewWritableFile(dot_cc_fname, &cc));
  std::string op_header_guard;
  StartFiles(false, dot_h_fname, h.get(), cc.get(), &op_header_guard);

  // Create the internal versions of these files for the hidden ops.
  std::unique_ptr<WritableFile> internal_h = nullptr;
  std::unique_ptr<WritableFile> internal_cc = nullptr;
  const std::string internal_dot_h_fname = MakeInternal(dot_h_fname);
  TF_CHECK_OK(env->NewWritableFile(internal_dot_h_fname, &internal_h));
  TF_CHECK_OK(env->NewWritableFile(MakeInternal(dot_cc_fname), &internal_cc));
  std::string internal_op_header_guard;
  StartFiles(true /* internal */, internal_dot_h_fname, internal_h.get(),
             internal_cc.get(), &internal_op_header_guard);

  for (const auto& graph_op_def : ops.op()) {
    // Skip deprecated ops.
    // TODO(josh11b): If needed, can put them into a "deprecated" namespace
    // instead of skipping.
    if (graph_op_def.has_deprecation() &&
        graph_op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION) {
      continue;
    }

    // We use a hand-written wrapper for "Const", since the generated
    // code depends on it.
    if (graph_op_def.name() == "Const") continue;

    const auto* api_def = api_def_map.GetApiDef(graph_op_def.name());

    std::vector<std::string> aliases;
    if (api_def->visibility() == ApiDef::SKIP) continue;
    // First endpoint is canonical, the rest are aliases.
    for (int endpoint_i = 1; endpoint_i < api_def->endpoint_size();
         ++endpoint_i) {
      aliases.push_back(api_def->endpoint(endpoint_i).name());
    }
    if (api_def->visibility() == ApiDef::HIDDEN) {
      // Write hidden ops to _internal.h and _internal.cc.
      WriteCCOp(graph_op_def, *api_def, aliases, internal_h.get(),
                internal_cc.get());
      continue;
    }
    // This isn't a hidden op, write it to the main files.
    WriteCCOp(graph_op_def, *api_def, aliases, h.get(), cc.get());
  }

  FinishFiles(false, h.get(), cc.get(), op_header_guard);
  FinishFiles(true /* internal */, internal_h.get(), internal_cc.get(),
              internal_op_header_guard);
}

}  // namespace cc_op
}  // namespace tensorflow
