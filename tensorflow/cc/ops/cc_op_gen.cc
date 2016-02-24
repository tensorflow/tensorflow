/* Copyright 2015 Google Inc. All Rights Reserved.

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

// TODO(josh11b): Rewrite function parameter names to avoid C++ keywords
// or "opts".

#include "tensorflow/cc/ops/cc_op_gen.h"

#include <unordered_map>
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

const int kRightMargin = 79;

const char* AttrTypeName(StringPiece attr_type) {
  static const char* kAttrTypeName[][2] = {
      {"string", "StringPiece"},
      {"list(string)", "gtl::ArraySlice<string>"},
      {"int", "int64"},
      {"list(int)", "gtl::ArraySlice<int>"},
      {"float", "float"},
      {"list(float)", "gtl::ArraySlice<float>"},
      {"bool", "bool"},
      {"list(bool)", "gtl::ArraySlice<bool>"},
      {"type", "DataType"},
      {"list(type)", "DataTypeSlice"},
      {"shape", "TensorShape"},
      {"list(shape)", "gtl::ArraySlice<TensorShape>"},
      {"tensor", "const Tensor&"},
      {"list(tensor)", "gtl::ArraySlice<Tensor>"},
      {"func", "const NameAttrList&"},
  };
  for (size_t i = 0; i < TF_ARRAYSIZE(kAttrTypeName); ++i) {
    if (attr_type == kAttrTypeName[i][0]) {
      return kAttrTypeName[i][1];
    }
  }
  LOG(FATAL) << "Unsupported Attr type: " << attr_type;
  return "";
}

// Change:     Into:
//   ABC         // ABC
//               //
//   DEF         // DEF
string MakeComment(StringPiece text) {
  string ret;
  while (!text.empty()) {
    int last_non_space = -1;
    int newline;
    for (newline = 0; newline < static_cast<int>(text.size()); ++newline) {
      if (text[newline] == '\n') break;
      if (text[newline] != ' ') last_non_space = newline;
    }
    if (last_non_space == -1) {
      strings::StrAppend(&ret, "//\n");
    } else {
      strings::StrAppend(&ret, "// ", text.substr(0, last_non_space + 1), "\n");
    }
    text.remove_prefix(newline + 1);
  }
  return ret;
}

void WriteCCOp(const OpDef& op_def, WritableFile* h, WritableFile* cc) {
  // TODO(josh11b): Better wrapping of comments.
  string comment;
  if (op_def.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(op_def.summary(), "\n");
    if (!op_def.description().empty()) {
      strings::StrAppend(&comment, "\n", op_def.description(), "\n");
    }
  }

  static const string kSingleInputType = "NodeOut";
  static const string kListInputType = "gtl::ArraySlice<NodeOut>";

  std::vector<string> arg_types;
  std::vector<string> arg_names;

  strings::StrAppend(&comment, "\nArguments:\n");

  // Map from attr name to the first input arg it is inferred from.
  std::unordered_map<string, string> inferred_attrs;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    arg_names.emplace_back(arg.name());
    bool is_list = false;

    if (!arg.type_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs, arg.type_attr(), arg.name());
    } else if (!arg.type_list_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs, arg.type_list_attr(),
                              arg.name());
      is_list = true;
    }
    if (!arg.number_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs, arg.number_attr(), arg.name());
      is_list = true;
    }
    if (is_list) {
      arg_types.emplace_back(kListInputType);
    } else {
      arg_types.emplace_back(kSingleInputType);
    }

    // TODO(josh11b): Include input type information.
    StringPiece description = arg.description();
    if (!description.empty()) {
      ConsumeEquals(&description);
      strings::StrAppend(&comment, "* ", arg_names.back(), ": ",
                         arg.description(), "\n");
    }
  }

  string options_comment;
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    // Do not add inferred attrs or attrs with defaults to the C++
    // function signature.
    if (inferred_attrs.find(attr.name()) == inferred_attrs.end()) {
      if (!attr.has_default_value()) {
        arg_names.emplace_back(attr.name());
        arg_types.emplace_back(AttrTypeName(attr.type()));
        if (!attr.description().empty()) {
          strings::StrAppend(&comment, "* ", arg_names.back(), ": ",
                             attr.description(), "\n");
        }
      } else {
        strings::StrAppend(&options_comment, "  .WithAttr(\"", attr.name(),
                           "\", ", AttrTypeName(attr.type()), "): Defaults to ",
                           SummarizeAttrValue(attr.default_value()), ".\n");
        if (!attr.description().empty()) {
          strings::StrAppend(&options_comment, "    ", attr.description(),
                             "\n");
        }
      }
    }
  }
  CHECK_EQ(arg_names.size(), arg_types.size());
  strings::StrAppend(&comment, "* opts:\n", options_comment,
                     R"comment(  .WithName(StringPiece): Set the Node's name
  .WithDevice(StringPiece): Set the Node's requested device
  .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
    Add control dependencies on the specified Node(s).

Returns a pointer to the created Node)comment");

  // TODO(josh11b): Include output type information.
  if (op_def.output_arg_size() == 0) {
    strings::StrAppend(&comment, ".\n");
  } else if (op_def.output_arg_size() == 1) {
    StringPiece description = op_def.output_arg(0).description();
    ConsumeEquals(&description);
    if (description.empty()) {
      strings::StrAppend(&comment, ".\n");
    } else {
      strings::StrAppend(&comment, ", with output:\n", description, "\n");
    }
  } else {
    strings::StrAppend(&comment, ", with outputs:\n");
    for (int o = 0; o < op_def.output_arg_size(); ++o) {
      StringPiece description = op_def.output_arg(o).description();
      ConsumeEquals(&description);
      if (description.empty()) {
        strings::StrAppend(&comment, "* ", op_def.output_arg(o).name(), "\n");
      } else {
        strings::StrAppend(&comment, "* ", op_def.output_arg(o).name(), ": ",
                           description, "\n");
      }
    }
  }

  // Write the header comment.
  TF_CHECK_OK(h->Append(MakeComment(comment)));

  // Declare the function wrapper.
  const string prefix = strings::StrCat("Node* ", op_def.name(), "(");
  string h_rest;
  for (size_t i = 0; i < arg_names.size(); ++i) {
    strings::StrAppend(&h_rest, arg_types[i], " ", arg_names[i], ", ");
  }
  strings::StrAppend(&h_rest, "const GraphDefBuilder::Options& opts");
  string cc_decl = h_rest;
  strings::StrAppend(&h_rest, ");");
  TF_CHECK_OK(h->Append(WordWrap(prefix, h_rest, kRightMargin) + "\n\n"));

  // Define the function wrapper.
  strings::StrAppend(&cc_decl, ") {");
  TF_CHECK_OK(cc->Append(WordWrap(prefix, cc_decl, kRightMargin) + "\n"));
  const string op_name = strings::StrCat("  static const string kOpName = \"",
                                         op_def.name(), "\";\n");

  if (arg_types.empty()) {
    TF_CHECK_OK(cc->Append(op_name));
    TF_CHECK_OK(cc->Append("  return SourceOp(kOpName, opts);\n}\n\n"));
  } else if (arg_types == std::vector<string>({kSingleInputType})) {
    TF_CHECK_OK(cc->Append(op_name));
    TF_CHECK_OK(cc->Append(strings::StrCat("  return UnaryOp(kOpName, ",
                                           arg_names[0], ", opts);\n}\n\n")));
  } else if (arg_types ==
             std::vector<string>({kSingleInputType, kSingleInputType})) {
    TF_CHECK_OK(cc->Append(op_name));
    // TODO(josh11b): Word wrap this if it ever becomes necessary.
    TF_CHECK_OK(
        cc->Append(strings::StrCat("  return BinaryOp(kOpName, ", arg_names[0],
                                   ", ", arg_names[1], ", opts);\n}\n\n")));
  } else {
    TF_CHECK_OK(cc->Append("  if (opts.HaveError()) return nullptr;\n"));
    TF_CHECK_OK(cc->Append(op_name));
    TF_CHECK_OK(cc->Append(
        "  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,\n"
        "                           opts.op_registry());\n"));
    for (size_t i = 0; i < arg_names.size(); ++i) {
      if (i < static_cast<size_t>(op_def.input_arg_size())) {
        TF_CHECK_OK(cc->Append(
            strings::StrCat("  node_builder.Input(", arg_names[i], ");\n")));
      } else {
        TF_CHECK_OK(
            cc->Append(strings::StrCat("  node_builder.Attr(\"", arg_names[i],
                                       "\", ", arg_names[i], ");\n")));
      }
    }
    TF_CHECK_OK(
        cc->Append("  return opts.FinalizeBuilder(&node_builder);\n"
                   "}\n\n"));
  }
}

// Converts:
//   bazel-out/.../genfiles/(external/YYY/)?XX
// to: XX.
string GetPath(const std::string& dot_h_fname) {
  auto pos = dot_h_fname.find("/genfiles/");
  string result = dot_h_fname;
  if (pos != string::npos) {
    // - 1 account for the terminating null character (\0) in "/genfiles/".
    result = dot_h_fname.substr(pos + sizeof("/genfiles/") - 1);
  }
  if (result.size() > sizeof("external/") &&
      result.compare(0, sizeof("external/") - 1, "external/") == 0) {
    result = result.substr(sizeof("external/") - 1);
    pos = result.find("/");
    if (pos != string::npos) {
      result = result.substr(pos + 1);
    }
  }
  return result;
}

// Converts:
//   cc/ops/gen_foo_ops.h
// to:
//   CC_OPS_GEN_FOO_OPS_H_
string ToGuard(const std::string& path) {
  string guard;
  guard.reserve(path.size() + 1);  // + 1 -> trailing _
  for (const char c : path) {
    if (c >= 'A' && c <= 'Z') {
      guard += c;
    } else if (c >= 'a' && c <= 'z') {
      guard += c + 'A' - 'a';
    } else {
      guard += '_';
    }
  }
  guard += '_';
  return guard;
}

}  // namespace

void WriteCCOps(const OpList& ops, const std::string& dot_h_fname,
                const std::string& dot_cc_fname) {
  Env* env = Env::Default();
  WritableFile* h = nullptr;
  WritableFile* cc = nullptr;
  TF_CHECK_OK(env->NewWritableFile(dot_h_fname, &h));
  TF_CHECK_OK(env->NewWritableFile(dot_cc_fname, &cc));

  // .h Header
  const string include = GetPath(dot_h_fname);
  const string guard = ToGuard(include);
  // TODO(josh11b): Mention the library for which wrappers are being generated.
  Status s;
  s = h->Append(
      strings::StrCat("// This file is MACHINE GENERATED! Do not edit.\n\n"
                      "#ifndef ",
                      guard,
                      "\n"
                      "#define ",
                      guard, R"header(

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


)header"));
  TF_CHECK_OK(s);
  // .cc Header
  s = cc->Append(
      strings::StrCat("// This file is MACHINE GENERATED! Do not edit.\n\n"
                      "#include \"",
                      include, R"header("

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

)header"));
  TF_CHECK_OK(s);

  for (const auto& op_def : ops.op()) {
    WriteCCOp(op_def, h, cc);
  }

  // .h Footer

  s = h->Append(strings::StrCat(R"footer(}  // namespace ops
}  // namespace tensorflow

#endif  // )footer",
                                guard, "\n"));
  TF_CHECK_OK(s);

  // .cc Footer

  s = cc->Append(R"footer(}  // namespace ops
}  // namespace tensorflow
)footer");
  TF_CHECK_OK(s);

  TF_CHECK_OK(cc->Close());
  TF_CHECK_OK(h->Close());
}

}  // namespace tensorflow
