/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/fuzzing/cc_op_fuzz_gen.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/framework/cc_op_gen_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace cc_op {
namespace {

std::string DefaultValue(OpDef_AttrDef attr) {
  static const auto* attr_default_value_map =
      new absl::flat_hash_map<StringPiece, StringPiece, StringPieceHasher>{
          {"int", "0"},
          {"string", "\"\""},
          {"list(int)", "{ 0, 1 }"},
          {"list(float)", "{0.0, 1.0}"},
          {"type", "DT_UINT8"},
          {"shape",
           "mediapipe::ParseTextProtoOrDie<TensorShapeProto>("
           "\"dim:[] unknown_rank:true\")"}};

  if (attr.has_minimum()) {
    if (attr.type() == "int") {
      return absl::StrCat(attr.minimum());
    } else if (attr.type() == "list(int)") {
      std::vector<int> v(attr.minimum());
      for (int i = 0; i < v.size(); ++i) v[i] = i;
      std::string s = strings::StrCat("{", absl::StrJoin(v, ","), "}");
      return s;
    }
  }
  if (attr.has_allowed_values()) {
    if (!attr.allowed_values().list().s().empty()) {
      return strings::StrCat("\"", attr.allowed_values().list().s(0), "\"");
    } else if (!attr.allowed_values().list().type().empty()) {
      return DataType_Name(attr.allowed_values().list().type(0));
    }
  }
  auto entry = attr_default_value_map->find(attr.type());
  if (entry == attr_default_value_map->end()) {
    LOG(ERROR) << "Unsupported Attr type: " << attr.type();
    return "";
  }
  return std::string(entry->second);
}

tsl::Status WriteClassFuzzDef(const OpInfo& op_info, WritableFile* fuzz) {
  string class_signature_str = absl::Substitute(
      "class Fuzz$0 : public FuzzSession<$1> {\n", op_info.op_name,
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](string* out, const auto arg) {
                      strings::StrAppend(out, "Tensor");
                      if (ArgIsList(arg)) strings::StrAppend(out, ", Tensor");
                    }));

  string build_graph_body = strings::StrCat(
      absl::StrJoin(
          op_info.graph_op_def.input_arg(), "",
          [op_info](string* out, const OpDef_ArgDef arg) {
            std::string type = "DT_UINT8";

            if (arg.type() != DT_INVALID) {
              type = DataType_Name(arg.type());
            } else if (!arg.type_attr().empty()) {
              OpDef_AttrDef attr =
                  *FindAttr(arg.type_attr(), op_info.graph_op_def);
              if (attr.has_default_value() &&
                  !attr.default_value().has_type()) {
                type = DataType_Name(attr.default_value().type());
              } else if (attr.has_allowed_values()) {
                if (attr.allowed_values().has_list() &&
                    !attr.allowed_values().list().type().empty()) {
                  type = DataType_Name(attr.allowed_values().list().type(0));
                }
              }
            }
            if (ArgIsList(arg)) {
              strings::StrAppend(
                  out, "    Input ", arg.name(),
                  "_0 = ", "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                  arg.name(), "\"), ", type, ");\n");
              strings::StrAppend(
                  out, "    Input ", arg.name(),
                  "_1 = ", "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                  arg.name(), "\"), ", type, ");\n");
              strings::StrAppend(
                  out, absl::Substitute("    InputList $0({$0_0, $0_1});\n",
                                        arg.name()));
            } else {
              strings::StrAppend(
                  out, "    auto ", arg.name(), " = ",
                  "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                  arg.name(), "\"), ", type, ");\n");
            }
          }),
      absl::StrJoin(op_info.graph_op_def.attr(), "",
                    [op_info](string* out, const OpDef_AttrDef attr) {
                      if (op_info.inferred_input_attrs.count(attr.name()) ==
                              0 &&
                          !attr.has_default_value()) {
                        strings::StrAppend(out, "    auto ", attr.name(), " = ",
                                           DefaultValue(attr), ";\n");
                      }
                    }));

  string constructor_call_str = absl::Substitute(
      "    tensorflow::ops::$0(scope.WithOpName(\"output\")$1);\n",
      op_info.op_name,
      strings::StrCat(
          op_info.api_def.arg_order().empty()
              ? absl::StrJoin(op_info.api_def.in_arg(), "",
                              [](string* out, const auto api_def_arg) {
                                strings::StrAppend(out, ", ",
                                                   api_def_arg.name());
                              })
              : absl::StrJoin(op_info.api_def.arg_order(), "",
                              [](string* out, const auto name) {
                                strings::StrAppend(out, ", ", name);
                              }),
          absl::StrJoin(op_info.graph_op_def.attr(), "",
                        [op_info](string* out, const OpDef_AttrDef attr) {
                          if (op_info.inferred_input_attrs.count(attr.name()) ==
                                  0 &&
                              !attr.has_default_value()) {
                            strings::StrAppend(out, ", ", attr.name());
                          }
                        })));

  string fuzz_impl_signature_str = absl::Substitute(
      "  void FuzzImpl($0) final {\n",
      absl::StrJoin(
          op_info.graph_op_def.input_arg(), ", ",
          [](string* out, const auto arg) {
            strings::StrAppend(out, "const Tensor& ", arg.name(), "_0");
            if (ArgIsList(arg))
              strings::StrAppend(out, ", const Tensor& ", arg.name(), "_1");
          }));

  string run_inputs_str = absl::Substitute(
      "    RunInputs({$0});\n",
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](string* out, const auto arg) {
                      if (ArgIsList(arg)) {
                        strings::StrAppend(
                            out, "{\"", arg.name(), "\", ", arg.name(), "_0}, ",
                            "{\"", arg.name(), "\", ", arg.name(), "_1}");
                      } else {
                        strings::StrAppend(out, "{\"", arg.name(), "\", ",
                                           arg.name(), "_0}");
                      }
                    }));

  string fuzz_class_def = strings::StrCat(
      class_signature_str, "  void BuildGraph(const Scope& scope) override {\n",
      build_graph_body, constructor_call_str, "  }\n", fuzz_impl_signature_str,
      run_inputs_str, "  }\n", "};\n");

  return fuzz->Append(fuzz_class_def);
}

tsl::Status WriteFuzzTest(const OpInfo& op_info, WritableFile* fuzz) {
  string fuzz_test_str = absl::Substitute(
      "FUZZ_TEST_F(Fuzz$0, Fuzz).WithDomains($1);\n", op_info.op_name,
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](string* out, const auto arg) {
                      strings::StrAppend(out, "AnyTensor()");
                      if (ArgIsList(arg))
                        strings::StrAppend(out, ", AnyTensor()");
                    }));
  return fuzz->Append(fuzz_test_str);
}

bool OpFuzzingIsOk(const OpInfo& op_info) {
  // TODO(unda, b/249347507): should we hide fuzzers for hidden ops?
  if (op_info.api_def.visibility() == ApiDef::HIDDEN) return false;

  if (op_info.api_def.visibility() == ApiDef::SKIP) return false;

  // TODO(unda) : zero input ops
  std::set<string> zero_input_ops = {"Placeholder", "ImmutableConst"};
  if (zero_input_ops.find(op_info.op_name) != zero_input_ops.end()) {
    std::cout << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " takes zero inputs.\n";
    return false;
  }

  // TODO(unda, 253431636): constrained kernel
  std::set<string> constrained_kernel = {"Diag",
                                         "DiagPart",
                                         "GatherNd",
                                         "GatherV2",
                                         "QuantizeAndDequantizeV2",
                                         "QuantizeAndDequantizeV3",
                                         "QuantizeAndDequantizeV4",
                                         "QuantizeAndDequantizeV4Grad",
                                         "QuantizedConcat",
                                         "QuantizedInstanceNorm",
                                         "QuantizedReshape",
                                         "ScatterNd",
                                         "TensorScatterUpdate"};

  // TODO(unda, b/253431636): constrained kernel
  if (constrained_kernel.find(op_info.op_name) != constrained_kernel.end()) {
    std::cout << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " has a constrained kernel.\n";
    return false;
  }

  for (int i = 0; i < op_info.graph_op_def.input_arg_size(); ++i) {
    const auto& arg(op_info.graph_op_def.input_arg(i));
    // TODO(unda, b/249298521): deal with inputs that are required to be refs
    if (arg.is_ref()) {
      std::cout << "NOT fuzzing: " << op_info.graph_op_def.name()
                << " requires a ref argument.\n";
      return false;
    }
  }

  std::set<string> unhandled_attr_types = {
      "list(type)",   "func",         "float",      "bool",
      "tensor",       "list(string)", "list(bool)", "list(shape)",
      "list(tensor)", "list(attr)"};
  for (int i = 0; i < op_info.graph_op_def.attr_size(); ++i) {
    const auto& attr(op_info.graph_op_def.attr(i));
    const auto& api_def_attr(op_info.api_def.attr(i));
    // Skip inferred arguments
    if (op_info.inferred_input_attrs.count(attr.name()) > 0) continue;
    // Skip if it has default value (TODO(unda, b/249345399): add our custom
    // values)
    if (api_def_attr.has_default_value()) continue;
    // TODO(unda, b/253432797): handle unimplemented input attribute types
    if (unhandled_attr_types.find(attr.type()) != unhandled_attr_types.end()) {
      std::cout << "NOT fuzzing: " << op_info.graph_op_def.name()
                << " requires an unhandled attr type (" << attr.type()
                << ").\n";
      return false;
    }
  }
  std::cout << "fuzzing: " << op_info.graph_op_def.name() << "\n";
  return true;
}

tsl::Status WriteFuzzer(const OpDef& graph_op_def, const ApiDef& api_def,
                        WritableFile* fuzz) {
  OpInfo op_info(graph_op_def, api_def, std::vector<string>());
  tsl::Status status;
  if (OpFuzzingIsOk(op_info)) {
    status.Update(WriteClassFuzzDef(op_info, fuzz));
    status.Update(WriteFuzzTest(op_info, fuzz));
    status.Update(fuzz->Append("\n"));
  }
  return status;
}

tsl::Status StartFile(const string& fname, WritableFile* fuzz) {
  const string fuzz_namespace_begin = R"namespace(
namespace tensorflow {
namespace fuzzing {

)namespace";

  const string fuzz_header = strings::StrCat(
      R"include(// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"
#include "third_party/mediapipe/framework/port/parse_text_proto.h"
)include",
      fuzz_namespace_begin);

  return fuzz->Append(fuzz_header);
}

tsl::Status FinishFile(WritableFile* fuzz) {
  const string fuzz_footer = R"footer(
}  // namespace fuzzing
}  // namespace tensorflow
)footer";

  auto status = fuzz->Append(fuzz_footer);
  status.Update(fuzz->Close());
  return status;
}

}  // namespace

void WriteFuzzers(const OpList& ops, const ApiDefMap& api_def_map,
                  const string& fname) {
  Env* env = Env::Default();

  // Write the initial boilerplate for the fuzzers.
  std::unique_ptr<WritableFile> fuzz = nullptr;
  auto status = env->NewWritableFile(fname, &fuzz);
  status.Update(StartFile(fname, fuzz.get()));

  for (const auto& graph_op_def : ops.op()) {
    // Skip deprecated ops.
    if (graph_op_def.has_deprecation() &&
        graph_op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION) {
      continue;
    }

    const auto* api_def = api_def_map.GetApiDef(graph_op_def.name());
    status.Update(WriteFuzzer(graph_op_def, *api_def, fuzz.get()));
  }

  status.Update(FinishFile(fuzz.get()));
  TF_CHECK_OK(status);
}

}  // namespace cc_op
}  // namespace tensorflow
