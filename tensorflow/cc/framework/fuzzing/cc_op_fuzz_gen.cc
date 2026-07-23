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

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/framework/cc_op_gen_util.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace cc_op {
namespace {

std::string DefaultValue(const OpDef_AttrDef& attr) {
  static const absl::NoDestructor<
      absl::flat_hash_map<absl::string_view, absl::string_view>>
      kAttrDefaultValueMap({{"int", "0"},
                            {"string", "\"\""},
                            {"list(int)", "{ 0, 1 }"},
                            {"list(float)", "{0.0, 1.0}"},
                            {"type", "DT_UINT8"},
                            {"shape",
                             "mediapipe::ParseTextProtoOrDie<TensorShapeProto>("
                             "\"dim:[] unknown_rank:true\")"}});

  if (attr.has_minimum()) {
    if (attr.type() == "int") {
      return absl::StrCat(attr.minimum());
    } else if (attr.type() == "list(int)") {
      std::vector<int> v(attr.minimum());
      std::iota(v.begin(), v.end(), 0);
      return absl::StrCat("{", absl::StrJoin(v, ","), "}");
    }
  }
  if (attr.has_allowed_values()) {
    if (!attr.allowed_values().list().s().empty()) {
      return absl::StrCat(
          "\"", absl::CEscape(attr.allowed_values().list().s(0)), "\"");
    } else if (!attr.allowed_values().list().type().empty()) {
      return DataType_Name(attr.allowed_values().list().type(0));
    }
  }
  auto entry = kAttrDefaultValueMap->find(attr.type());
  if (entry == kAttrDefaultValueMap->end()) {
    LOG(ERROR) << "Unsupported Attr type: " << attr.type();
    return "";
  }
  return std::string(entry->second);
}

std::string WriteClassFuzzDef(const OpInfo& op_info) {
  std::string class_signature_str = absl::Substitute(
      "class Fuzz$0 : public FuzzSession<$1> {\n", op_info.op_name,
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](std::string* out, const auto& arg) {
                      absl::StrAppend(out, "Tensor");
                      if (ArgIsList(arg)) absl::StrAppend(out, ", Tensor");
                    }));

  std::string build_graph_body = absl::StrCat(
      absl::StrJoin(
          op_info.graph_op_def.input_arg(), "",
          [&op_info](std::string* out, const OpDef_ArgDef& arg) {
            std::string type = "DT_UINT8";

            if (arg.type() != DT_INVALID) {
              type = DataType_Name(arg.type());
            } else if (!arg.type_attr().empty()) {
              const OpDef_AttrDef* attr =
                  FindAttr(arg.type_attr(), op_info.graph_op_def);
              if (attr != nullptr && attr->has_default_value() &&
                  attr->default_value().value_case() == AttrValue::kType) {
                type = DataType_Name(attr->default_value().type());
              } else if (attr != nullptr && attr->has_allowed_values() &&
                         attr->allowed_values().value_case() ==
                             AttrValue::kList &&
                         !attr->allowed_values().list().type().empty()) {
                type = DataType_Name(attr->allowed_values().list().type(0));
              }
            }
            if (ArgIsList(arg)) {
              absl::StrAppend(
                  out, "    Input ", arg.name(),
                  "_0 = ", "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                  absl::CEscape(arg.name()), "\"), ", type, ");\n");
              absl::StrAppend(
                  out, "    Input ", arg.name(),
                  "_1 = ", "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                  absl::CEscape(arg.name()), "\"), ", type, ");\n");
              absl::StrAppend(
                  out, absl::Substitute("    InputList $0({$0_0, $0_1});\n",
                                        arg.name()));
            } else {
              absl::StrAppend(
                  out, "    auto ", arg.name(), " = ",
                  "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                  absl::CEscape(arg.name()), "\"), ", type, ");\n");
            }
          }),
      absl::StrJoin(op_info.graph_op_def.attr(), "",
                    [&op_info](std::string* out, const OpDef_AttrDef& attr) {
                      if (op_info.inferred_input_attrs.count(attr.name()) ==
                              0 &&
                          !attr.has_default_value()) {
                        absl::StrAppend(out, "    auto ", attr.name(), " = ",
                                        DefaultValue(attr), ";\n");
                      }
                    }));

  std::string constructor_call_str = absl::Substitute(
      "    tensorflow::ops::$0(scope.WithOpName(\"output\")$1);\n",
      op_info.op_name,
      absl::StrCat(
          op_info.api_def.arg_order().empty()
              ? absl::StrJoin(op_info.api_def.in_arg(), "",
                              [](std::string* out, const auto& api_def_arg) {
                                absl::StrAppend(out, ", ", api_def_arg.name());
                              })
              : absl::StrJoin(op_info.api_def.arg_order(), "",
                              [](std::string* out, const auto& name) {
                                absl::StrAppend(out, ", ", name);
                              }),
          absl::StrJoin(op_info.graph_op_def.attr(), "",
                        [&op_info](std::string* out,
                                   const OpDef_AttrDef& attr) {
                          if (op_info.inferred_input_attrs.count(attr.name()) ==
                                  0 &&
                              !attr.has_default_value()) {
                            absl::StrAppend(out, ", ", attr.name());
                          }
                        })));

  std::string fuzz_impl_signature_str = absl::Substitute(
      "  void FuzzImpl($0) final {\n",
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](std::string* out, const auto& arg) {
                      absl::StrAppend(out, "const Tensor& ", arg.name(), "_0");
                      if (ArgIsList(arg))
                        absl::StrAppend(out, ", const Tensor& ", arg.name(),
                                        "_1");
                    }));

  std::string run_inputs_str = absl::Substitute(
      "    RunInputs({$0});\n",
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](std::string* out, const auto& arg) {
                      if (ArgIsList(arg)) {
                        absl::StrAppend(out, "{\"", absl::CEscape(arg.name()),
                                        "\", ", arg.name(), "_0}, ", "{\"",
                                        absl::CEscape(arg.name()), "\", ",
                                        arg.name(), "_1}");
                      } else {
                        absl::StrAppend(out, "{\"", absl::CEscape(arg.name()),
                                        "\", ", arg.name(), "_0}");
                      }
                    }));

  std::string fuzz_class_def = absl::StrCat(
      class_signature_str, "  void BuildGraph(const Scope& scope) override {\n",
      build_graph_body, constructor_call_str, "  }\n", fuzz_impl_signature_str,
      run_inputs_str, "  }\n", "};\n");

  return fuzz_class_def;
}

std::string WriteFuzzTest(const OpInfo& op_info) {
  return absl::Substitute(
      "FUZZ_TEST_F(Fuzz$0, Fuzz).WithDomains($1);\n", op_info.op_name,
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](std::string* out, const auto& arg) {
                      absl::StrAppend(out, "AnyTensor()");
                      if (ArgIsList(arg)) absl::StrAppend(out, ", AnyTensor()");
                    }));
}

std::string FuzzerFileStart() {
  constexpr absl::string_view kFuzzNamespaceBegin = R"namespace(
namespace tensorflow {
namespace fuzzing {

)namespace";

  return absl::StrCat(R"include(// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"
#include "third_party/mediapipe/framework/port/parse_text_proto.h"
)include",
                      kFuzzNamespaceBegin);
}

std::string FuzzerFileEnd() {
  constexpr absl::string_view kFuzzFooter = R"footer(
}  // namespace fuzzing
}  // namespace tensorflow
)footer";

  return std::string(kFuzzFooter);
}

}  // namespace

bool OpFuzzingIsOk(const OpInfo& op_info) {
  // Skip deprecated ops.
  if (op_info.graph_op_def.has_deprecation() &&
      op_info.graph_op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION) {
    LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " is deprecated.";
    return false;
  }

  // TODO(unda, b/249347507): should we hide fuzzers for hidden ops?
  if (op_info.api_def.visibility() == ApiDef::HIDDEN) {
    LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " is hidden.";
    return false;
  }

  if (op_info.api_def.visibility() == ApiDef::SKIP) {
    LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " is skipped.";
    return false;
  }

  // TODO(unda, b/253431636): zero input ops
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>>
      kZeroInputOps({"Placeholder", "ImmutableConst"});
  if (kZeroInputOps->contains(op_info.op_name)) {
    LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " takes zero inputs.";
    return false;
  }

  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>>
      kConstrainedKernel({"Diag", "DiagPart", "GatherNd", "GatherV2",
                          "QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3",
                          "QuantizeAndDequantizeV4",
                          "QuantizeAndDequantizeV4Grad", "QuantizedConcat",
                          "QuantizedInstanceNorm", "QuantizedReshape",
                          "ScatterNd", "TensorScatterUpdate"});

  // TODO(unda, b/253431636): constrained kernel
  if (kConstrainedKernel->contains(op_info.op_name)) {
    LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
              << " has a constrained kernel.";
    return false;
  }

  for (int i = 0; i < op_info.graph_op_def.input_arg_size(); ++i) {
    const auto& arg = op_info.graph_op_def.input_arg(i);
    // TODO(unda, b/249298521): deal with inputs that are required to be refs
    if (arg.is_ref()) {
      LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
                << " requires a ref argument.";
      return false;
    }
  }

  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>>
      kUnhandledAttrTypes({"list(type)", "func", "float", "bool", "tensor",
                           "list(string)", "list(bool)", "list(shape)",
                           "list(tensor)", "list(attr)"});
  for (int i = 0; i < op_info.graph_op_def.attr_size(); ++i) {
    const auto& attr = op_info.graph_op_def.attr(i);
    const auto& api_def_attr = op_info.api_def.attr(i);
    // Skip inferred arguments
    if (op_info.inferred_input_attrs.count(attr.name()) > 0) continue;
    // Skip if it has default value (TODO(unda, b/249345399): add our custom
    // values)
    if (api_def_attr.has_default_value()) continue;
    // TODO(unda, b/253432797): handle unimplemented input attribute types
    if (kUnhandledAttrTypes->contains(attr.type())) {
      LOG(INFO) << "NOT fuzzing: " << op_info.graph_op_def.name()
                << " requires an unhandled attr type (" << attr.type() << ").";
      return false;
    }
  }
  LOG(INFO) << "fuzzing: " << op_info.graph_op_def.name();
  return true;
}

std::string WriteSingleFuzzer(const OpInfo& op_info, bool is_fuzzable) {
  return absl::StrCat(FuzzerFileStart(),
                      is_fuzzable ? WriteClassFuzzDef(op_info) : std::string(),
                      is_fuzzable ? WriteFuzzTest(op_info) : std::string(),
                      FuzzerFileEnd());
}

}  // namespace cc_op
}  // namespace tensorflow
