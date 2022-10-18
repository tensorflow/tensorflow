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

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/framework/cc_op_gen_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace cc_op {
namespace {

tsl::Status WriteClassFuzzDef(const OpInfo& op_info, WritableFile* fuzz) {
  string class_signature_str = absl::Substitute(
      "class Fuzz$0 : public FuzzSession<$1> {\n", op_info.op_name,
      absl::StrJoin(op_info.graph_op_def.input_arg(), ", ",
                    [](string* out, const auto arg) {
                      strings::StrAppend(out, "Tensor");
                      if (ArgIsList(arg)) strings::StrAppend(out, ", Tensor");
                    }));

  string build_graph_body = absl::StrJoin(
      op_info.graph_op_def.input_arg(), "", [](string* out, const auto arg) {
        if (ArgIsList(arg)) {
          strings::StrAppend(out, "    Input ", arg.name(), "_0 = ",
                             "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                             arg.name(), "\"), DT_UINT8);\n");
          strings::StrAppend(out, "    Input ", arg.name(), "_1 = ",
                             "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                             arg.name(), "\"), DT_UINT8);\n");
          strings::StrAppend(
              out, absl::Substitute("    InputList $0({$0_0, $0_1});\n",
                                    arg.name()));
        } else {
          strings::StrAppend(out, "    auto ", arg.name(), " = ",
                             "tensorflow::ops::Placeholder(scope.WithOpName(\"",
                             arg.name(), "\"), DT_UINT8);\n");
        }
      });

  string constructor_call_str = absl::Substitute(
      "    tensorflow::ops::$0(scope.WithOpName(\"output\")$1);\n",
      op_info.op_name,
      absl::StrJoin(op_info.graph_op_def.input_arg(), "",
                    [](string* out, const auto arg) {
                      strings::StrAppend(out, ", ", arg.name());
                    }));

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

  // TODO(unda, b/249298326): no kernel problem
  std::set<string> skip_ops = {"MatrixDiag", "MatrixDiagPart", "MatrixSetDiag"};

  for (int i = 0; i < op_info.graph_op_def.input_arg_size(); ++i) {
    const auto& arg(op_info.graph_op_def.input_arg(i));
    // TODO(unda, b/249298521): deal with inputs that are required to be refs
    if (arg.is_ref()) return false;

    // TODO(unda, b/249298840): deal with restricted allowed types
    if (!arg.type_attr().empty()) {
      const OpDef::AttrDef* attr =
          FindAttr(arg.type_attr(), op_info.graph_op_def);
      if (attr->has_allowed_values()) return false;
    }
    if (!arg.number_attr().empty()) {
      const OpDef::AttrDef* attr =
          FindAttr(arg.number_attr(), op_info.graph_op_def);
      if (attr->has_allowed_values()) return false;
    }
    if (!arg.type_list_attr().empty()) {
      const OpDef::AttrDef* attr =
          FindAttr(arg.type_list_attr(), op_info.graph_op_def);
      if (attr->has_allowed_values()) return false;
    }

    // TODO(unda, b/249298265): deal with non Tensor input types
    if (arg.type_attr().empty()) return false;

    // TODO(unda, b/249298326): no kernel problem
    if (skip_ops.find(op_info.op_name) != skip_ops.end()) return false;
  }

  for (int i = 0; i < op_info.graph_op_def.attr_size(); ++i) {
    const auto& attr(op_info.graph_op_def.attr(i));
    const auto& api_def_attr(op_info.api_def.attr(i));
    // Skip inferred arguments
    if (op_info.inferred_input_attrs.count(attr.name()) > 0) continue;
    // Skip if it has default value (TODO(unda, b/249345399): add our custom
    // values)
    if (api_def_attr.has_default_value()) continue;
    return false;
    // TODO(unda, b/249298525): deal with non standard attributes
    // const auto attr_type_name = AttrTypeName(attr.type()).first;
    // strings::StrAppend(&build_graph_body, "    ", attr_type_name, " ",
    //                 attr.name(), " = ", DefaultValue(attr_type_name), ";\n");
    // strings::StrAppend(&constructor_call_str, ", ", attr.name());
  }

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
#include "tensorflow/security/fuzzing/cc/ops/fuzz_session.h"
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

    // We use a hand-written wrapper for "Const", since the generated
    // code depends on it.
    if (graph_op_def.name() == "Const") continue;

    const auto* api_def = api_def_map.GetApiDef(graph_op_def.name());
    status.Update(WriteFuzzer(graph_op_def, *api_def, fuzz.get()));
  }

  status.Update(FinishFile(fuzz.get()));
  TF_CHECK_OK(status);
}

}  // namespace cc_op
}  // namespace tensorflow
