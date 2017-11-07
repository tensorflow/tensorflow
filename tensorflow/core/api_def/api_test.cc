/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Test that verifies tensorflow/core/api_def/base_api/api_def*.pbtxt files
// are correct. If api_def*.pbtxt do not match expected contents, run
// tensorflow/core/api_def/base_api/update_api_def.sh script to update them.

#include <ctype.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/op_gen_overrides.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {
constexpr char kDefaultApiDefDir[] =
    "tensorflow/core/api_def/base_api";
constexpr char kOverridesFilePath[] =
    "tensorflow/cc/ops/op_gen_overrides.pbtxt";
constexpr char kApiDefFileFormat[] = "api_def_%s.pbtxt";
constexpr char kApiDefFilePattern[] = "api_def_*.pbtxt";

void FillBaseApiDef(ApiDef* api_def, const OpDef& op) {
  api_def->set_graph_op_name(op.name());
  // Add arg docs
  for (auto& input_arg : op.input_arg()) {
    if (!input_arg.description().empty()) {
      auto* api_def_in_arg = api_def->add_in_arg();
      api_def_in_arg->set_name(input_arg.name());
      api_def_in_arg->set_description(input_arg.description());
    }
  }
  for (auto& output_arg : op.output_arg()) {
    if (!output_arg.description().empty()) {
      auto* api_def_out_arg = api_def->add_out_arg();
      api_def_out_arg->set_name(output_arg.name());
      api_def_out_arg->set_description(output_arg.description());
    }
  }
  // Add attr docs
  for (auto& attr : op.attr()) {
    if (!attr.description().empty()) {
      auto* api_def_attr = api_def->add_attr();
      api_def_attr->set_name(attr.name());
      api_def_attr->set_description(attr.description());
    }
  }
  // Add docs
  api_def->set_summary(op.summary());
  api_def->set_description(op.description());
}

// Checks if arg1 should be before arg2 according to ordering in args.
bool CheckArgBefore(const ApiDef::Arg* arg1, const ApiDef::Arg* arg2,
                    const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
  for (auto& arg : args) {
    if (arg.name() == arg2->name()) {
      return false;
    } else if (arg.name() == arg1->name()) {
      return true;
    }
  }
  return false;
}

// Checks if attr1 should be before attr2 according to ordering in op_def.
bool CheckAttrBefore(const ApiDef::Attr* attr1, const ApiDef::Attr* attr2,
                     const OpDef& op_def) {
  for (auto& attr : op_def.attr()) {
    if (attr.name() == attr2->name()) {
      return false;
    } else if (attr.name() == attr1->name()) {
      return true;
    }
  }
  return false;
}

// Applies renames to args.
void ApplyArgOverrides(
    protobuf::RepeatedPtrField<ApiDef::Arg>* args,
    const protobuf::RepeatedPtrField<OpGenOverride::Rename>& renames,
    const protobuf::RepeatedPtrField<OpDef::ArgDef>& op_args,
    const string& op_name) {
  for (auto& rename : renames) {
    // First check if rename is valid.
    bool valid = false;
    for (const auto& op_arg : op_args) {
      if (op_arg.name() == rename.from()) {
        valid = true;
      }
    }
    QCHECK(valid) << rename.from() << " is not a valid argument for "
                  << op_name;
    bool found_arg = false;
    // If Arg is already in ApiDef, just update it.
    for (int i = 0; i < args->size(); ++i) {
      auto* arg = args->Mutable(i);
      if (arg->name() == rename.from()) {
        arg->set_rename_to(rename.to());
        found_arg = true;
        break;
      }
    }
    if (!found_arg) {  // not in ApiDef, add a new arg.
      auto* new_arg = args->Add();
      new_arg->set_name(rename.from());
      new_arg->set_rename_to(rename.to());
    }
  }
  // We don't really need a specific order here right now.
  // However, it is clearer if order follows OpDef.
  std::sort(args->pointer_begin(), args->pointer_end(),
            [&](ApiDef::Arg* arg1, ApiDef::Arg* arg2) {
              return CheckArgBefore(arg1, arg2, op_args);
            });
}

// Returns existing attribute with the given name if such
// attribute exists. Otherwise, adds a new attribute and returns it.
ApiDef::Attr* FindOrAddAttr(ApiDef* api_def, const string attr_name) {
  // If Attr is already in ApiDef, just update it.
  for (int i = 0; i < api_def->attr_size(); ++i) {
    auto* attr = api_def->mutable_attr(i);
    if (attr->name() == attr_name) {
      return attr;
    }
  }
  // Add a new Attr.
  auto* new_attr = api_def->add_attr();
  new_attr->set_name(attr_name);
  return new_attr;
}

// Applies renames and default values to attributes.
void ApplyAttrOverrides(ApiDef* api_def, const OpGenOverride& op_override,
                        const OpDef& op_def) {
  for (auto& attr_rename : op_override.attr_rename()) {
    auto* attr = FindOrAddAttr(api_def, attr_rename.from());
    attr->set_rename_to(attr_rename.to());
  }

  for (auto& attr_default : op_override.attr_default()) {
    auto* attr = FindOrAddAttr(api_def, attr_default.name());
    *(attr->mutable_default_value()) = attr_default.value();
  }
  // We don't really need a specific order here right now.
  // However, it is clearer if order follows OpDef.
  std::sort(api_def->mutable_attr()->pointer_begin(),
            api_def->mutable_attr()->pointer_end(),
            [&](ApiDef::Attr* attr1, ApiDef::Attr* attr2) {
              return CheckAttrBefore(attr1, attr2, op_def);
            });
}

void ApplyOverridesToApiDef(ApiDef* api_def, const OpDef& op,
                            const OpGenOverride& op_override) {
  // Fill ApiDef with data based on op and op_override.
  // Set visibility
  if (op_override.skip()) {
    api_def->set_visibility(ApiDef_Visibility_SKIP);
  } else if (op_override.hide()) {
    api_def->set_visibility(ApiDef_Visibility_HIDDEN);
  }
  // Add endpoints
  if (!op_override.rename_to().empty()) {
    api_def->add_endpoint()->set_name(op_override.rename_to());
  } else if (!op_override.alias().empty()) {
    api_def->add_endpoint()->set_name(op.name());
  }

  for (auto& alias : op_override.alias()) {
    auto* endpoint = api_def->add_endpoint();
    endpoint->set_name(alias);
  }

  ApplyArgOverrides(api_def->mutable_in_arg(), op_override.input_rename(),
                    op.input_arg(), api_def->graph_op_name());
  ApplyArgOverrides(api_def->mutable_out_arg(), op_override.output_rename(),
                    op.output_arg(), api_def->graph_op_name());
  ApplyAttrOverrides(api_def, op_override, op);
}

// Get map from ApiDef file path to corresponding ApiDefs proto.
std::unordered_map<string, ApiDefs> GenerateApiDef(
    const string& api_def_dir, const OpList& ops,
    const OpGenOverrides& overrides) {
  std::unordered_map<string, OpGenOverride> name_to_override;
  for (const auto& op_override : overrides.op()) {
    name_to_override[op_override.name()] = op_override;
  }

  std::unordered_map<string, ApiDefs> api_defs_map;

  for (const auto& op : ops.op()) {
    CHECK(!op.name().empty())
        << "Encountered empty op name: %s" << op.DebugString();
    string file_path = io::JoinPath(api_def_dir, kApiDefFileFormat);
    file_path = strings::Printf(file_path.c_str(), op.name().c_str());
    ApiDef* api_def = api_defs_map[file_path].add_op();
    FillBaseApiDef(api_def, op);

    if (name_to_override.find(op.name()) != name_to_override.end()) {
      ApplyOverridesToApiDef(api_def, op, name_to_override[op.name()]);
    }
  }
  return api_defs_map;
}

// Reads golden ApiDef files and returns a map from file name to ApiDef file
// contents.
std::unordered_map<string, string> GetGoldenApiDefs(
    Env* env, const string& api_files_dir) {
  std::vector<string> matching_paths;
  TF_CHECK_OK(env->GetMatchingPaths(
      io::JoinPath(api_files_dir, kApiDefFilePattern), &matching_paths));

  std::unordered_map<string, string> file_path_to_api_def;
  for (auto& file_path : matching_paths) {
    string file_contents;
    TF_CHECK_OK(ReadFileToString(env, file_path, &file_contents));
    file_path_to_api_def[file_path] = file_contents;
  }
  return file_path_to_api_def;
}

void RunApiTest(bool update_api_def, const string& api_files_dir) {
  // Read C++ overrides file
  OpGenOverrides overrides;
  Env* env = Env::Default();
  TF_EXPECT_OK(ReadTextProto(env, kOverridesFilePath, &overrides));

  // Read all ops
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);
  const std::vector<string> multi_line_fields = {"description"};

  // Get expected ApiDefs
  const auto new_api_defs_map = GenerateApiDef(api_files_dir, ops, overrides);

  bool updated_at_least_one_file = false;
  const auto golden_api_defs_map = GetGoldenApiDefs(env, api_files_dir);

  for (auto new_api_entry : new_api_defs_map) {
    const auto& file_path = new_api_entry.first;
    const auto& golden_api_defs_str = golden_api_defs_map.at(file_path);
    string new_api_defs_str = new_api_entry.second.DebugString();
    new_api_defs_str = PBTxtToMultiline(new_api_defs_str, multi_line_fields);
    if (golden_api_defs_str == new_api_defs_str) {
      continue;
    }
    if (update_api_def) {
      std::cout << "Updating " << file_path << "..." << std::endl;
      TF_EXPECT_OK(WriteStringToFile(env, file_path, new_api_defs_str));
      updated_at_least_one_file = true;
    } else {
      EXPECT_EQ(golden_api_defs_str, new_api_defs_str)
          << "To update golden API files, run "
          << "tensorflow/core/api_def/update_api_def.sh.";
    }
  }

  for (const auto& golden_api_entry : golden_api_defs_map) {
    const auto& file_path = golden_api_entry.first;
    if (new_api_defs_map.find(file_path) == new_api_defs_map.end()) {
      if (update_api_def) {
        std::cout << "Deleting " << file_path << "..." << std::endl;
        TF_EXPECT_OK(env->DeleteFile(file_path));
        updated_at_least_one_file = true;
      } else {
        EXPECT_EQ("", golden_api_entry.second)
            << "To update golden API files, run "
            << "tensorflow/core/api_def/update_api_def.sh.";
      }
    }
  }

  if (update_api_def && !updated_at_least_one_file) {
    std::cout << "Api def files are already up to date." << std::endl;
  }
}

TEST(ApiTest, GenerateBaseAPIDef) { RunApiTest(false, kDefaultApiDefDir); }
}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  bool update_api_def = false;
  tensorflow::string api_files_dir = tensorflow::kDefaultApiDefDir;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag(
          "update_api_def", &update_api_def,
          "Whether to update tensorflow/core/api_def/base_api/api_def*.pbtxt "
          "files if they differ from expected API."),
      tensorflow::Flag("api_def_dir", &api_files_dir,
                       "Base directory of api_def*.pbtxt files.")};
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_values_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parsed_values_ok) {
    std::cerr << usage << std::endl;
    return 2;
  }
  if (update_api_def) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);
    tensorflow::RunApiTest(update_api_def, api_files_dir);
    return 0;
  }
  testing::InitGoogleTest(&argc, argv);
  // Run tests
  return RUN_ALL_TESTS();
}
