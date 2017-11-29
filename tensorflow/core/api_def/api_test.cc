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
constexpr char kApiDefFileFormat[] = "api_def_%c.pbtxt";
constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// Get map from first character to ApiDefs for ops
// that start with that character.
std::unordered_map<char, ApiDefs> GenerateApiDef(
    const OpList& ops, const OpGenOverrides& overrides) {
  std::unordered_map<string, OpGenOverride> name_to_override;
  for (const auto& op_override : overrides.op()) {
    name_to_override[op_override.name()] = op_override;
  }

  std::unordered_map<char, ApiDefs> api_defs_map;

  for (const auto& op : ops.op()) {
    CHECK(!op.name().empty())
        << "Encountered empty op name: %s" << op.DebugString();
    const char file_id = toupper(op.name()[0]);
    CHECK(isalpha(file_id)) << "Unexpected op name: " << op.name();
    ApiDef* api_def = api_defs_map[file_id].add_op();
    api_def->set_graph_op_name(op.name());

    if (name_to_override.find(op.name()) != name_to_override.end()) {
      const auto& op_override = name_to_override[op.name()];
      // Set visibility
      if (op_override.skip()) {
        api_def->set_visibility(ApiDef_Visibility_SKIP);
      } else if (op_override.hide()) {
        api_def->set_visibility(ApiDef_Visibility_HIDDEN);
      }
      // Add endpoints
      if (!op_override.rename_to().empty()) {
        auto* endpoint = api_def->add_endpoint();
        endpoint->set_name(op_override.rename_to());
      } else {
        auto* endpoint = api_def->add_endpoint();
        endpoint->set_name(op.name());
      }
      for (auto& alias : op_override.alias()) {
        auto* endpoint = api_def->add_endpoint();
        endpoint->set_name(alias);
      }
      // Add attributes
      for (auto& attr : op.attr()) {
        auto* api_def_attr = api_def->add_attr();
        api_def_attr->set_name(attr.name());
        for (auto& attr_override : op_override.attr_default()) {
          if (attr.name() == attr_override.name()) {
            *(api_def_attr->mutable_default_value()) = attr_override.value();
          }
        }
        for (auto& attr_rename : op_override.attr_rename()) {
          if (attr.name() == attr_rename.from()) {
            api_def_attr->set_rename_to(attr_rename.to());
          }
        }
      }
    } else {
      auto* endpoint = api_def->add_endpoint();
      endpoint->set_name(op.name());
    }
    // Add docs
    api_def->set_summary(op.summary());
    api_def->set_description(op.description());
  }
  return api_defs_map;
}

// Reads golden api defs file with the given suffix.
string GetGoldenApiDefsStr(Env* env, const string& api_files_dir, char suffix) {
  string file_path = strings::Printf(
      io::JoinPath(api_files_dir, kApiDefFileFormat).c_str(), suffix);
  if (env->FileExists(file_path).ok()) {
    string file_contents;
    TF_EXPECT_OK(ReadFileToString(env, file_path, &file_contents));
    return file_contents;
  }
  return "";
}

void RunApiTest(bool update_api_def, const string& api_files_dir) {
  // Read C++ overrides file
  string overrides_file_contents;
  Env* env = Env::Default();
  TF_EXPECT_OK(
      ReadFileToString(env, kOverridesFilePath, &overrides_file_contents));

  // Read all ops
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);
  const std::vector<string> multi_line_fields = {"description"};

  // Get expected ApiDefs
  OpGenOverrides overrides;
  auto new_api_defs_map = GenerateApiDef(ops, overrides);

  bool updated_at_least_one_file = false;

  for (char c : kAlphabet) {
    string golden_api_defs_str = GetGoldenApiDefsStr(env, api_files_dir, c);
    string new_api_defs_str = new_api_defs_map[c].DebugString();
    new_api_defs_str = PBTxtToMultiline(new_api_defs_str, multi_line_fields);
    if (golden_api_defs_str == new_api_defs_str) {
      continue;
    }
    if (update_api_def) {
      string output_file_path =
          io::JoinPath(api_files_dir, strings::Printf(kApiDefFileFormat, c));
      if (new_api_defs_str.empty()) {
        std::cout << "Deleting " << output_file_path << "..." << std::endl;
        TF_EXPECT_OK(env->DeleteFile(output_file_path));
      } else {
        std::cout << "Updating " << output_file_path << "..." << std::endl;
        TF_EXPECT_OK(
            WriteStringToFile(env, output_file_path, new_api_defs_str));
      }
      updated_at_least_one_file = true;
    } else {
      EXPECT_EQ(golden_api_defs_str, new_api_defs_str)
          << "To update golden API files, run "
          << "tensorflow/core/api_def/update_api_def.sh.";
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
