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

// Test that validates tensorflow/core/api_def/base_api/api_def*.pbtxt files.

#include <ctype.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/api_def/excluded_ops.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

constexpr char kApiDefFilePattern[] = "api_def_*.pbtxt";

string DefaultApiDefDir() {
  return GetDataDependencyFilepath(
      io::JoinPath("tensorflow", "core", "api_def", "base_api"));
}

string PythonApiDefDir() {
  return GetDataDependencyFilepath(
      io::JoinPath("tensorflow", "core", "api_def", "python_api"));
}

// Reads golden ApiDef files and returns a map from file name to ApiDef file
// contents.
void GetGoldenApiDefs(Env* env, const string& api_files_dir,
                      std::unordered_map<string, ApiDef>* name_to_api_def) {
  std::vector<string> matching_paths;
  TF_CHECK_OK(env->GetMatchingPaths(
      io::JoinPath(api_files_dir, kApiDefFilePattern), &matching_paths));

  for (auto& file_path : matching_paths) {
    string file_contents;
    TF_CHECK_OK(ReadFileToString(env, file_path, &file_contents));
    file_contents = PBTxtFromMultiline(file_contents);

    ApiDefs api_defs;
    QCHECK(tensorflow::protobuf::TextFormat::ParseFromString(file_contents,
                                                             &api_defs))
        << "Failed to load " << file_path;
    CHECK_EQ(api_defs.op_size(), 1);
    (*name_to_api_def)[api_defs.op(0).graph_op_name()] = api_defs.op(0);
  }
}

void TestAllApiDefsHaveCorrespondingOp(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
  std::unordered_set<string> op_names;
  for (const auto& op : ops.op()) {
    op_names.insert(op.name());
  }
  for (const auto& name_and_api_def : api_defs_map) {
    ASSERT_TRUE(op_names.find(name_and_api_def.first) != op_names.end())
        << name_and_api_def.first << " op has ApiDef but missing from ops. "
        << "Does api_def_" << name_and_api_def.first << " need to be deleted?";
  }
}

void TestAllApiDefInputArgsAreValid(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
  for (const auto& op : ops.op()) {
    const auto api_def_iter = api_defs_map.find(op.name());
    if (api_def_iter == api_defs_map.end()) {
      continue;
    }
    const auto& api_def = api_def_iter->second;
    for (const auto& api_def_arg : api_def.in_arg()) {
      bool found_arg = false;
      for (const auto& op_arg : op.input_arg()) {
        if (api_def_arg.name() == op_arg.name()) {
          found_arg = true;
          break;
        }
      }
      ASSERT_TRUE(found_arg)
          << "Input argument " << api_def_arg.name()
          << " (overwritten in api_def_" << op.name()
          << ".pbtxt) is not defined in OpDef for " << op.name();
    }
  }
}

void TestAllApiDefOutputArgsAreValid(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
  for (const auto& op : ops.op()) {
    const auto api_def_iter = api_defs_map.find(op.name());
    if (api_def_iter == api_defs_map.end()) {
      continue;
    }
    const auto& api_def = api_def_iter->second;
    for (const auto& api_def_arg : api_def.out_arg()) {
      bool found_arg = false;
      for (const auto& op_arg : op.output_arg()) {
        if (api_def_arg.name() == op_arg.name()) {
          found_arg = true;
          break;
        }
      }
      ASSERT_TRUE(found_arg)
          << "Output argument " << api_def_arg.name()
          << " (overwritten in api_def_" << op.name()
          << ".pbtxt) is not defined in OpDef for " << op.name();
    }
  }
}

void TestAllApiDefAttributeNamesAreValid(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
  for (const auto& op : ops.op()) {
    const auto api_def_iter = api_defs_map.find(op.name());
    if (api_def_iter == api_defs_map.end()) {
      continue;
    }
    const auto& api_def = api_def_iter->second;
    for (const auto& api_def_attr : api_def.attr()) {
      bool found_attr = false;
      for (const auto& op_attr : op.attr()) {
        if (api_def_attr.name() == op_attr.name()) {
          found_attr = true;
        }
      }
      ASSERT_TRUE(found_attr)
          << "Attribute " << api_def_attr.name() << " (overwritten in api_def_"
          << op.name() << ".pbtxt) is not defined in OpDef for " << op.name();
    }
  }
}

void TestDeprecatedAttributesSetCorrectly(
    const std::unordered_map<string, ApiDef>& api_defs_map) {
  for (const auto& name_and_api_def : api_defs_map) {
    int num_deprecated_endpoints = 0;
    const auto& api_def = name_and_api_def.second;
    for (const auto& endpoint : api_def.endpoint()) {
      if (endpoint.deprecated()) {
        ++num_deprecated_endpoints;
      }
    }

    const auto& name = name_and_api_def.first;
    ASSERT_TRUE(api_def.deprecation_message().empty() ||
                num_deprecated_endpoints == 0)
        << "Endpoints are set to 'deprecated' for deprecated op " << name
        << ". If an op is deprecated (i.e. deprecation_message is set), "
        << "all the endpoints are deprecated implicitly and 'deprecated' "
        << "field should not be set.";
    if (num_deprecated_endpoints > 0) {
      ASSERT_NE(num_deprecated_endpoints, api_def.endpoint_size())
          << "All " << name << " endpoints are deprecated. Please, set "
          << "deprecation_message in api_def_" << name << ".pbtxt instead. "
          << "to indicate that the op is deprecated.";
    }
  }
}

void TestDeprecationVersionSetCorrectly(
    const std::unordered_map<string, ApiDef>& api_defs_map) {
  for (const auto& name_and_api_def : api_defs_map) {
    const auto& name = name_and_api_def.first;
    const auto& api_def = name_and_api_def.second;
    if (api_def.deprecation_version() != 0) {
      ASSERT_TRUE(api_def.deprecation_version() > 0)
          << "Found ApiDef with negative deprecation_version";
      ASSERT_FALSE(api_def.deprecation_message().empty())
          << "ApiDef that includes deprecation_version > 0 must also specify "
          << "a deprecation_message. Op " << name
          << " has deprecation_version > 0 but deprecation_message is not set.";
    }
  }
}

class BaseApiTest : public ::testing::Test {
 protected:
  BaseApiTest() {
    OpRegistry::Global()->Export(false, &ops_);
    const std::vector<string> multi_line_fields = {"description"};

    Env* env = Env::Default();
    GetGoldenApiDefs(env, DefaultApiDefDir(), &api_defs_map_);
  }
  OpList ops_;
  std::unordered_map<string, ApiDef> api_defs_map_;
};

// Check that all ops have an ApiDef.
TEST_F(BaseApiTest, AllOpsAreInApiDef) {
  auto* excluded_ops = GetExcludedOps();
  for (const auto& op : ops_.op()) {
    if (excluded_ops->find(op.name()) != excluded_ops->end()) {
      continue;
    }
    EXPECT_TRUE(api_defs_map_.find(op.name()) != api_defs_map_.end())
        << op.name() << " op does not have api_def_*.pbtxt file. "
        << "Please add api_def_" << op.name() << ".pbtxt file "
        << "under tensorflow/core/api_def/base_api/ directory.";
  }
}

// Check that ApiDefs have a corresponding op.
TEST_F(BaseApiTest, AllApiDefsHaveCorrespondingOp) {
  TestAllApiDefsHaveCorrespondingOp(ops_, api_defs_map_);
}

string GetOpDefHasDocStringError(const string& op_name) {
  return strings::Printf(
      "OpDef for %s has a doc string. "
      "Doc strings must be defined in ApiDef instead of OpDef. "
      "Please, add summary and descriptions in api_def_%s"
      ".pbtxt file instead",
      op_name.c_str(), op_name.c_str());
}

// Check that OpDef's do not have descriptions and summaries.
// Descriptions and summaries must be in corresponding ApiDefs.
TEST_F(BaseApiTest, OpDefsShouldNotHaveDocs) {
  auto* excluded_ops = GetExcludedOps();
  for (const auto& op : ops_.op()) {
    if (excluded_ops->find(op.name()) != excluded_ops->end()) {
      continue;
    }
    ASSERT_TRUE(op.summary().empty()) << GetOpDefHasDocStringError(op.name());
    ASSERT_TRUE(op.description().empty())
        << GetOpDefHasDocStringError(op.name());
    for (const auto& arg : op.input_arg()) {
      ASSERT_TRUE(arg.description().empty())
          << GetOpDefHasDocStringError(op.name());
    }
    for (const auto& arg : op.output_arg()) {
      ASSERT_TRUE(arg.description().empty())
          << GetOpDefHasDocStringError(op.name());
    }
    for (const auto& attr : op.attr()) {
      ASSERT_TRUE(attr.description().empty())
          << GetOpDefHasDocStringError(op.name());
    }
  }
}

// Checks that input arg names in an ApiDef match input
// arg names in corresponding OpDef.
TEST_F(BaseApiTest, AllApiDefInputArgsAreValid) {
  TestAllApiDefInputArgsAreValid(ops_, api_defs_map_);
}

// Checks that output arg names in an ApiDef match output
// arg names in corresponding OpDef.
TEST_F(BaseApiTest, AllApiDefOutputArgsAreValid) {
  TestAllApiDefOutputArgsAreValid(ops_, api_defs_map_);
}

// Checks that attribute names in an ApiDef match attribute
// names in corresponding OpDef.
TEST_F(BaseApiTest, AllApiDefAttributeNamesAreValid) {
  TestAllApiDefAttributeNamesAreValid(ops_, api_defs_map_);
}

// Checks that deprecation is set correctly.
TEST_F(BaseApiTest, DeprecationSetCorrectly) {
  TestDeprecatedAttributesSetCorrectly(api_defs_map_);
}

// Checks that deprecation_version is set for entire op only if
// deprecation_message is set.
TEST_F(BaseApiTest, DeprecationVersionSetCorrectly) {
  TestDeprecationVersionSetCorrectly(api_defs_map_);
}

class PythonApiTest : public ::testing::Test {
 protected:
  PythonApiTest() {
    OpRegistry::Global()->Export(false, &ops_);
    const std::vector<string> multi_line_fields = {"description"};

    Env* env = Env::Default();
    GetGoldenApiDefs(env, PythonApiDefDir(), &api_defs_map_);
  }
  OpList ops_;
  std::unordered_map<string, ApiDef> api_defs_map_;
};

// Check that ApiDefs have a corresponding op.
TEST_F(PythonApiTest, AllApiDefsHaveCorrespondingOp) {
  TestAllApiDefsHaveCorrespondingOp(ops_, api_defs_map_);
}

// Checks that input arg names in an ApiDef match input
// arg names in corresponding OpDef.
TEST_F(PythonApiTest, AllApiDefInputArgsAreValid) {
  TestAllApiDefInputArgsAreValid(ops_, api_defs_map_);
}

// Checks that output arg names in an ApiDef match output
// arg names in corresponding OpDef.
TEST_F(PythonApiTest, AllApiDefOutputArgsAreValid) {
  TestAllApiDefOutputArgsAreValid(ops_, api_defs_map_);
}

// Checks that attribute names in an ApiDef match attribute
// names in corresponding OpDef.
TEST_F(PythonApiTest, AllApiDefAttributeNamesAreValid) {
  TestAllApiDefAttributeNamesAreValid(ops_, api_defs_map_);
}

// Checks that deprecation is set correctly.
TEST_F(PythonApiTest, DeprecationSetCorrectly) {
  TestDeprecatedAttributesSetCorrectly(api_defs_map_);
}

// Checks that deprecation_version is set for entire op only if
// deprecation_message is set.
TEST_F(PythonApiTest, DeprecationVersionSetCorrectly) {
  TestDeprecationVersionSetCorrectly(api_defs_map_);
}

}  // namespace
}  // namespace tensorflow
