/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/cc/experimental/libtf/module.h"

#include <string>

#include "tensorflow/cc/experimental/libtf/runtime/core/core.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tf {
namespace libtf {
namespace impl {

using ::tensorflow::libexport::TFPackage;
using ::tensorflow::testing::StatusIs;
using ::tf::libtf::runtime::Runtime;

TEST(ModuleTest, TestStubbedFunctions) {
  Runtime runtime = runtime::core::Runtime();
  TFPackage tf_package;
  tensorflow::StatusOr<Handle> result = BuildProgram(runtime, tf_package);
  ASSERT_FALSE(result.status().ok());
}

TEST(ModuleTest, TestBuildObjectsDataStructures) {
  const std::string path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/cc/experimental/libtf/tests/testdata/data-structure-model");
  TF_ASSERT_OK_AND_ASSIGN(TFPackage tf_package, TFPackage::Load(path));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Handle> objects,
                          BuildObjects(tf_package));
  EXPECT_EQ(objects.size(), 7);
  // The first node of data-structure-model is a dictionary.
  TF_ASSERT_OK_AND_ASSIGN(tf::libtf::Dictionary node,
                          Cast<tf::libtf::Dictionary>(objects.front()));

  // The next three nodes of data-structure-model are lists.
  for (unsigned int i = 1; i < 4; i++) {
    TF_ASSERT_OK_AND_ASSIGN(tf::libtf::List node,
                            Cast<tf::libtf::List>(objects.at(i)));
  }
  // The last three nodes of data-structure-model are dictionaries.
  for (unsigned int i = 4; i < 7; i++) {
    TF_ASSERT_OK_AND_ASSIGN(tf::libtf::Dictionary node,
                            Cast<tf::libtf::Dictionary>(objects.at(i)));
  }
}

TEST(ModuleTest, TestBuildEmptyList) {
  tensorflow::SavedObject saved_object_proto;
  const std::string pb_txt = R"pb(
    user_object {
      identifier: "trackable_list_wrapper"
      version { producer: 1 min_consumer: 1 }
    }
  )pb";

  ASSERT_TRUE(::tensorflow::protobuf::TextFormat::ParseFromString(
      pb_txt, &saved_object_proto));
  TF_ASSERT_OK_AND_ASSIGN(Handle result,
                          BuildSavedUserObject(saved_object_proto));
  EXPECT_EQ(Cast<tf::libtf::List>(result)->size(), 0);
}

TEST(ModuleTest, TestBuildEmptyDict) {
  tensorflow::SavedObject saved_object_proto;
  const std::string pb_txt = R"pb(
    user_object {
      identifier: "trackable_dict_wrapper"
      version { producer: 1 min_consumer: 1 }
    }
  )pb";

  ASSERT_TRUE(::tensorflow::protobuf::TextFormat::ParseFromString(
      pb_txt, &saved_object_proto));

  TF_ASSERT_OK_AND_ASSIGN(Handle result,
                          BuildSavedUserObject(saved_object_proto));
  EXPECT_EQ(Cast<tf::libtf::Dictionary>(result)->size(), 0);
}

TEST(ModuleTest, TestBuildSignatureMap) {
  tensorflow::SavedObject saved_object_proto;
  const std::string pb_txt = R"pb(
    user_object {
      identifier: "signature_map"
      version { producer: 1 min_consumer: 1 }
    }
  )pb";

  ASSERT_TRUE(::tensorflow::protobuf::TextFormat::ParseFromString(
      pb_txt, &saved_object_proto));
  TF_ASSERT_OK_AND_ASSIGN(Handle result,
                          BuildSavedUserObject(saved_object_proto));
  EXPECT_EQ(Cast<tf::libtf::Dictionary>(result)->size(), 0);
}

TEST(ModuleTest, TestUnimplementedUserObject) {
  tensorflow::SavedObject saved_object_proto;
  const std::string pb_txt = R"pb(
    user_object {
      identifier: "foo"
      version { producer: 1 min_consumer: 1 }
    }
  )pb";

  ASSERT_TRUE(::tensorflow::protobuf::TextFormat::ParseFromString(
      pb_txt, &saved_object_proto));

  EXPECT_THAT(
      BuildSavedUserObject(saved_object_proto),
      StatusIs(tensorflow::error::UNIMPLEMENTED, ::testing::HasSubstr("foo")));
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
