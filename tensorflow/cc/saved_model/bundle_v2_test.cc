/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/bundle_v2.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "json/reader.h"
#include "json/value.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

constexpr char kTestData[] = "cc/saved_model/testdata";

class BundleV2Test : public ::testing::Test {
 protected:
  BundleV2Test() {}

  void RestoreVarsAndVerify(SavedModelV2Bundle* bundle,
                            std::vector<std::string> expected_names) {
    // Collect saved_node_id, full_name, checkpoint_key into a vector.
    using RestoredVarType = std::tuple<int, std::string, std::string>;
    std::vector<RestoredVarType> restored_vars;
    TF_ASSERT_OK(bundle->VisitObjectsToRestore(
        [&](int saved_node_id,
            const TrackableObjectGraph::TrackableObject& trackable_object)
            -> absl::Status {
          for (const auto& attr : trackable_object.attributes()) {
            if (attr.name() == "VARIABLE_VALUE") {
              restored_vars.emplace_back(saved_node_id, attr.full_name(),
                                         attr.checkpoint_key());
            }
          }
          return absl::OkStatus();
        }));

    // Should be one of each var name restored.
    for (const auto& expected_name : expected_names) {
      EXPECT_EQ(1, std::count_if(restored_vars.begin(), restored_vars.end(),
                                 [&](RestoredVarType t) {
                                   return std::get<1>(t) == expected_name;
                                 }));
    }

    for (const auto& restored_var : restored_vars) {
      // Each restored var should match a SavedObjectGraph node with the same
      // variable name.
      const auto& saved_node =
          bundle->saved_object_graph().nodes(std::get<0>(restored_var));
      EXPECT_EQ(std::get<1>(restored_var), saved_node.variable().name());

      // And should be able to load it from the tensor_bundle.
      Tensor value;
      TF_ASSERT_OK(
          bundle->variable_reader()->Lookup(std::get<2>(restored_var), &value));
    }
  }
};

TEST_F(BundleV2Test, LoadsVarsAndArithmeticObjectGraph) {
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), kTestData, "VarsAndArithmeticObjectGraph");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  // Ensure that there are nodes in the trackable_object_graph.
  EXPECT_GT(bundle.trackable_object_graph().nodes_size(), 0);

  RestoreVarsAndVerify(&bundle, {"variable_x", "variable_y", "child_variable"});
}

TEST_F(BundleV2Test, LoadsCyclicModule) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestData, "CyclicModule");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  // Ensure that there are nodes in the trackable_object_graph.
  EXPECT_GT(bundle.trackable_object_graph().nodes_size(), 0);

  RestoreVarsAndVerify(&bundle, {"MyVariable"});
}

TEST_F(BundleV2Test, UpdatesMetrics) {
  const std::string kCCLoadBundleV2Label = "cc_load_bundle_v2";
  const int read_count = metrics::SavedModelReadCount("2").value();
  const int api_count =
      metrics::SavedModelReadApi(kCCLoadBundleV2Label).value();
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), kTestData, "VarsAndArithmeticObjectGraph");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  EXPECT_EQ(metrics::SavedModelReadCount("2").value(), read_count + 1);
  EXPECT_EQ(metrics::SavedModelReadApi(kCCLoadBundleV2Label).value(),
            api_count + 1);
  // Check that the gauge contains the path and fingerprint.
  EXPECT_EQ(metrics::SavedModelReadPath().value(), export_dir);

  Json::Value fingerprint = Json::objectValue;
  Json::Reader reader = Json::Reader();
  reader.parse(metrics::SavedModelReadFingerprint().value(), fingerprint);
  EXPECT_EQ(fingerprint["saved_model_checksum"].asUInt64(),
            15788619162413586750ULL);
  EXPECT_EQ(fingerprint["graph_def_program_hash"].asUInt64(),
            706963557435316516ULL);
  EXPECT_EQ(fingerprint["signature_def_hash"].asUInt64(),
            5693392539583495303ULL);
  EXPECT_EQ(fingerprint["saved_object_graph_hash"].asUInt64(),
            12074714563970609759ULL);
  EXPECT_EQ(fingerprint["checkpoint_hash"].asUInt64(), 10788359570789890102ULL);

  TF_ASSERT_OK_AND_ASSIGN(
      auto path_and_singleprint,
      metrics::ParseSavedModelPathAndSingleprint(
          metrics::SavedModelReadPathAndSingleprint().value()));
  auto [path, singleprint] = path_and_singleprint;
  EXPECT_TRUE(absl::StrContains(
      path, absl::StrCat(kTestData, "/VarsAndArithmeticObjectGraph")));
  EXPECT_EQ(singleprint,
            "706963557435316516/"     // graph_def_program_hash
            "5693392539583495303/"    // signature_def_hash
            "12074714563970609759/"   // saved_object_graph_hash
            "10788359570789890102");  // checkpoint_hash
}

}  // namespace
}  // namespace tensorflow
