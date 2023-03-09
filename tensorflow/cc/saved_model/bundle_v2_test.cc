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

#include <string>
#include <tuple>
#include <vector>

#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kTestData[] = "cc/saved_model/testdata";
// This is the value in testdata/VarsAndArithmeticObjectGraph/fingerprint.pb
constexpr char kV2ModuleSavedModelChecksum[] = "15788619162413586750";

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
            -> Status {
          for (const auto& attr : trackable_object.attributes()) {
            if (attr.name() == "VARIABLE_VALUE") {
              restored_vars.emplace_back(saved_node_id, attr.full_name(),
                                         attr.checkpoint_key());
            }
          }
          return OkStatus();
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
  const string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), kTestData, "VarsAndArithmeticObjectGraph");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  // Ensure that there are nodes in the trackable_object_graph.
  EXPECT_GT(bundle.trackable_object_graph().nodes_size(), 0);

  RestoreVarsAndVerify(&bundle, {"variable_x", "variable_y", "child_variable"});
}

TEST_F(BundleV2Test, LoadsCyclicModule) {
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestData, "CyclicModule");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  // Ensure that there are nodes in the trackable_object_graph.
  EXPECT_GT(bundle.trackable_object_graph().nodes_size(), 0);

  RestoreVarsAndVerify(&bundle, {"MyVariable"});
}

TEST_F(BundleV2Test, UpdatesMetrics) {
  const string kCCLoadBundleV2Label = "cc_load_bundle_v2";
  const int read_count = metrics::SavedModelReadCount("2").value();
  const int api_count =
      metrics::SavedModelReadApi(kCCLoadBundleV2Label).value();
  const string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), kTestData, "VarsAndArithmeticObjectGraph");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  EXPECT_EQ(metrics::SavedModelReadCount("2").value(), read_count + 1);
  EXPECT_EQ(metrics::SavedModelReadApi(kCCLoadBundleV2Label).value(),
            api_count + 1);
  // Check that the gauge contains the fingerprint.
  EXPECT_EQ(metrics::SavedModelReadFingerprint().value(),
            kV2ModuleSavedModelChecksum);
  EXPECT_EQ(metrics::SavedModelReadPath().value(), export_dir);
}

}  // namespace
}  // namespace tensorflow
