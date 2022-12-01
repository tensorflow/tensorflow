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

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

TEST_F(XlaCompilationCacheSerializeTest, PersistentCacheOptionsTest) {
  GraphDef graph = GetTestGraph({-1, 4});

  // Warmup the persistent cache(s) with multiple runs. 4 is a magic number to
  // detect non-determinism in TF when running the test.
  listener()->ClearListenerHistory();
  for (int b = 1; b < 4; ++b) {
    TF_ASSERT_OK(ExecuteWithBatch(graph, b));
  }
  TF_ASSERT_OK(listener()->VerifyPersistentCacheUseListenerHistory(
      /*expect_persistent_cache_use=*/false));

  // Reset the cluster numbering between sessions so we can get the same
  // cluster numbering.
  testing::ResetClusterSequenceNumber();

  auto status =
      AlterPersistentCacheEntryHloModuleNames(tensorflow::testing::TmpDir());
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(
      status.error_message(),
      "Did not find any persistent XLA compilation cache entries to alter."));

  TF_ASSERT_OK(AlterPersistentCacheEntryHloModuleNames(
      tensorflow::testing::TmpDir(), "my_test_prefix"));

  // Run again and these should all hit in the persistent cache despite having
  // altered the persistent cache entries' HLO modules (disabled strict
  // signature checks).
  listener()->ClearListenerHistory();
  for (int b = 1; b < 4; ++b) {
    TF_ASSERT_OK(ExecuteWithBatch(graph, b));
  }
  TF_ASSERT_OK(listener()->VerifyPersistentCacheUseListenerHistory(
      /*expect_persistent_cache_use=*/true));
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_deterministic_cluster_names = true;
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = tensorflow::testing::TmpDir();
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_disable_strict_signature_checks = true;
  tensorflow::GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_prefix =
      "my_test_prefix";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
