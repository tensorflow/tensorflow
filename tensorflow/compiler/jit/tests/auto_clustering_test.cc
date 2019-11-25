/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/tests/auto_clustering_test_helper.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {
class AutoClusteringTestImpl : public AutoClusteringTest {
 protected:
  // Test auto-clustering with a proto text file ${key}.pbtxt.
  Status RunAutoClusteringTestWithPbtxt(absl::string_view key) {
    string file_name_without_extension =
        absl::StrCat(testing::TensorFlowSrcRoot(), "/compiler/jit/tests/", key);

    return AutoClusteringTest::RunAutoClusteringTestWithPbtxt(
        absl::StrCat(file_name_without_extension, ".pbtxt"),
        absl::StrCat(file_name_without_extension, ".golden_summary"));
  }

  // Test auto-clustering with a gzipped proto text file ${key}.pbtxt.gz.
  Status RunAutoClusteringTestWithGzippedPbtxt(absl::string_view key) {
    string file_name_without_extension =
        absl::StrCat(testing::TensorFlowSrcRoot(), "/compiler/jit/tests/", key);

    return AutoClusteringTest::RunAutoClusteringTestWithGzippedPbtxt(
        absl::StrCat(file_name_without_extension, ".pbtxt.gz"),
        absl::StrCat(file_name_without_extension, ".golden_summary"));
  }
};

TEST_F(AutoClusteringTestImpl, KerasImagenetMain) {
  // Generated from
  //
  //  TARGET_PATH=tensorflow_models/official/vision/image_classification       \
  //  bazel run -c opt --config=cuda ${TARGET_PATH}:resnet_imagenet_main       \
  //    -- --skip_eval --num_gpus=1 --dtype=fp16 --batch_size=192              \
  //    --train_steps=210 --enable_xla --enable_eager=true
  //
  // At CL 245846452
  TF_ASSERT_OK(RunAutoClusteringTestWithPbtxt("keras_imagenet_main"));
}

TEST_F(AutoClusteringTestImpl, KerasImagenetMainGraphMode) {
  // Generated from
  //
  //  TARGET_PATH=tensorflow_models/official/vision/image_classification       \
  //  bazel run -c opt --config=cuda ${TARGET_PATH}:resnet_imagenet_main       \
  //   -- --use_synthetic_data --num_gpus=1 --batch_size=117 --train_steps=600 \
  //   --skip_eval=True --logtostderr --enable_xla
  TF_ASSERT_OK(
      RunAutoClusteringTestWithPbtxt("keras_imagenet_main_graph_mode"));
}

TEST_F(AutoClusteringTestImpl, OpenSeq2SeqGNMT) {
  // Model is from https://github.com/NVIDIA/OpenSeq2Seq.
  // Generated from
  //
  // python run.py \
  // --config_file=example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py \
  // --use_xla_jit
  TF_ASSERT_OK(
      RunAutoClusteringTestWithGzippedPbtxt("opens2s_gnmt_mixed_precision"));
}

#if defined(PLATFORM_GOOGLE)
Status BenchmarkHelper(absl::string_view key, benchmark::State& state) {
  return BenchmarkMarkForCompilation(
      absl::StrCat(testing::TensorFlowSrcRoot(), "/compiler/jit/tests/", key,
                   ".pbtxt"),
      state);
}

void BM_MarkForCompilationPass_KerasImagenetMain(benchmark::State& state) {
  TF_CHECK_OK(BenchmarkHelper("keras_imagenet_main", state));
}

BENCHMARK(BM_MarkForCompilationPass_KerasImagenetMain);
#endif  // PLATFORM_GOOGLE

}  // namespace
}  // namespace tensorflow
