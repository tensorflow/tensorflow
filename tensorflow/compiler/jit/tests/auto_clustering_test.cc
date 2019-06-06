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
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace {
class AutoClusteringTestImpl : public AutoClusteringTest {
 protected:
  Status RunAutoClusteringTest(absl::string_view key) {
    string file_name_without_extension =
        absl::StrCat(testing::TensorFlowSrcRoot(), "/compiler/jit/tests/", key);

    return AutoClusteringTest::RunAutoClusteringTest(
        absl::StrCat(file_name_without_extension, ".pbtxt"),
        absl::StrCat(file_name_without_extension, ".golden_summary"));
  }

  // Decompress file ${key}.pbtxt.gz into ${key}.pbtxt
  // and test auto clustering with the .pbtxt.
  Status RunAutoClusteringTestWithGzippedPbtxt(absl::string_view key);
};

Status AutoClusteringTestImpl::RunAutoClusteringTestWithGzippedPbtxt(
    absl::string_view key) {
  string file_name_without_extension =
      absl::StrCat(testing::TensorFlowSrcRoot(), "/compiler/jit/tests/", key);
  string input_fname = absl::StrCat(file_name_without_extension, ".pbtxt.gz");
  string pbtxt_fname = absl::StrCat(file_name_without_extension, ".pbtxt");
  string summary_fname =
      absl::StrCat(file_name_without_extension, ".golden_summary");

  Env* env = Env::Default();
  std::unique_ptr<RandomAccessFile> file_reader;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(input_fname, &file_reader));
  std::unique_ptr<io::RandomAccessInputStream> input_stream(
      new io::RandomAccessInputStream(file_reader.get()));
  constexpr int k_buffer_size = 256 << 10;  // 256kb
  io::ZlibInputStream in(input_stream.get(),
                         /*input_buffer_bytes=*/k_buffer_size,
                         /*output_buffer_bytes=*/k_buffer_size,
                         io::ZlibCompressionOptions::GZIP());
  string decompressed_data;
  Status s = in.ReadNBytes(INT_MAX, &decompressed_data);
  if (!s.ok() && !errors::IsOutOfRange(s)) {
    // OutOfRange is fine since we set the number of read bytes to INT_MAX.
    // Only return other kinds of errors.
    return s;
  }
  TF_RETURN_IF_ERROR(WriteStringToFile(env, pbtxt_fname, decompressed_data));

  return AutoClusteringTest::RunAutoClusteringTest(pbtxt_fname, summary_fname);
}

TEST_F(AutoClusteringTestImpl, KerasImagenetMain) {
  // Generated from
  //
  //  bazel run -c opt --config=cuda                                           \
  //   tensorflow_models/official/resnet/keras:keras_imagenet_main             \
  //    -- --skip_eval --num_gpus=1 --dtype=fp16 --batch_size=192              \
  //    --train_steps=210 --enable_xla --enable_eager=true
  //
  // At CL 245846452
  TF_ASSERT_OK(RunAutoClusteringTest("keras_imagenet_main"));
}

TEST_F(AutoClusteringTestImpl, KerasImagenetMainGraphMode) {
  // Generated from
  //
  // bazel run -c opt --config=cuda                                            \
  //   tensorflow_models/official/resnet/keras:keras_imagenet_main             \
  //   -- --use_synthetic_data --num_gpus=1 --batch_size=117 --train_steps=600 \
  //   --skip_eval=True --logtostderr --enable_xla
  TF_ASSERT_OK(RunAutoClusteringTest("keras_imagenet_main_graph_mode"));
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
