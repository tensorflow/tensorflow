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

#include "tensorflow/compiler/jit/tests/auto_clustering_test_helper.h"

#include "absl/strings/numbers.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/tools/optimization/optimization_pass_runner.h"

namespace tensorflow {
namespace {
StatusOr<string> SummarizeClustering(const GraphDef& auto_clustered_graph_def) {
  testing::ResetClusterSequenceNumber();
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions graph_opts;
  graph_opts.expect_device_spec = true;
  graph_opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_opts, auto_clustered_graph_def, &graph));

  // cluster_id -> (operation name -> # of operations)
  const int kNoCluster = -1;
  std::map<int, std::map<string, int>> clusters;
  std::map<int, int> cluster_size;
  int clustered_nodes = 0;
  for (Node* n : graph.op_nodes()) {
    int cluster = kNoCluster;
    if (absl::optional<absl::string_view> maybe_cluster =
            GetXlaClusterForNode(*n)) {
      maybe_cluster->remove_prefix(absl::string_view("cluster_").size());
      TF_RET_CHECK(absl::SimpleAtoi(*maybe_cluster, &cluster));
      clustered_nodes++;
    }
    clusters[cluster][n->type_string()]++;
    cluster_size[cluster]++;
  }

  string result =
      absl::StrCat("Clustered nodes: ", clustered_nodes,
                   "\nUnclustered nodes: ", cluster_size[kNoCluster],
                   "\nNumber of clusters: ", clusters.size() - 1, "\n\n");
  for (const auto& pair : clusters) {
    if (pair.first == kNoCluster) {
      absl::StrAppend(&result, "unclustered");
    } else {
      absl::StrAppend(&result, "cluster ", pair.first);
    }

    absl::StrAppend(&result, " size ", cluster_size[pair.first], "\n");

    for (const auto& ops_and_counts : pair.second) {
      absl::StrAppend(&result, " ", ops_and_counts.first, " ",
                      ops_and_counts.second, "\n");
    }
  }

  return result;
}

Status AssertGraphDefIsUnclustered(const GraphDef& graphdef) {
  const char* kXlaClusterAttr = "_XlaCluster";
  const char* kXlaAlreadyClusteredAttr = "_XlaAlreadyClustered";

  for (const NodeDef& node : graphdef.node()) {
    if (node.attr().count(kXlaClusterAttr) ||
        node.attr().count(kXlaAlreadyClusteredAttr)) {
      return errors::InvalidArgument(
          "Input files are already clustered, you probably copied in "
          "mark_for_compilation_<n>.pbtxt when you should have copied in "
          "before_mark_for_compilation_<n>.pbtxt");
    }
  }

  return Status::OK();
}

Status ReadTextProtoFromString(Env* env, const string& data,
                               ::tensorflow::protobuf::Message* proto) {
  if (!::tensorflow::protobuf::TextFormat::ParseFromString(data, proto)) {
    return errors::DataLoss("Can't parse input data as text proto");
  }
  return Status::OK();
}
}  // namespace

Status AutoClusteringTest::RunAutoClusteringTestImpl(
    GraphDef graphdef, absl::string_view golden_summary_file_path) {
  if (!IsGoogleCudaEnabled()) {
    // There is some slight change in the clustering decisions under
    // --config=cuda.  I have not looked closely at why that is happening, but
    // most likely some of the partial declustering passes behave differently
    // with --config=cuda because of different HostMemory.  So for now only test
    // the non-CUDA config, under the assumption that regressions with
    // --config=cuda would also be detected as regressions without
    // --config=cuda.

    LOG(INFO) << "Not running "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " since test was not built with --config=cuda";
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(AssertGraphDefIsUnclustered(graphdef));

  OptimizationPassRunner runner;
  TF_RETURN_IF_ERROR(runner.SetJitLevel(tensorflow::OptimizerOptions::ON_2));
  TF_RETURN_IF_ERROR(runner.AddCpus(32));
  TF_RETURN_IF_ERROR(runner.AddGpus(8));

  for (absl::string_view auto_clustering_pass :
       {"CloneConstantsForBetterClusteringPass", "MarkForCompilationPass",
        "IncreaseDynamismForAutoJitPass", "PartiallyDeclusterPass"}) {
    GraphDef next;
    TF_RETURN_IF_ERROR(
        runner.Run(auto_clustering_pass, std::move(graphdef), &next));
    graphdef = std::move(next);
  }

  TF_ASSIGN_OR_RETURN(string clustering_summary, SummarizeClustering(graphdef));

  // To update golden files flip this to true and run
  //
  // bazel test --test_strategy=local \
  //   tensorflow/compiler/jit/tests:auto_clustering_test
  bool update_golden = false;
  if (update_golden) {
    TF_RETURN_IF_ERROR(WriteStringToFile(
        Env::Default(), string(golden_summary_file_path), clustering_summary));
  }

  string golden_file_contents;
  TF_RETURN_IF_ERROR(ReadFileToString(
      Env::Default(), string(golden_summary_file_path), &golden_file_contents));

  EXPECT_EQ(golden_file_contents, clustering_summary);

  return Status::OK();
}

Status AutoClusteringTest::RunAutoClusteringTestWithPbtxt(
    absl::string_view pbtxt_file_path,
    absl::string_view golden_summary_file_path) {
  GraphDef graphdef;
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), string(pbtxt_file_path), &graphdef));
  return RunAutoClusteringTestImpl(std::move(graphdef),
                                   golden_summary_file_path);
}

Status AutoClusteringTest::RunAutoClusteringTestWithGzippedPbtxt(
    absl::string_view gzipped_pbtxt_file_path,
    absl::string_view golden_summary_file_path) {
  Env* env = Env::Default();
  std::unique_ptr<RandomAccessFile> file_reader;
  TF_RETURN_IF_ERROR(
      env->NewRandomAccessFile(string(gzipped_pbtxt_file_path), &file_reader));
  std::unique_ptr<io::RandomAccessInputStream> input_stream(
      new io::RandomAccessInputStream(file_reader.get()));
  constexpr int k_buffer_size = 256 << 10;  // 256kb
  io::ZlibInputStream in(input_stream.get(),
                         /*input_buffer_bytes=*/k_buffer_size,
                         /*output_buffer_bytes=*/k_buffer_size,
                         io::ZlibCompressionOptions::GZIP());
  tstring decompressed_pbtxt_string;
  Status s = in.ReadNBytes(INT_MAX, &decompressed_pbtxt_string);
  if (!s.ok() && !errors::IsOutOfRange(s)) {
    // OutOfRange is fine since we set the number of read bytes to INT_MAX.
    // Only return other kinds of errors.
    return s;
  }

  GraphDef graphdef;
  TF_RETURN_IF_ERROR(ReadTextProtoFromString(
      Env::Default(), decompressed_pbtxt_string, &graphdef));
  return RunAutoClusteringTestImpl(std::move(graphdef),
                                   golden_summary_file_path);
}

#if defined(PLATFORM_GOOGLE)
Status BenchmarkMarkForCompilation(absl::string_view graph_def_path,
                                   benchmark::State& state) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), string(graph_def_path), &graph_def));

  OptimizationPassRunner runner;
  TF_RETURN_IF_ERROR(runner.SetJitLevel(tensorflow::OptimizerOptions::ON_2));
  TF_RETURN_IF_ERROR(runner.AddCpus(32));
  TF_RETURN_IF_ERROR(runner.AddGpus(8));

  for (auto _ : state) {
    state.PauseTiming();
    GraphDef result;
    GraphDef graph_def_copy = graph_def;
    state.ResumeTiming();
    TF_RETURN_IF_ERROR(runner.Run("MarkForCompilationPass",
                                  std::move(graph_def_copy), &result));
  }

  return Status::OK();
}
#endif  // PLATFORM_GOOGLE

}  // namespace tensorflow
