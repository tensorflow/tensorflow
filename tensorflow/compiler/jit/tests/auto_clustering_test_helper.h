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
#ifndef TENSORFLOW_COMPILER_JIT_TESTS_AUTO_CLUSTERING_TEST_HELPER_H_
#define TENSORFLOW_COMPILER_JIT_TESTS_AUTO_CLUSTERING_TEST_HELPER_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
// Helper to write integration tests and benchmarks for the auto-clustering pass
// pipeline.  These tests run auto-clustering on a graphdef and compare a
// summary of the auto-clustering decisions with a "golden" summary.
//
// To create a new test from an TF workload first run the workload with the
// following environment variables set:
//
//   TF_DUMP_GRAPH_PREFIX=<some temporary directory>
//   TF_XLA_FLAGS="--tf_xla_clustering_debug"
//
// If auto-clustering is enabled this should produce files named
// before_mark_for_compilation_<N>.pbtxt in the temporary directory.  As the
// file name suggests, these are graphdefs that have been dumped right before
// the mark_for_compilation pass.  There should be one
// before_mark_for_compilation_<N>.pbtxt for every TF graph that was
// auto-clustered, out of which usually only one is the "main" graph that's
// running training/inference.
//
// Copy the pbtxt for that "main" graph to tensorflow/compiler/jit/tests/
// (i.e. this directory) and create a corresponding empty .golden_summary file.
// Add the .pbtxt and .golden_summary files to the "data" section of the cc_test
// rule for :auto_clustering_test and then see the comment on update_golden on
// how to auto-generate the .golden_summary file.

class AutoClusteringTest : public ::testing::Test {
 protected:
  Status RunAutoClusteringTestWithPbtxt(
      absl::string_view pbtxt_file_path,
      absl::string_view golden_summary_file_path);
  Status RunAutoClusteringTestWithGzippedPbtxt(
      absl::string_view gzipped_pbtxt_file_path,
      absl::string_view golden_summary_file_path);

 private:
  Status RunAutoClusteringTestImpl(GraphDef graphdef,
                                   absl::string_view golden_summary_file_path);
};

#if defined(PLATFORM_GOOGLE)
// Reads the GraphDef stored in graph_def_path (which must be a pbtxt file) and
// benchmarks MarkForCompilationPass on this graphdef.
Status BenchmarkMarkForCompilation(absl::string_view graph_def_path,
                                   benchmark::State& state);
#endif  // PLATFORM_GOOGLE

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_TESTS_AUTO_CLUSTERING_TEST_HELPER_H_
