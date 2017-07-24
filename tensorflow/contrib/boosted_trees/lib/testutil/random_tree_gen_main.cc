// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// Randomly generate a tree ensemble and write to file.

#include "tensorflow/contrib/boosted_trees/lib/testutil/random_tree_gen.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Flags;
using tensorflow::int32;
using tensorflow::string;

int main(int argc, char* argv[]) {
  int32 dense_feature_size = 100;
  int32 sparse_feature_size = 100;
  int32 depth = 8;
  int32 tree_count = 10;
  string filename = "/tmp/trees.pb";
  std::vector<Flag> flag_list = {
      Flag("dense_feature_size", &dense_feature_size, "dense feature size"),
      Flag("sparse_feature_size", &sparse_feature_size, "sparse_feature_size"),
      Flag("depth", &depth, "tree depth"),
      Flag("tree_count", &tree_count, "tree count"),
      Flag("filename", &filename, "Output filename."),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return -1;
  }

  tensorflow::random::PhiloxRandom philox(1);
  tensorflow::random::SimplePhilox rng(&philox);
  tensorflow::boosted_trees::testutil::RandomTreeGen tree_gen(
      &rng, dense_feature_size, sparse_feature_size);
  const auto& trees = tree_gen.GenerateEnsemble(depth, tree_count);
  tensorflow::Status status =
      tensorflow::WriteBinaryProto(tensorflow::Env::Default(), filename, trees);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to write: " << filename << " : " << status;
  } else {
    LOG(INFO) << "Tree ensemble written to: " << filename;
  }
  return 0;
}
