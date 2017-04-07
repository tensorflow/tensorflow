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

// Usage: dumped_computation_to_tf_graph \
//          --output_dir=/tmp/graphs/ some_binary_snapshot_proto*
//
// Dumps a tensorflow GraphDef in text format for a snapshot computation. The
// dumped graph is an HLO computation with HLO instructions as nodes and can be
// visualized on Tensorboard. Upload the dumped files on Tensorboard.
//
// some_binary_snapshot_proto is obtained by serializing the SessionModule from
// ServiceInterface::SnapshotComputation to disk.

#include <stdio.h>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

using tensorflow::Env;
using tensorflow::io::JoinPath;
using tensorflow::strings::StrAppend;

namespace xla {
namespace tools {
namespace {

// Dumps all computations in the module to the given directory.
void DumpTfGraph(const HloModule& module, const string& directory_path) {
  Env* env = Env::Default();
  TF_CHECK_OK(env->RecursivelyCreateDir(directory_path));
  string fname = module.name();
  std::replace(fname.begin(), fname.end(), '/', '_');
  // Since the file name will be used as the top-level scope name, clean it up
  // to make it a valid scope name.
  CleanNodeName(&fname);
  StrAppend(&fname, ".pbtxt");
  string path = JoinPath(directory_path, fname);
  HloTfGraphBuilder builder;
  TF_CHECK_OK(builder.AddComputation(*module.entry_computation()));
  std::cout << "Dumping " << module.name() << " to " << path << std::endl;
  TF_CHECK_OK(WriteTextProto(env, path, builder.GetGraphDef()));
}

}  // namespace

void RealMain(tensorflow::gtl::ArraySlice<char*> args,
              const string& output_dir) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  // To avoid adding a new flag, use local service and lower the computations
  // locally.
  LocalService* local_service =
      ClientLibrary::GetXlaService(client->platform());
  // Build HloModule for each Computation and dump to file.
  for (char* arg : args) {
    SessionModule session_module;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), arg,
                                            &session_module));
    auto computation_status = client->LoadSnapshot(session_module);
    if (!computation_status.ok()) {
      fprintf(stderr, "could not load snapshot for %s: %s\n", arg,
              computation_status.status().ToString().c_str());
      continue;
    }
    Computation computation = computation_status.ConsumeValueOrDie();

    StatusOr<UserComputation*> user_computation_status =
        local_service->computation_tracker().Resolve(computation.handle());
    if (!user_computation_status.ok()) {
      fprintf(stderr,
              "failed to resolve computation to UserComputation %s: %s\n", arg,
              user_computation_status.status().ToString().c_str());
      continue;
    }

    auto* user_computation = user_computation_status.ValueOrDie();
    StatusOr<std::unique_ptr<HloModule>> module_status =
        local_service->computation_tracker().BuildHloModule(
            user_computation->GetVersionedHandle());

    if (!module_status.ok()) {
      fprintf(stderr, "failed to build HloModule %s: %s\n", arg,
              module_status.status().ToString().c_str());
      continue;
    }

    DumpTfGraph(*module_status.ValueOrDie(), output_dir);
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  string output_dir = "";
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("output_dir", &output_dir,
                       "Directory to write GraphDef data to."),
  };

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || output_dir.empty()) {
    LOG(QFATAL) << usage;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::gtl::ArraySlice<char*> args(argv, argc);
  args.pop_front();  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args, output_dir);
  return 0;
}
