/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#include <cstdio>
#include <ctime>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {

void DumpXlaGraph(const string& logdir, const string& graphdef_file) {
  string run_dir = io::JoinPath(logdir, strings::StrCat("xla_graph", 0));

  auto* env = tensorflow::Env::Default();

  TF_CHECK_OK(env->RecursivelyCreateDir(run_dir));

  GraphDef graphdef;
  if (graphdef_file.substr(graphdef_file.size()-3, 3) == "txt") {
    TF_CHECK_OK(ReadTextProto(env, graphdef_file, &graphdef));
  } else {
    std::cout << "trying to read file\n";
    TF_CHECK_OK(ReadBinaryProto(env, graphdef_file, &graphdef));
    std::cout << "done\n";
  }

  EventsWriter event_writer(run_dir);
  Event event;

  // Add the computation graph.
  event.set_graph_def(graphdef.SerializeAsString());
  event_writer.WriteEvent(event);
  std::cout << "Wrote a HLO graph to " << event_writer.FileName() << std::endl;
}

}

int main(int argc, char** argv) {
  tensorflow::string FLAGS_xla_graphdef;
  tensorflow::string FLAGS_poplar_compute;
  tensorflow::string FLAGS_logdir;

  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("xla_in", &FLAGS_xla_graphdef,
                       "XLA Graphdef file"),
      tensorflow::Flag("compute_in", &FLAGS_poplar_compute,
                       "Poplar compute graph file"),
      tensorflow::Flag("logdir", &FLAGS_logdir,
                       "Path of TensorBoard log directory e.g. /tmp/tb_log"),
  };

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || FLAGS_logdir.empty()) {
    std::printf("%s", usage.c_str());
    return 2;
  }

  if (!FLAGS_xla_graphdef.empty()) {
    tensorflow::DumpXlaGraph(FLAGS_logdir, FLAGS_xla_graphdef);
  }

  return 0;
}
