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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/aot/codegen.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/aot/tfcompile.pb.h"
#include "tensorflow/compiler/aot/tfcompile_util.h"
#include "tensorflow/compiler/xla/legacy_flags/compiler_functor_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_runtime_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tfcompile {

const char kUsageHeader[] =
    "tfcompile performs ahead-of-time compilation of a TensorFlow graph,\n"
    "resulting in an object file compiled for your target architecture, and a\n"
    "header file that gives access to the functionality in the object file.\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ tfcompile --graph=mygraph.pb --config=myfile.pbtxt "
    "--cpp_class=\"mynamespace::MyComputation\"\n"
    "\n";

Status ReadProtoFile(const string& kind, const string& fname,
                     protobuf::Message* proto) {
  if (StringPiece(fname).ends_with(".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

void ParseTensorId(const string& name, TensorId* id) {
  const std::pair<StringPiece, int> name_index = ParseTensorName(name);
  id->set_node_name(name_index.first.ToString());
  id->set_output_index(name_index.second);
}

Status Main(const MainFlags& flags) {
  // Process config.
  Config config;
  if (flags.config.empty()) {
    return errors::InvalidArgument("Must specify --config");
  }
  TF_RETURN_IF_ERROR(ReadProtoFile("config", flags.config, &config));
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  if (flags.dump_fetch_nodes) {
    std::set<string> nodes;
    for (const Fetch& fetch : config.fetch()) {
      nodes.insert(fetch.id().node_name());
    }
    std::cout << str_util::Join(nodes, ",");
    return Status::OK();
  }

  // Read and initialize the graph.
  if (flags.graph.empty()) {
    return errors::InvalidArgument("Must specify --graph");
  }
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadProtoFile("graph", flags.graph, &graph_def));
  std::unique_ptr<Graph> graph;
  TF_RETURN_IF_ERROR(InitGraph(graph_def, config, flags, &graph));

  CompileResult compile_result;
  TF_RETURN_IF_ERROR(CompileGraph(std::move(graph), flags, &compile_result));

  // Write output files.
  Env* env = Env::Default();
  const std::vector<char>& obj = compile_result.aot->object_file_data();
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_object,
                                       StringPiece(obj.data(), obj.size())));
  HeaderOpts header_opts;
  if (flags.cpp_class.empty()) {
    return errors::InvalidArgument("Must specify --cpp_class");
  }
  TF_RETURN_IF_ERROR(ParseCppClass(flags.cpp_class, &header_opts.class_name,
                                   &header_opts.namespaces));
  string header;
  TF_RETURN_IF_ERROR(
      GenerateHeader(header_opts, config, compile_result, &header));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_header, header));
  return Status::OK();
}

}  // end namespace tfcompile
}  // end namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::tfcompile::MainFlags flags;
  flags.target_triple = "x86_64-pc-linux";
  flags.out_object = "out.o";
  flags.out_header = "out.h";

  std::vector<tensorflow::Flag> flag_list;
  AppendMainFlags(&flag_list, &flags);
  xla::legacy_flags::AppendCompilerFunctorFlags(&flag_list);
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::legacy_flags::AppendCpuRuntimeFlags(&flag_list);

  tensorflow::string usage = tensorflow::tfcompile::kUsageHeader;
  usage += tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags\n\n"
                    << usage;
  tensorflow::Status status = tensorflow::tfcompile::Main(flags);
  if (status.code() == tensorflow::error::INVALID_ARGUMENT) {
    std::cerr << "INVALID ARGUMENTS: " << status.error_message() << "\n\n"
              << usage;
    return 1;
  } else {
    TF_QCHECK_OK(status);
  }
  return 0;
}
