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

// Dumps out the operations that are present in a serialized computation.

#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {

class OperationDumper : public DfsHloVisitorWithDefault {
 public:
  explicit OperationDumper(const string& path) : path_(path) {}

  Status DefaultAction(HloInstruction* hlo) override {
    string params = tensorflow::str_util::Join(
        hlo->operands(), ", ", [](string* out, const HloInstruction* operand) {
          tensorflow::strings::StrAppend(
              out, ShapeUtil::HumanString(operand->shape()));
        });
    // Spit `op_name(params...) -> result_type :: path` to stdout.
    std::cout << tensorflow::strings::Printf(
        "%s :: (%s) -> %s :: %s\n", HloOpcodeString(hlo->opcode()).c_str(),
        params.c_str(), ShapeUtil::HumanString(hlo->shape()).c_str(),
        path_.c_str());
    return Status::OK();
  }

 private:
  string path_;
};

void RealMain(tensorflow::gtl::ArraySlice<char*> args) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  LocalService* local_service =
      ClientLibrary::GetXlaService(client->platform());
  for (char* arg : args) {
    HloSnapshot snapshot;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), arg,
                                            &snapshot));
    auto computation_status = client->LoadSnapshot(snapshot);
    if (!computation_status.ok()) {
      fprintf(stderr, "could not load snapshot for %s: %s\n", arg,
              computation_status.status().ToString().c_str());
      continue;
    }
    XlaComputation computation = computation_status.ConsumeValueOrDie();

    std::unique_ptr<ProgramShape> program_shape =
        client->GetComputationShape(computation).ConsumeValueOrDie();

    std::vector<const Shape*> layouts;
    layouts.reserve(program_shape->parameters_size());
    for (int i = 0; i < program_shape->parameters_size(); ++i) {
      layouts.push_back(&program_shape->parameters(i));
    }
    ExecutableBuildOptions build_options;
    build_options.set_device_ordinal(0);
    build_options.set_result_layout(program_shape->result());
    StatusOr<std::unique_ptr<Executable>> executable =
        local_service->CompileExecutable(computation, layouts, build_options);

    const HloModule& module = executable.ValueOrDie()->module();

    OperationDumper dumper(arg);
    for (auto* computation : module.computations()) {
      TF_CHECK_OK(computation->Accept(&dumper));
    }
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::gtl::ArraySlice<char*> args(argv, argc);
  args.pop_front();  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args);
  return 0;
}
