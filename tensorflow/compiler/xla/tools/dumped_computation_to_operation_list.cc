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

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {

class OperationDumper : public DfsHloVisitorWithDefault {
 public:
  explicit OperationDumper(const std::string& path) : path_(path) {}

  Status DefaultAction(HloInstruction* hlo) override {
    std::string params = absl::StrJoin(
        hlo->operands(), ", ",
        [](std::string* out, const HloInstruction* operand) {
          absl::StrAppend(out, ShapeUtil::HumanString(operand->shape()));
        });
    // Spit `op_name(params...) -> result_type :: path` to stdout.
    std::cout << absl::StrFormat("%s :: (%s) -> %s :: %s\n",
                                 HloOpcodeString(hlo->opcode()), params,
                                 ShapeUtil::HumanString(hlo->shape()), path_);
    return ::tensorflow::OkStatus();
  }

 private:
  std::string path_;
};

void RealMain(absl::Span<char* const> args) {
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
    auto executables =
        local_service->CompileExecutables(computation, layouts, build_options)
            .ConsumeValueOrDie();
    CHECK_EQ(executables.size(), 1);
    const HloModule& module = executables[0]->module();

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

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args);
  return 0;
}
