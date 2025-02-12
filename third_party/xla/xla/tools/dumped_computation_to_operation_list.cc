/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/local_service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {
namespace tools {

class OperationDumper : public DfsHloVisitorWithDefault {
 public:
  explicit OperationDumper(const std::string& path) : path_(path) {}

  absl::Status DefaultAction(HloInstruction* hlo) override {
    std::string params = absl::StrJoin(
        hlo->operands(), ", ",
        [](std::string* out, const HloInstruction* operand) {
          absl::StrAppend(out, ShapeUtil::HumanString(operand->shape()));
        });
    // Spit `op_name(params...) -> result_type :: path` to stdout.
    std::cout << absl::StrFormat("%s :: (%s) -> %s :: %s\n",
                                 HloOpcodeString(hlo->opcode()), params,
                                 ShapeUtil::HumanString(hlo->shape()), path_);
    return absl::OkStatus();
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
    TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), arg, &snapshot));
    auto computation_status = client->LoadSnapshot(snapshot);
    if (!computation_status.ok()) {
      fprintf(stderr, "could not load snapshot for %s: %s\n", arg,
              computation_status.status().ToString().c_str());
      continue;
    }
    XlaComputation computation = std::move(computation_status).value();

    std::unique_ptr<ProgramShape> program_shape =
        client->GetComputationShape(computation).value();

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
            .value();
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
  tsl::port::InitMain(argv[0], &argc, &argv);

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args);
  return 0;
}
