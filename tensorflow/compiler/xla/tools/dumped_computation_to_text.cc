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

#include <stdio.h>
#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {

void RealMain(absl::Span<char* const> args, bool compile) {
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

    if (compile) {
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

      fprintf(stdout, "HLO compiled for %s backend:\n%s\n",
              local_service->backend().platform()->Name().c_str(),
              module.ToString(HloPrintOptions::ShortParsable()).c_str());
    } else {
      auto config = HloModule::CreateModuleConfigFromProto(computation.proto(),
                                                           DebugOptions())
                        .ConsumeValueOrDie();
      std::unique_ptr<HloModule> module =
          HloModule::CreateFromProto(computation.proto(), config)
              .ConsumeValueOrDie();

      fprintf(stdout, "%s\n",
              module->ToString(HloPrintOptions::ShortParsable()).c_str());
    }
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  bool compile = false;
  std::vector<tensorflow::Flag> flag_list = {
      {"compile", &compile,
       "If true, compile the computation using the default client before "
       "dumping the HLO. Otherwise dump the raw (uncompiled) HLO."},
  };
  const xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc > 1) << "\nERROR: must specify at least one module\n" << usage;

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args, compile);
  return 0;
}
