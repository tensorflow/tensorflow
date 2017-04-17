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

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {

void RealMain(tensorflow::gtl::ArraySlice<char*> args) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  LocalService* local_service =
      ClientLibrary::GetXlaService(client->platform());
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

    std::unique_ptr<ProgramShape> program_shape =
        client->GetComputationShape(computation).ConsumeValueOrDie();

    std::vector<const Shape*> layouts;
    for (int i = 0; i < program_shape->parameters_size(); ++i) {
      layouts.push_back(&program_shape->parameters(i));
    }
    StatusOr<std::unique_ptr<Executable>> executable =
        local_service->CompileExecutable(
            computation.handle(), layouts, &program_shape->result(),
            /*device_ordinal=*/0, /*has_hybrid_result=*/true);

    const HloModule& module = executable.ValueOrDie()->module();

    fprintf(stdout, "HLO for %s backend:\n%s\n",
            local_service->backend().platform()->Name().c_str(),
            module.ToString().c_str());
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
