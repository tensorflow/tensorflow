/* Copyright 2024 The OpenXLA Authors.

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
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/service/gpu/fusions/tools/test_lib.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status Run(const std::string& filename) {
  TF_ASSIGN_OR_RETURN(auto module, LoadTestModule(filename));
  TF_ASSIGN_OR_RETURN(auto emitter_data, GetMlirFusionEmitter(*module));

  auto context = GetMlirContextForTest();
  TF_ASSIGN_OR_RETURN(auto mlir_module,
                      emitter_data->emitter->CreateMLIRModule(
                          context, *emitter_data->fusion, "main",
                          /*buffer_assignment=*/nullptr));
  llvm::outs() << *mlir_module;
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  CHECK_EQ(argc, 2) << "Must specify an input file";
  CHECK_OK(xla::gpu::Run(argv[1]));
  return 0;
}
