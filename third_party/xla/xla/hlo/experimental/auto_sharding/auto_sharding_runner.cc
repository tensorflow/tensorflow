/* Copyright 2022 The OpenXLA Authors.

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

#include <iostream>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_parser.h"
#include "xla/tools/hlo_module_loader.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace spmd {
namespace {

absl::Status RunAutoShardingPassFromFile(const std::string& file_name) {
  std::string hlo_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), file_name, &hlo_text));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      LoadModuleFromData(/*data=*/hlo_text, /*format=*/"hlo"));

  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSIGN_OR_RETURN(bool changed, AutoSharding(option).Run(hlo_module.get()));
  CHECK(changed);
  std::cout << hlo_module->ToString() << std::endl;
  return absl::OkStatus();
}

}  // namespace
}  // namespace spmd
}  // namespace xla

int main(int argc, char** argv) {
  tsl::port::InitMain("Run AutoSharding Pass", &argc, &argv);
  QCHECK(argc == 2) << "Must specify a single input file";
  TF_CHECK_OK(xla::spmd::RunAutoShardingPassFromFile(argv[1]));
  return 0;
}
