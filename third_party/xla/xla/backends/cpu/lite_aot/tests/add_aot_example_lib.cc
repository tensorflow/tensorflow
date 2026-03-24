/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/lite_aot/tests/add_aot_example_lib.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/lite_aot/xla_aot_function.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::cpu {

namespace {

std::string GetRootDir() {
  std::string root_dir;
  CHECK(tsl::io::GetTestWorkspaceDir(&root_dir));

  std::string path_to_data_dir =
      tsl::kIsOpenSource
          ? "xla/backends/cpu/lite_aot/tests"
          : "third_party/tensorflow/compiler/xla/backends/cpu/lite_aot/tests";

  return tsl::io::JoinPath(root_dir, path_to_data_dir);
}

}  // namespace

absl::StatusOr<std::unique_ptr<XlaAotFunction>> GetAddAotFunction() {
  CompilationResultProto proto;
  RETURN_IF_ERROR(tsl::ReadBinaryProto(
      tsl::Env::Default(), tsl::io::JoinPath(GetRootDir(), "add_aot"), &proto));
  return XlaAotFunction::Create(std::move(proto));
}

}  // namespace xla::cpu
