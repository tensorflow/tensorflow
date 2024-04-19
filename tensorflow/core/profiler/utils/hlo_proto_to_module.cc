/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hlo_proto_to_module.h"

#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/util.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace profiler {

absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertHloProtoToModule(
    const xla::HloProto& hlo_proto) {
  if (!hlo_proto.has_hlo_module()) {
    return xla::Internal("No HLO module found in the HLO proto");
  }
  const xla::HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(auto config, xla::HloModule::CreateModuleConfigFromProto(
                                       module_proto, xla::DebugOptions()));
  TF_ASSIGN_OR_RETURN(auto module,
                      xla::HloModule::CreateFromProto(module_proto, config));
  return module;
}

std::unique_ptr<xla::HloModule> ConvertHloProtoToModuleIgnoringErrors(
    const xla::HloProto& hlo_proto) {
  auto module = ConvertHloProtoToModule(hlo_proto);
  if (!module.ok()) {
    LOG(ERROR) << module.status();
    return nullptr;
  }
  return std::move(module).value();
}

}  // namespace profiler
}  // namespace tensorflow
