/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/stream_executor_executable.h"

#include <memory>
#include <string>

#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/compiler.h"
#include "xla/statusor.h"
#include "tsl/platform/statusor.h"

namespace xla {
StatusOr<std::string> StreamExecutorExecutable::SerializeExecutable() const {
  StreamExecutorExecutableProto proto;
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  for (const std::unique_ptr<xla::AotCompilationResult>& aot_executable :
       aot_executables_) {
    TF_ASSIGN_OR_RETURN(*proto.add_executables(),
                        aot_executable->SerializeAsString());
  }
  proto.set_num_replicas(num_replicas_);
  proto.set_num_partitions(num_partitions_);
  proto.set_name(name_);
  proto.set_fingerprint(fingerprint_);
  return proto.SerializeAsString();
}
}  // namespace xla
