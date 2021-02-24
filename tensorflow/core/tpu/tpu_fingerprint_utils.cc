/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/tpu_fingerprint_utils.h"

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"

namespace tensorflow {

Status FingerprintFunctionLibrary(const FunctionLibraryDefinition& library,
                                  uint64* fingerprint) {
  // TODO(phawkins): rather than fingerprinting the entire function library,
  // consider fingerprinting just the transitive dependencies of a
  // computation.
  std::string serialized;
  FunctionDefLibrary library_proto = library.ToProto();
  if (library_proto.ByteSizeLong() >= 1.5 * 1024 * 1024 * 1024) {
    LOG(WARNING) << "Serializing large proto, size: "
                 << library_proto.ByteSizeLong();
  }
  TF_RET_CHECK(SerializeToStringDeterministic(library_proto, &serialized));
  *fingerprint = TpuCompileInterface::Get()->FingerprintString(serialized);
  return Status::OK();
}

}  // namespace tensorflow
