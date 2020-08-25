/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_support.h"

#if defined(LIBTFTPU)
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache.pb.h"
#endif  // LIBTFTPU

namespace tensorflow {
namespace tpu {
std::shared_ptr<::grpc::ChannelCredentials> CreateChannelCredentials() {
  return ::grpc::InsecureChannelCredentials();
}

#if defined(LIBTFTPU)
template <>
Status FillCacheEntryFromGetTpuProgramResponse<GetTpuProgramResponseExternal>(
    absl::string_view local_proto_key, GetTpuProgramResponseExternal* response,
    std::shared_ptr<CacheEntry>* cache_entry) {
  // TODO(b/162904194): implement this method.
  LOG(FATAL) << "Not implemented yet.";
}

void SendGetTpuProgramResponseHelper(
    const TpuCompilationCacheEntry& cache_entry,
    std::function<void(::grpc::ByteBuffer*, ::grpc::Status)> call_fn) {
  // TODO(b/162904194): implement this method.
  LOG(FATAL) << "Not implemented yet.";
}
#endif  // LIBTFTPU
}  // namespace tpu
}  // namespace tensorflow
