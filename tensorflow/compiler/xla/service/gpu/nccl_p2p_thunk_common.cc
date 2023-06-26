/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nccl_p2p_thunk_common.h"

#include <utility>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

StatusOr<std::vector<std::pair<int64_t, int64_t>>> GetSourceTargetPairs(
    mlir::DictionaryAttr frontend_attributes) {
  mlir::StringAttr src_dst_string = frontend_attributes.getAs<mlir::StringAttr>(
      kSendRecvSourceTargetPairsAttr);
  if (!src_dst_string) {
    return absl::AbortedError(
        absl::StrCat("expecting send/recv op with string attribute ",
                     kSendRecvSourceTargetPairsAttr));
  }
  TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                      ParseReplicaGroupsOnly(src_dst_string.str()));
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
  source_target_pairs.reserve(replica_groups.size());
  for (const ReplicaGroup& replica_group : replica_groups) {
    TF_RET_CHECK(replica_group.replica_ids_size() == 2);
    source_target_pairs.emplace_back(replica_group.replica_ids(0),
                                     replica_group.replica_ids(1));
  }
  return source_target_pairs;
}

}  // namespace gpu
}  // namespace xla
