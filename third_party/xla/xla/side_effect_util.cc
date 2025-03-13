/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/side_effect_util.h"

namespace xla {

const char kXlaHostTransferRendezvousNameAttr[] =
    "_xla_host_transfer_rendezvous";

const char kXlaHostTransferHandlerNameAttr[] =
    "_xla_host_transfer_handler_name";

const char kXlaHostTransferTfRendezvousHandlerName[] = "tf_rendezvous";

const char kXlaComputeTypeAttr[] = "_xla_compute_type";

const char kXlaComputeTypeSparse[] = "sparse";

const char kXlaComputeTypeDense[] = "dense";

const char kXlaComputeTypeHost[] = "host";

const char kXlaMaxIdsPerPartitionAttr[] = "_xla_max_ids_per_partition";

const char kXlaMaxUniqueIdsPerPartitionAttr[] =
    "_xla_max_unique_ids_per_partition";

const char kXlaShardingStrategyAttr[] = "_xla_sharding_strategy";

const char kXlaShardingStrategyMod[] = "mod";

const char kXlaShardingStrategyDiv[] = "div";

const char kXlaPadValueAttr[] = "_xla_pad_value";

const char kXlaQuantizationHighValueAttr[] = "_xla_quantization_high_value";

const char kXlaQuantizationLowValueAttr[] = "_xla_quantization_low_value";

const char kXlaQuantizationNumBucketsValueAttr[] =
    "_xla_quantization_num_buckets_value";

const char kXlaTableId[] = "_xla_table_id";

const char kXlaBufferPlacementAttr[] = "_xla_buffer_placement";

const char kXlaBufferPlacementParam[] = "arg";

const char kXlaStreamAnnotationAttr[] = "_xla_stream_annotation";

const char kXlaCollectiveMatmulAttr[] = "_xla_collective_matmul";

const char kXlaCollectiveMatmulLhsAg[] = "lhs_ag";

const char kXlaCollectiveMatmulRhsAg[] = "rhs_ag";

const char kXlaCollectiveMatmulRs[] = "rs";

const char kXlaCollectiveMatmulNone[] = "none";

const char kXlaMultiRecvCountAttr[] = "_xla_multi_recv_count";

const char kXlaSchedulingGroupIdAttr[] = "_scheduling_group_id";

const char kMustFuseAttr[] = "MUST_FUSE";

const char kMaximalFuseAttr[] = "MAXIMAL_FUSE";

}  // namespace xla
