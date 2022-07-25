/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/side_effect_util.h"

namespace xla {

const char kXlaHostTransferRendezvousNameAttr[] =
    "_xla_host_transfer_rendezvous";

const char kXlaHostTransferOriginalTypeAttr[] =
    "_xla_host_transfer_original_type";

const char kXlaHostTransferIsLowerBitsAttr[] =
    "_xla_host_transfer_is_lower_bits";

const char kXlaHostTransferHandlerNameAttr[] =
    "_xla_host_transfer_handler_name";

const char kXlaHostTransferTfRendezvousHandlerName[] = "tf_rendezvous";

const char kXlaComputeTypeAttr[] = "_xla_compute_type";

const char kXlaComputeTypeSparse[] = "sparse";

const char kXlaComputeTypeDense[] = "dense";

const char kXlaMaxIdsPerPartitionAttr[] = "_xla_max_ids_per_partition";

const char kXlaMaxUniqueIdsPerPartitionAttr[] =
    "_xla_max_unique_ids_per_partition";

const char kXlaShardingStrategyAttr[] = "_xla_sharding_strategy";

const char kXlaShardingStrategyMod[] = "mod";

const char kXlaShardingStrategyDiv[] = "div";

}  // namespace xla
