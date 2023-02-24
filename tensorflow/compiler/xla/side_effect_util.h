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

#ifndef TENSORFLOW_COMPILER_XLA_SIDE_EFFECT_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SIDE_EFFECT_UTIL_H_

namespace xla {

// XLA frontend attribute name which specifies TensorFlow rendezvous name.
extern const char kXlaHostTransferRendezvousNameAttr[];

// XLA frontend attribute name which specifies original host transfer type.
// Value is XLA primitive type in lower case.
extern const char kXlaHostTransferOriginalTypeAttr[];

// XLA frontend attribute name which specifies whether a host transfer
// instruction is lower bits for a splitted X64 host transfer. Value is "true"
// or "false".
extern const char kXlaHostTransferIsLowerBitsAttr[];

// XLA frontend attribute name which specifies the name of host side handler
// associates with this transfer.
extern const char kXlaHostTransferHandlerNameAttr[];

// XLA frontend attribute value of the name of TensorFlow Rendezvous Host
// Command Handler.
extern const char kXlaHostTransferTfRendezvousHandlerName[];

// XLA frontend attribute name which specifies the type of computation.
extern const char kXlaComputeTypeAttr[];

// XLA frontend attribute values for kXlaComputeTypeAttr
extern const char kXlaComputeTypeSparse[];
extern const char kXlaComputeTypeDense[];
extern const char kXlaComputeTypeHost[];

// XLA frontend attribute name for the maximum number of ids expected per
// partition *before* an input batch is partitioned.
extern const char kXlaMaxIdsPerPartitionAttr[];

// XLA frontend attribute name for the maximum number of unique ids expected per
// partition *after* an input batch is partitioned.
extern const char kXlaMaxUniqueIdsPerPartitionAttr[];

// XLA frontend attribute for how to assign ids to partitions.
extern const char kXlaShardingStrategyAttr[];

// XLA frontend attribute values for kXlaShardingStrategyAttr.
extern const char kXlaShardingStrategyMod[];
extern const char kXlaShardingStrategyDiv[];

// XLA frontend attribute for pad value.
extern const char kXlaPadValueAttr[];

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SIDE_EFFECT_UTIL_H_
