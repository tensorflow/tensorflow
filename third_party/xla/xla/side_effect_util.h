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

#ifndef XLA_SIDE_EFFECT_UTIL_H_
#define XLA_SIDE_EFFECT_UTIL_H_

namespace xla {

// XLA frontend attribute name which specifies TensorFlow rendezvous name.
extern const char kXlaHostTransferRendezvousNameAttr[];

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

// XLA frontend attribute name for the maximum valency of a sample. Currently
// only used for the custom combiner coarse-grain op.
extern const char kXlaMaxValencyAttr[];

// XLA frontend attribute for how to assign ids to partitions.
extern const char kXlaShardingStrategyAttr[];

// XLA frontend attribute values for kXlaShardingStrategyAttr.
extern const char kXlaShardingStrategyMod[];
extern const char kXlaShardingStrategyDiv[];

// XLA frontend attribute for pad value.
extern const char kXlaPadValueAttr[];

// XLA frontend attributes for simulated quantization.
extern const char kXlaQuantizationHighValueAttr[];
extern const char kXlaQuantizationLowValueAttr[];
extern const char kXlaQuantizationNumBucketsValueAttr[];

// XLA frontend attribute for table id.
extern const char kXlaTableId[];

// XLA frontend attribute for buffer placement.
extern const char kXlaBufferPlacementAttr[];
extern const char kXlaBufferPlacementParam[];

// XLA frontend attribute for stream annotation.
extern const char kXlaStreamAnnotationAttr[];

// XLA frontend attribute for collective matmul control.
extern const char kXlaCollectiveMatmulAttr[];

// XLA frontend attribute values for kXlaCollectiveMatmulAttr
extern const char kXlaCollectiveMatmulLhsAg[];
extern const char kXlaCollectiveMatmulRhsAg[];
extern const char kXlaCollectiveMatmulRs[];
extern const char kXlaCollectiveMatmulNone[];

// XLA frontend attribute for specifying the number of sends this recv should
// match.
extern const char kXlaMultiRecvCountAttr[];

// XLA frontend attribute for specifying the scheduling group id annotations.
extern const char kXlaSchedulingGroupIdAttr[];
// XLA frontend attribute value for a group that will not actually be scheduled.
extern const char kXlaNoOpSchedulingGroup[];

// XLA frontend attributes for specifying fusion directives.
// MUST_FUSE: all ops labeled so should form as single fusion,
// MAXIMAL_FUSE: all ops labeled should be in a fusion, but can be split among
// multiple fusions, else, the compiler will return errors.
// TODO(b/366060148): Currently, the JAX framework has not finalized on the
// frontend attribute name for MUST_FUSE and MAXIMAL_FUSE. Update this code
// once the name is finalized and any additional attributes related to fusion
// are added.
extern const char kMustFuseAttr[];
extern const char kMaximalFuseAttr[];
extern const char kFuseLimitAttr[];

// XLA frontend attribute for specifying groups of collectives that should be
// launched together.
extern const char kCollectivesGroupAttr[];
}  // namespace xla

#endif  // XLA_SIDE_EFFECT_UTIL_H_
