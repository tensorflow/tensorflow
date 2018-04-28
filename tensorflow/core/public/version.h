/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PUBLIC_VERSION_H_
#define TENSORFLOW_CORE_PUBLIC_VERSION_H_

// TensorFlow uses semantic versioning, see http://semver.org/.

#define TF_MAJOR_VERSION 1
#define TF_MINOR_VERSION 8
#define TF_PATCH_VERSION 0

// TF_VERSION_SUFFIX is non-empty for pre-releases (e.g. "-alpha", "-alpha.1",
// "-beta", "-rc", "-rc.1")
#define TF_VERSION_SUFFIX "-rc1"

#define TF_STR_HELPER(x) #x
#define TF_STR(x) TF_STR_HELPER(x)

// e.g. "0.5.0" or "0.6.0-alpha".
#define TF_VERSION_STRING                                            \
  (TF_STR(TF_MAJOR_VERSION) "." TF_STR(TF_MINOR_VERSION) "." TF_STR( \
      TF_PATCH_VERSION) TF_VERSION_SUFFIX)

// TODO(josh11b): Public API functions for exporting the above.

// GraphDef compatibility versions (the versions field in graph.proto).
//
// Each graph has producer and min_consumer versions, and each
// consumer has its own version and a min_producer.  In addition, graphs can
// mark specific consumer versions as bad (to prevent bugs from executing).
// A consumer will execute a graph if the consumer's version is at least the
// graph's min_consumer, the graph's producer version is at least the consumer's
// min_producer, and the consumer version isn't specifically disallowed by the
// graph.
//
// By default, newly created graphs have producer version TF_GRAPH_DEF_VERSION
// min_consumer TF_GRAPH_DEF_MIN_CONSUMER, and no other bad consumer versions.
//
// Version history:
//
// 0. Graphs created before GraphDef versioning
// 1. First real version (2dec2015)
// 2. adjust_contrast only takes float, doesn't perform clamping (11dec2015)
// 3. Remove TileGrad, since it was equivalent to reduce_sum (30dec2015)
// 4. When support for this version is removed, we can safely make AttrValue
//    parsing more strict with respect to empty list values (see
//    111635679, 7jan2016).
// 5. Graphs are wholly-validated during Session::Create() (7jan2016).
// 6. TensorFlow is scalar strict within Google (27jan2016).
// 7. Remove TopK in favor of TopKV2 (5feb2016).
// 8. Replace RandomCrop from C++ with pure Python (5feb2016).
// 9. Deprecate batch_norm_with_global_normalization (16feb2016).
// 10. Deprecate conv3d_backprop_{filter,input} (10jun2016).
// 11. Deprecate {batch}_self_adjoint_eig (3aug2016).
// 12. Graph consumers understand the node_def field of FunctionDef (22aug2016).
// 13. Deprecate multiple batch linear algebra ops (9sep2016).
// 14. Deprecate batch_matrix_* ops. (10sep2016).
// 15. Deprecate batch_fft_* ops. (14sep2016).
// 16. Deprecate tensor_array (v1) ops in favor of v2 (10nov2016).
// 17. Deprecate inv (11nov2016).
// 17. Expose reverse_v2 (10nov2016)
// 18. Add VariableV2 (30nov2016)
// 19. Deprecated ops created by models moved out of core SkipGram, NegTrain.
//     (08dec2016)
// 20. Catch all version 1.0 changes to Python API generation. SplitV is now
//     used for tf.split, ReverseV2 is now used by tf.reverse, ConcatV2 is
//     now used by tf.concat. Graphs use flooring
//     division and mod semantics. TensorArrayV3. (12dec2016)
//     Also considered the version for when it is required for reduction
//     ops' indices to be scalar or vector, and not higher rank.
//     Some earlier graph def versions allowed this.
// 21. Dropped FunctionDef.Node support, switched to node_def introduced
//     in version 12. (11jan2017)
// 22. Placeholder now can specify and enforce scalar and partial
//     shapes, particularly when restoring a graph from GraphDef
//     produced at version 22 or later.  (04/10/2016)
// 23. Remove NonMaxSuppression in favor of NonMaxSuppressionV2.
// 24. Deprecate lookup ops (v1) ops in favor of v2 (30may2017)
// 25. Deprecate stack (v1) ops in favor of v2 (2017/6/15).
// 25. Deprecate RandomPoisson (v1) ops in favor of v2 (2017/10/25).
// 26. Add a bool 'stripped_default_attrs' to MetaInfoDef indicating
//     whether default-valued attrs have been stripped from the nodes in the
//     GraphDef. (7dec2017)
// 27. Deprecate TensorArray ops v2 in favor of v3 and deprecated io_ops
//     deprecated in favor of V2 ops. (2018/01/23)

#define TF_GRAPH_DEF_VERSION_MIN_PRODUCER 0
#define TF_GRAPH_DEF_VERSION_MIN_CONSUMER 0
#define TF_GRAPH_DEF_VERSION 26

// Checkpoint compatibility versions (the versions field in SavedSliceMeta).
//
// The checkpoint versions have the same semantics as GraphDef versions, but the
// numbering scheme is separate.  We have no plans to ever deprecate checkpoint
// versions, but it's good to have this in place in case we ever need to.
//
// Version history:
//
// 0. Checkpoints saved before checkpoint versioning.
// 1. First real version (10feb2015).
#define TF_CHECKPOINT_VERSION_MIN_PRODUCER 0
#define TF_CHECKPOINT_VERSION_MIN_CONSUMER 0
#define TF_CHECKPOINT_VERSION 1

/// Version query functions (defined in generated version_info.cc)

// Host compiler version (declared elsewhere to be __VERSION__)
extern const char* tf_compiler_version();
// The git commit designator when tensorflow was built
// If no git repository, this will be "internal".
extern const char* tf_git_version();
// Value of the _GLIBCXX_USE_CXX11_ABI flag, or 0 if it's not set.
extern const int tf_cxx11_abi_flag();
// Returns 1 if build is monolithic, or 0 otherwise.
extern const int tf_monolithic_build();

#endif  // TENSORFLOW_CORE_PUBLIC_VERSION_H_
