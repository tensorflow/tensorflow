/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_CONSTANTS_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_CONSTANTS_H_

namespace tensorflow {
namespace tfprof {

// Op name of root of everything. Aggregates all stats.
static const char* const kTFProfRoot = "_TFProfRoot";
// Op type for nodes that doesn't represent a physical node in the
// TensorFlow model. Only exist as a placehold to aggregate children.
// For example, kTFProfRoot belongs to this type.
static const char* const kTFGraphParent = "_TFGraphParent";
static const char* const kTFScopeParent = "_kTFScopeParent";
// Op type for tf.trainable_variables().
static const char* const kTrainableVarType = "_trainable_variables";
// Op type for tensors in the checkpoint file.
static const char* const kCkptVarType = "_checkpoint_variables";

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_CONSTANTS_H_
