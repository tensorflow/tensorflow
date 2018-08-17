/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_wide_const.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static const std::vector<FusedGraphInfo> fuse_info = {
    {"wide_const", 1},
};

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

static const std::vector<HloMatcherPattern> patterns = {
    // Broadcast scalar constant (must be low priority)
    {{HloOpcode::kBroadcast, true, 0, nullptr, {1}},
     {HloOpcode::kConstant, true, 0, IsScalarConstant, {}}},
};

FuseWideConst::FuseWideConst(struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, fuse_info, "_pop_op_") {}

}  // namespace poplarplugin
}  // namespace xla
