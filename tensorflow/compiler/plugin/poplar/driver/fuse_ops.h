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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_FUSE_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_FUSE_OPS_H_

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"

namespace xla {

namespace poplarplugin {

static const int FUSED_BASE =
        static_cast<int>(HloInstruction::FusionKind::kCustom);

static const int FUSED_SLICE_UPDATE = FUSED_BASE + 0;
static const int FUSED_SLICE = FUSED_BASE + 1;
static const int FUSED_RELU = FUSED_BASE + 2;
static const int FUSED_SIGMOID = FUSED_BASE + 3;
static const int FUSED_BIASADD_BROADCAST = FUSED_BASE + 4;
static const int FUSED_BIASADD = FUSED_BASE + 5;
static const int FUSED_ZERO_PAD = FUSED_BASE + 6;
static const int FUSED_TRUNCATED_NORMAL_WITH_SCALE = FUSED_BASE + 7;
static const int FUSED_TRUNCATED_NORMAL = FUSED_BASE + 8;
static const int FUSED_RANDOM_NORMAL_WITH_SCALE = FUSED_BASE + 9;
static const int FUSED_RANDOM_UNIFORM_WITH_SCALE = FUSED_BASE + 10;
static const int FUSED_RANDOM_NORMAL = FUSED_BASE + 11;
static const int FUSED_RANDOM_UNIFORM = FUSED_BASE + 12;
static const int FUSED_BERNOULLI = FUSED_BASE + 13;
static const int FUSED_AVG_POOL_SAME = FUSED_BASE + 14;
static const int FUSED_AVG_POOL_VALID = FUSED_BASE + 15;

class FuseOps : public HloMatcher {
public:
  FuseOps();

  ~FuseOps() override = default;

  tensorflow::StringPiece name() const override { return "poplar-fuse"; }

  ReplacedInstructions ReplaceNodes(int pattern,
                                    const HloMatcherMatched& match) override;

};

}
}

#endif
