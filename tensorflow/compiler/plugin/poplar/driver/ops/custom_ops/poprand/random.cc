/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/stream_executor/lib/statusor.h"

#include "absl/container/flat_hash_map.h"

#include <string>

namespace xla {
namespace poplarplugin {
namespace {
class TruncatedNormalOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) override {
    poplar::program::Program prog;

    TF_ASSIGN_OR_RETURN(prog,
                        TruncatedNormal(res, inst, output_shape, tensor_map));

    poplar::program::Sequence seq;
    seq.add(prog);
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poprand, TruncatedNormal, TruncatedNormalOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
