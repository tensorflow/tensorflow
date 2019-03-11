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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#include <string>
#include "absl/container/flat_hash_map.h"

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;
class HloCustomCallInstruction;
struct TensorTarget;
namespace poplarplugin {

class PoplibsOpDef {
 public:
  PoplibsOpDef() = default;
  // By default the op is not allocating.
  virtual StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map,
      const TensorMap& tensor_map) {
    return xla::FailedPrecondition(
        "Non-allocating op should not be allocating.");
  }

  virtual StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) = 0;
};

// The following singleton class is used to register and access custom poplibs
// ops.
class PoplibsOpManager {
 public:
  // Registration method
  static void RegsiterOp(PoplibsOp::Lib lib, PoplibsOp::Op op,
                         std::unique_ptr<PoplibsOpDef> poplibs_op_def);
  static StatusOr<PoplibsOpDef*> GetOp(const HloCustomCallInstruction* inst);

 private:
  PoplibsOpManager() = default;
  static PoplibsOpManager& GetInstance();

  absl::flat_hash_map<std::pair<PoplibsOp::Lib, PoplibsOp::Op>,
                      std::unique_ptr<PoplibsOpDef>>
      ops;
};

class PoplibsOpRegistrar {
 public:
  PoplibsOpRegistrar(PoplibsOp::Lib lib, PoplibsOp::Op op,
                     std::unique_ptr<PoplibsOpDef> poplibs_op_def);

  PoplibsOpRegistrar() = delete;
};

#define REGISTER_POPLIBS_OP(poplibs_lib, poplibs_op, poplibs_op_def)  \
  namespace {                                                         \
  static PoplibsOpRegistrar                                           \
      registrar__poplibs_op__##poplibs_lib##__##poplibs_op##__object( \
          PoplibsOp::poplibs_lib, PoplibsOp::poplibs_op,              \
          std::unique_ptr<PoplibsOpDef>(new poplibs_op_def));         \
  }

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_
