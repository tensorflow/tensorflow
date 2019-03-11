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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace xla {
namespace poplarplugin {

void PoplibsOpManager::RegsiterOp(
    PoplibsOp::Lib lib, PoplibsOp::Op op,
    std::unique_ptr<PoplibsOpDef> poplibs_op_def) {
  auto& ops = GetInstance().ops;
  auto key = std::make_pair(lib, op);
  if (ops.contains(key)) {
    LOG(FATAL) << "Trying to register the same op twice ("
               << PoplibsOp_Lib_Name(lib) << ", " << PoplibsOp_Op_Name(op)
               << ").";
  }
  ops[key] = std::move(poplibs_op_def);
}

PoplibsOpManager& PoplibsOpManager::GetInstance() {
  static PoplibsOpManager instance;
  return instance;
}

StatusOr<PoplibsOpDef*> PoplibsOpManager::GetOp(
    const HloCustomCallInstruction* inst) {
  // Find the poplibs info given a CustomCall instruction.
  auto ret = GetPoplibsCustomOp(inst);
  if (!ret) {
    return xla::FailedPrecondition("Could not find custom call target %s.",
                                   inst->custom_call_target().c_str());
  }

  auto& ops = GetInstance().ops;
  auto itr = ops.find(*ret);
  if (itr != ops.end()) {
    return itr->second.get();
  }
  return xla::FailedPrecondition("Could not find definition for %s::%s.",
                                 PoplibsOp_Lib_Name(ret->first).c_str(),
                                 PoplibsOp_Op_Name(ret->second).c_str());
}

PoplibsOpRegistrar::PoplibsOpRegistrar(
    PoplibsOp::Lib lib, PoplibsOp::Op op,
    std::unique_ptr<PoplibsOpDef> poplibs_op_def) {
  PoplibsOpManager::RegsiterOp(lib, op, std::move(poplibs_op_def));
}

}  // namespace poplarplugin
}  // namespace xla
