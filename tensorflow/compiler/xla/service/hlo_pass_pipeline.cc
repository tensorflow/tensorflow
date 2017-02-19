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

#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"

#include <functional>

#include "tensorflow/compiler/xla/legacy_flags/hlo_pass_pipeline_flags.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {
void DumpModule(const Compiler::HloDumper& dumper_, const HloModule& module,
                const string& message) {
  dumper_(module, message);
  VLOG(2) << "HLO " << message << ":";
  XLA_VLOG_LINES(2, module.ToString());
}
}  // namespace

StatusOr<bool> HloPassPipeline::Run(HloModule* module) {
  legacy_flags::HloPassPipelineFlags* flags =
      legacy_flags::GetHloPassPipelineFlags();
  std::vector<string> tmp =
      tensorflow::str_util::Split(flags->xla_disable_hlo_passes, ',');
  tensorflow::gtl::FlatSet<string> disabled_passes(tmp.begin(), tmp.end());

  string prefix = name().ToString() + ": pipeline start";
  bool changed = false;
  string message;
  for (auto& pass : passes_) {
    if (!disabled_passes.empty() &&
        disabled_passes.count(pass->name().ToString()) > 0) {
      continue;
    }

    // Emit label containing: "after foo-pass, before bar-pass".
    message.clear();
    tensorflow::strings::StrAppend(&message, prefix, ", before ", pass->name());
    DumpModule(dumper_, *module, message);

    TF_ASSIGN_OR_RETURN(bool changed_this_pass, pass->Run(module));

    changed |= changed_this_pass;
    prefix.clear();
    tensorflow::strings::StrAppend(&prefix, name(), ": after ", pass->name());
  }
  DumpModule(dumper_, *module, prefix + ", pipeline end");
  return changed;
}

}  // namespace xla
