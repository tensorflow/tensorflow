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

#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

using ::tensorflow::strings::StrAppend;

namespace xla {

namespace {
void DumpModule(const HloModule& module,

                const string& message) {
  hlo_graph_dumper::MaybeDumpHloModule(module, message);
  VLOG(2) << "HLO " << message << ":";
  XLA_VLOG_LINES(2, module.ToString());
}
}  // namespace

StatusOr<bool> HloPassPipeline::Run(HloModule* module) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline " << name();

  auto repeated_field =
      module->config().debug_options().xla_disable_hlo_passes();
  tensorflow::gtl::FlatSet<string> disabled_passes(repeated_field.begin(),
                                                   repeated_field.end());
  if (!disabled_passes.empty()) {
    VLOG(1) << "Passes disabled by --xla_disable_hlo_passes: "
            << tensorflow::str_util::Join(disabled_passes, ", ");
  }

  auto run_invariant_checkers = [this, module]() -> Status {
    for (auto& invariant_checker : invariant_checkers_) {
      TF_ASSIGN_OR_RETURN(bool changed, invariant_checker->Run(module));
      TF_RET_CHECK(!changed) << "invariant checkers must not change the graph";
    }
    return Status::OK();
  };

  string prefix = name().ToString() + ": pipeline start";
  bool changed = false;
  string message;
  for (auto& pass : passes_) {
    if (disabled_passes.count(pass->name().ToString()) > 0) {
      VLOG(1) << "  Skipping HLO pass " << pass->name()
              << ", disabled by --xla_disable_hlo_passes";
      continue;
    }

    VLOG(1) << "  HLO pass " << pass->name();

    // Emit label containing: "after foo-pass, before bar-pass".
    message.clear();
    StrAppend(&message, prefix, ", before ", pass->name());
    DumpModule(*module, message);

    TF_RETURN_IF_ERROR(run_invariant_checkers());
    TF_ASSIGN_OR_RETURN(bool changed_this_pass, pass->Run(module));

    changed |= changed_this_pass;
    prefix.clear();
    StrAppend(&prefix, name(), ": after ", pass->name());
  }
  TF_RETURN_IF_ERROR(run_invariant_checkers());
  DumpModule(*module, prefix + ", pipeline end");
  return changed;
}

}  // namespace xla
