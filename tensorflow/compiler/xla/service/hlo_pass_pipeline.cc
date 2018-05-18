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
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

namespace xla {

namespace {
void DumpModuleGraph(const HloModule& module, const string& message) {
  hlo_graph_dumper::MaybeDumpHloModule(module, message);
  VLOG(3) << "HLO " << message << ":";
  XLA_VLOG_LINES(3, module.ToString());
}

void DumpModuleProto(const HloModule& module, const string& dump_to,
                     const string& pipeline_name, const string& pass_name) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static auto* const module_id_to_pass_number =
      new tensorflow::gtl::FlatMap<int64, int64>();

  tensorflow::mutex_lock lock(mu);
  const int64 pass_number = (*module_id_to_pass_number)[module.unique_id()]++;

  const string mod_name = SanitizeFileName(tensorflow::strings::Printf(
      "module_%04d.%04lld.%s.after_%s", module.unique_id(), pass_number,
      pipeline_name.c_str(), pass_name.c_str()));

  TF_QCHECK_OK(protobuf_util::DumpProtoToDirectory(MakeHloProto(module),
                                                   dump_to, mod_name));
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

  auto run_invariant_checkers = [this,
                                 module](const string& message) -> Status {
    for (auto& invariant_checker : invariant_checkers_) {
      VLOG(1) << "    Invariant checker " << invariant_checker->name();
      StatusOr<bool> changed_status = invariant_checker->Run(module);
      VLOG(1) << "    Invariant checker done " << invariant_checker->name();
      if (!changed_status.ok()) {
        VLOG(2) << "Module failed invariant check:";
        XLA_VLOG_LINES(2, module->ToString());
        return Status(changed_status.status().code(),
                      StrCat(changed_status.status().error_message(),
                             "\n\nFailed ", message));
      }
      TF_RET_CHECK(!changed_status.ValueOrDie())
          << "invariant checkers must not change the graph";
    }
    return Status::OK();
  };

  string prefix = std::string(name()) + ": pipeline start";
  bool changed = false;
  string message;
  TF_RETURN_IF_ERROR(
      run_invariant_checkers(StrCat("before running pipeline: ", name())));
  const string xla_dump_per_pass_hlo_proto_to =
      module->config().debug_options().xla_dump_per_pass_hlo_proto_to();
  if (!xla_dump_per_pass_hlo_proto_to.empty()) {
    DumpModuleProto(*module, xla_dump_per_pass_hlo_proto_to,
                    std::string(name()), "pipeline_start");
  }

  for (auto& pass : passes_) {
    if (disabled_passes.count(std::string(pass->name())) > 0) {
      VLOG(1) << "  Skipping HLO pass " << pass->name()
              << ", disabled by --xla_disable_hlo_passes";
      continue;
    }

    VLOG(1) << "  HLO pass " << pass->name();

    // Emit label containing: "after foo-pass, before bar-pass".
    message.clear();
    StrAppend(&message, prefix, ", before ", pass->name());
    DumpModuleGraph(*module, message);

    TF_ASSIGN_OR_RETURN(bool changed_this_pass, pass->Run(module));
    TF_RETURN_IF_ERROR(
        run_invariant_checkers(StrCat("after running pass: ", pass->name())));
    if (!xla_dump_per_pass_hlo_proto_to.empty()) {
      DumpModuleProto(*module, xla_dump_per_pass_hlo_proto_to,
                      std::string(name()), std::string(pass->name()));
    }

    changed |= changed_this_pass;
    prefix.clear();
    StrAppend(&prefix, name(), ": after ", pass->name());
  }
  DumpModuleGraph(*module, prefix + ", pipeline end");
  return changed;
}

}  // namespace xla
