/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_HOST_OFFLOADING_PREPARE_H_
#define XLA_HLO_TRANSFORMS_HOST_OFFLOADING_PREPARE_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// This is a collection of rewrites that prepares an HLO module for host
// offloading. These rewrites can be placed in a different parts of
// the overall compilation pipeline to prepare HLO module for host offloading
// for the given backend.
class HostOffloadingPrepare : public HloModulePass {
 public:
  enum class Rewrite {
    // This rewrite removes `MoveToHost` custom calls that feed directly into
    // the a host computation.
    //
    // In the HLO, it will look like HBM is directly fed into the host
    // computation. The runtime will, once the async-call-start is executed,
    // allocate a buffer on the host and copy the HBM buffer into it. This has
    // the benefit that the device will never be blocking directly on the
    // tranfser, since that's clumped together with the computation.
    kElideMoveToHost,

    // Currently host compute offloading does not support tiled layouts, and
    // because of that layouts on the call instruction arguments might be
    // different from the layouts in the called computation body.
    //
    // Host offloading handles layout mismatches at run time by delinearizing
    // arguments and linearizing results on the fly.
    //
    // To keep HLO module valid we rewrite calls to host offloaded computations
    // into custom calls with the only purpose to suppress verification error.
    // Host offloading compiler later does its own verification to check that
    // arguments are compatible with parameters in the offloaded computation and
    // knows how to handle mismatched layouts.
    kConvertToCustomCall,
  };

  static std::string RewriteName(Rewrite rewrite) {
    switch (rewrite) {
      case Rewrite::kElideMoveToHost:
        return "elide-move-to-host";
      case Rewrite::kConvertToCustomCall:
        return "convert-to-custom-call";
    }
  }

  explicit HostOffloadingPrepare(Rewrite rewrite)
      : rewrite_(rewrite),
        pass_name_(absl::StrCat("host-offloading-prepare", "-",
                                RewriteName(rewrite_))) {}

  absl::string_view name() const override { return pass_name_; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Rewrite rewrite_;
  std::string pass_name_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_HOST_OFFLOADING_PREPARE_H_
