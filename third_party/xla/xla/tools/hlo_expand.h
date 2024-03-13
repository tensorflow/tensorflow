/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_HLO_EXPAND_H_
#define XLA_TOOLS_HLO_EXPAND_H_

#include <string>
#include <vector>

#include "xla/service/hlo_pass_pipeline.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla {

// Command-line options to this tool. See hlo_opt.cc for the descriptions of
// these fields.
struct HloExpandConfig {
  // Optional flags.
  bool help{false};
  std::string input_format;
  std::string output_file;
  std::string output_format;
  // Compound flags setting multiple passes.
  bool batch_norm_expander{false};
  bool expand_all{false};
  bool rng_bit_generator_expander{false};
  // Flags for individual passes.
  bool batch_norm_grad_expander{false};
  bool batch_norm_inference_expander{false};
  bool batch_norm_training_expander{false};
  bool cholesky_expander{false};
  bool rng_expander{false};
  bool rng_bit_generator_philox_expander{false};
  bool rng_bit_generator_three_fry_expander{false};
  bool triangular_solve_expander{false};
  bool spmd_expander{false};
  bool verify_hlo{false};
};

// Adds passes to the `pipeline` for flags set in `config`.
void AddPassesToPipeline(xla::HloExpandConfig& config,
                         xla::HloPassPipeline& pipeline,
                         const xla::HloModuleConfig& hlo_module_config);

// Wraps `config` with flag descriptions and returns a vector of `tsl::Flag`s.
std::vector<tsl::Flag> GetFlags(xla::HloExpandConfig& config);

// Parses compound flags that sets multiple flags from `config` and overrides
// individual flags that were set previously.
void ParseCompoundFlags(xla::HloExpandConfig& config);

}  // namespace xla

#endif  // XLA_TOOLS_HLO_EXPAND_H_
