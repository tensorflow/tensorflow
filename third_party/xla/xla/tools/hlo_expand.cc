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

#include "xla/tools/hlo_expand.h"

#include <vector>

#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"

namespace xla {

void AddPassesToPipeline(HloExpandConfig& config, HloPassPipeline& pipeline,
                         const HloModuleConfig& hlo_module_config) {
  if (config.batch_norm_grad_expander || config.batch_norm_inference_expander ||
      config.batch_norm_training_expander) {
    pipeline.AddPass<xla::BatchNormExpander>(
        /*rewrite_training_op=*/config.batch_norm_training_expander,
        /*rewrite_inference_op=*/config.batch_norm_inference_expander,
        /*rewrite_grad_op=*/config.batch_norm_grad_expander);
  }
  if (config.cholesky_expander) {
    pipeline.AddPass<xla::CholeskyExpander>();
  }
  if (config.rng_expander) {
    pipeline.AddPass<xla::RngExpander>();
  }
  if (config.rng_bit_generator_philox_expander) {
    pipeline.AddPass<xla::RngBitGeneratorExpander>(
        xla::RandomAlgorithm::RNG_PHILOX);
  }
  if (config.rng_bit_generator_three_fry_expander) {
    pipeline.AddPass<xla::RngBitGeneratorExpander>(
        xla::RandomAlgorithm::RNG_THREE_FRY);
  }
  if (config.triangular_solve_expander) {
    pipeline.AddPass<xla::TriangularSolveExpander>();
  }
  if (config.spmd_expander) {
    pipeline.AddPass<ShardingPropagation>(
        /*is_spmd=*/true, /*propagate_metadata=*/false,
        hlo_module_config.allow_spmd_sharding_propagation_to_output(),
        hlo_module_config.allow_spmd_sharding_propagation_to_parameters());
    pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
        hlo_module_config.num_partitions(), hlo_module_config.replica_count(),
        hlo_module_config.debug_options()
            .xla_gpu_threshold_for_windowed_einsum_mib());
  }
  if (config.verify_hlo) {
    pipeline.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                       /*allow_mixed_precision=*/false);
  }
}

std::vector<tsl::Flag> GetFlags(HloExpandConfig& config) {
  return {
      tsl::Flag("h", &config.help, "Alias of --help"),
      tsl::Flag("help", &config.help, "Display available options"),
      tsl::Flag(
          "input_format", &config.input_format,
          "The format of the input file. If this flag is not specified, it's"
          "inferred from the file extension instead. Valid values:\n "
          "* hlo|txt : HLO textual format\n"
          "* pb      : xla::HloProto in binary proto format\n"
          "* pbtxt   : xla::HloProto in text proto format"),
      tsl::Flag("o", &config.output_file, "Alias of --output_file="),
      tsl::Flag("output_file", &config.output_file, "Full output file path"),
      tsl::Flag("output_format", &config.output_format,
                "The format of the output file. Defaults to input_format. "
                "Valid values:\n"
                "* hlo|txt : HLO textual format\n"
                "* pb      : xla::HloProto in binary proto format\n"
                "* pbtxt   : xla::HloProto in text proto format"),
      tsl::Flag("batch_norm_expander", &config.batch_norm_expander,
                "Overrides and expands batch_norm_grad, batch_norm_inference, "
                "and batch_norm_training ops"),
      tsl::Flag("batch_norm_grad_expander", &config.batch_norm_grad_expander,
                "Expands batch_norm_grad op"),
      tsl::Flag("batch_norm_inference_expander",
                &config.batch_norm_inference_expander,
                "Expands batch_norm_inference_grad op"),
      tsl::Flag("batch_norm_training_expander",
                &config.batch_norm_training_expander,
                "Expands batch_norm_training_grad op"),
      tsl::Flag("cholesky_expander", &config.cholesky_expander,
                "Expands cholesky op"),
      tsl::Flag("spmd_expander", &config.spmd_expander,
                "Expands SPMD sharding"),
      tsl::Flag("expand_all", &config.expand_all,
                "Overrides and expands all supported passes below"),
      tsl::Flag("rng_expander", &config.rng_expander, "Expands rng op"),
      tsl::Flag(
          "rng_bit_generator_expander", &config.rng_bit_generator_expander,
          "Overrides and expands rng_bit_generator op on all prng algorithms"),
      tsl::Flag("rng_bit_generator_philox_expander",
                &config.rng_bit_generator_philox_expander,
                "Expands rng_bit_generator op using philox prng algorithm"),
      tsl::Flag("rng_bit_generator_three_fry_expander",
                &config.rng_bit_generator_three_fry_expander,
                "Expands rng_bit_generator op using three_fry prng algorithm"),
      tsl::Flag("triangular_solve_expander", &config.triangular_solve_expander,
                "Expands triangular_solve op"),
      tsl::Flag("verify_hlo", &config.verify_hlo,
                "Run HLO verifier after passes"),
  };
}

void ParseCompoundFlags(HloExpandConfig& config) {
  config.batch_norm_grad_expander |=
      config.expand_all || config.batch_norm_expander;
  config.batch_norm_inference_expander |=
      config.expand_all || config.batch_norm_expander;
  config.batch_norm_training_expander |=
      config.expand_all || config.batch_norm_expander;
  config.cholesky_expander |= config.expand_all;
  config.rng_bit_generator_philox_expander |=
      config.expand_all || config.rng_bit_generator_expander;
  config.rng_bit_generator_three_fry_expander |=
      config.expand_all || config.rng_bit_generator_expander;
  config.rng_expander |= config.expand_all;
  config.triangular_solve_expander |= config.expand_all;
}

}  // namespace xla
