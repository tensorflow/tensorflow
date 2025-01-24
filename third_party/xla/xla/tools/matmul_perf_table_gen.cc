/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tools/matmul_perf_table_gen.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Defines how many times do we profile a matrix multiplication of interest.
constexpr size_t kNumProfilingRuns = 5;

struct StaticSpec {
  int m;
  int n;
  int k;
  std::string dtype_lhs;
  std::string dtype_rhs;
  std::string dtype_out;
};

struct ExplicitSpec {
  std::unique_ptr<HloModule> module;
};

using EntrySpec = std::variant<StaticSpec, ExplicitSpec>;

void ReportProgress(int i, int size) {
  if (i % (size / std::min(size, 10)) == 0) {
    LOG(INFO) << "Progress: " << 100 * i / size << "%.";
  }
}

std::unique_ptr<HloModule> GetModule(absl::string_view lhs_dtype,
                                     absl::string_view rhs_dtype,
                                     absl::string_view out_dtype, int m, int n,
                                     int k) {
  std::string text = absl::Substitute(R"(
    HloModule m

    ENTRY e {
      lhs = $0[$3,$5] parameter(0)
      rhs = $1[$5,$4] parameter(1)
      ROOT _ = $2[$3,$4] dot(lhs,rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )",
                                      lhs_dtype, rhs_dtype, out_dtype, m, n, k);

  auto parsed = ParseAndReturnUnverifiedModule(text);
  CHECK_OK(parsed.status());
  return *std::move(parsed);
}

void Measure(HloRunner& runner, Executable* executable,
             const std::vector<Literal>& args_small,
             const std::vector<Literal>& args_large) {
  CHECK_OK(runner.ExecuteWithExecutable(executable, args_small).status());
  CHECK_OK(runner.ExecuteWithExecutable(executable, args_large).status());
}

void AddDotsFromHlos(const std::string& hlo_scan_path,
                     std::vector<EntrySpec>& specs) {
  if (hlo_scan_path.empty()) {
    return;
  }

  std::vector<std::string> filenames;
  CHECK_OK(tsl::Env::Default()->GetChildren(hlo_scan_path, &filenames));
  for (const std::string& filename : filenames) {
    // Read file.
    std::string hlo_data;
    std::string hlo_path = absl::StrJoin({hlo_scan_path, filename}, "/");
    CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo_data));

    // Parse and verify HloModule. Warn about bogus ones.
    auto module = ParseAndReturnUnverifiedModule(hlo_data);
    if (!module.ok()) {
      LOG(ERROR) << "Cannot parse: " << hlo_path;
      continue;
    }

    hlo_query::ForEachInstructionWithOpcode(
        **module, HloOpcode::kDot, [&specs](HloInstruction* instr) {
          // Create module.
          HloModuleConfig config;
          config.set_debug_options(GetDebugOptionsFromFlags());
          auto module = std::make_unique<HloModule>("module", config);

          // Create entry computation with dot.
          HloComputation::Builder entry_builder("entry");
          HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
          HloInstruction* p0 =
              entry_builder.AddInstruction(HloInstruction::CreateParameter(
                  0, dot->operand(0)->shape(), "p0"));
          HloInstruction* p1 =
              entry_builder.AddInstruction(HloInstruction::CreateParameter(
                  1, dot->operand(1)->shape(), "p1"));
          entry_builder.AddInstruction(HloInstruction::CreateDot(
              dot->shape(), p0, p1, dot->dot_dimension_numbers(),
              dot->precision_config()));
          module->AddEntryComputation(entry_builder.Build());

          // Add spec to the profiling set.
          ExplicitSpec spec;
          spec.module = std::move(module);
          specs.push_back(std::move(spec));
        });
  }
}

}  // namespace

std::unique_ptr<Executable> MatmulPerfTableGen::Compile(
    std::unique_ptr<HloModule> module) {
  auto compiled =
      runner_.CreateExecutable(std::move(module), /*run_hlo_passes=*/true);
  CHECK_OK(compiled.status());
  return std::move(*compiled);
}

absl::Duration MatmulPerfTableGen::Profile(std::unique_ptr<HloModule> module) {
  VLOG(1) << "Profiling module: " << module->ToString();

  // Flip flop between arguments to prevent caching.
  std::minstd_rand0 engine;

  std::vector<Literal> args_small = MakeFakeArguments(module.get(), &engine,
                                                      /*use_large_range=*/false)
                                        .value();
  std::vector<Literal> args_large = MakeFakeArguments(module.get(), &engine,
                                                      /*use_large_range=*/true)
                                        .value();

  std::unique_ptr<Executable> compiled = Compile(std::move(module));

  // First run to warm up stuff.
  CHECK_OK(runner_.ExecuteWithExecutable(compiled.get(), args_small).status());

  // Run matrix multiplications but do not trace.
  if (config_.dry_run) {
    for (int i = 0; i < kNumProfilingRuns; i++) {
      Measure(runner_, compiled.get(), args_small, args_large);
    }
    return absl::ZeroDuration();
  }

  // Trace `kNumProfilingRuns` times to get decent measurement.
  std::unique_ptr<HloOpProfiler::KernelTracer> tracer =
      HloOpProfiler::GetKernelTracer();
  for (int i = 0; i < kNumProfilingRuns; i++) {
    Measure(runner_, compiled.get(), args_small, args_large);
  }

  return absl::Nanoseconds(std::move(*tracer).getMedianKernelTimeNs());
}

gpu::DeviceHloInstructionProfiles MatmulPerfTableGen::ComputeTable() {
  gpu::DeviceHloInstructionProfiles device_profiles;
  gpu::HloInstructionProfileList profile_list;

  std::vector<EntrySpec> specs;

  // Sweep over statically defined search space.
  auto inc = [](uint32_t i, const MatmulPerfTableGen::StepSpec& spec) {
    if (spec.step > 0) {
      return i + spec.step;
    }
    if (spec.factor > 0) {
      return i * spec.factor;
    }
    return i;
  };

  MatmulPerfTableGen::StepSpec m_spec = config_.m_spec;
  MatmulPerfTableGen::StepSpec n_spec = config_.n_spec;
  MatmulPerfTableGen::StepSpec k_spec = config_.k_spec;
  for (MatmulPerfTableGen::DataTypeSpec& dtype : config_.dtypes) {
    for (int m = m_spec.start; m <= m_spec.stop; m = inc(m, m_spec)) {
      for (int n = n_spec.start; n <= n_spec.stop; n = inc(n, n_spec)) {
        for (int k = k_spec.start; k <= k_spec.stop; k = inc(k, k_spec)) {
          StaticSpec spec;
          spec.m = m;
          spec.k = k;
          spec.n = n;
          spec.dtype_lhs = dtype.lhs_dtype;
          spec.dtype_rhs = dtype.rhs_dtype;
          spec.dtype_out = dtype.out_dtype;
          specs.push_back(spec);
        }
      }
    }
  }

  // Sweep over provided HLOs.
  AddDotsFromHlos(config_.hlo_scan_path, specs);

  std::minstd_rand0 engine;
  std::shuffle(specs.begin(), specs.end(), engine);

  auto& device_info =
      runner_.backend().stream_executors()[0]->GetDeviceDescription();

  for (int i = 0; i < specs.size(); i++) {
    EntrySpec& spec = specs[i];
    std::unique_ptr<HloModule> module;
    if (std::holds_alternative<StaticSpec>(spec)) {
      StaticSpec& static_spec = std::get<StaticSpec>(spec);
      module = GetModule(static_spec.dtype_lhs, static_spec.dtype_rhs,
                         static_spec.dtype_out, static_spec.m, static_spec.n,
                         static_spec.k);
    }

    if (std::holds_alternative<ExplicitSpec>(spec)) {
      module = std::move(std::get<ExplicitSpec>(spec).module);
    }

    CHECK_NOTNULL(module);

    HloInstructionProto instr =
        module->entry_computation()->root_instruction()->ToProto();
    absl::Duration time = Profile(std::move(module));

    gpu::HloInstructionProfile entry;
    *entry.mutable_instruction() = instr;
    entry.set_clock_cycles(device_info.clock_rate_ghz() *
                           absl::ToInt64Nanoseconds(time));

    *profile_list.add_entries() = entry;

    ReportProgress(i + 1, specs.size());
  }
  std::string device_key = gpu::HloOpProfiles::GetProfileName(device_info);
  device_profiles.mutable_entries()->insert({device_key, profile_list});
  return device_profiles;
}

absl::Status MatmulPerfTableGen::Dump(
    const DeviceHloInstructionProfiles& table) {
  if (config_.output == "stdout") {
    LOG(INFO) << table.DebugString();
    return absl::OkStatus();
  }

  DeviceHloInstructionProfiles file;
  if (tsl::Env::Default()->FileExists(config_.output).ok()) {
    TF_RETURN_IF_ERROR(
        tsl::ReadTextOrBinaryProto(tsl::Env::Default(), config_.output, &file));
  }

  CHECK_EQ(table.entries_size(), 1)
      << "Expecting one program run, for one device config";
  std::string sm_ver = table.entries().begin()->first;
  if (file.entries().contains(sm_ver)) {
    file.mutable_entries()->at(sm_ver).MergeFrom(table.entries().at(sm_ver));
  } else {
    file.MergeFrom(table);
  }

  if (absl::StrContains(config_.output, ".pbtxt")) {
    return tsl::WriteTextProto(tsl::Env::Default(), config_.output, file);
  }
  if (absl::StrContains(config_.output, ".pb")) {
    return tsl::WriteBinaryProto(tsl::Env::Default(), config_.output, file);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported file: ", config_.output,
                   ". Expecting .pb or .pbtxt suffix."));
}

}  // namespace xla::gpu
