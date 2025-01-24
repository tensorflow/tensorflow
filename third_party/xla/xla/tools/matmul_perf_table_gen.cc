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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_runner.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Defines how many times do we profile a matrix multiplication of interest.
constexpr size_t kNumProfilingRuns = 5;

struct EntrySpec {
  int m;
  int n;
  int k;
  std::string dtype_lhs;
  std::string dtype_rhs;
  std::string dtype_out;
};

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
  MatmulPerfTableGen::StepSpec m_spec = config_.m_spec;
  MatmulPerfTableGen::StepSpec n_spec = config_.n_spec;
  MatmulPerfTableGen::StepSpec k_spec = config_.k_spec;

  std::vector<EntrySpec> specs;
  for (MatmulPerfTableGen::DataTypeSpec& dtype : config_.dtypes) {
    for (int m = m_spec.start; m <= m_spec.stop; m += m_spec.step) {
      for (int n = n_spec.start; n <= n_spec.stop; n += n_spec.step) {
        for (int k = k_spec.start; k <= k_spec.stop; k += k_spec.step) {
          EntrySpec spec;
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

  std::minstd_rand0 engine;
  std::shuffle(specs.begin(), specs.end(), engine);

  auto& device_info =
      runner_.backend().stream_executors()[0]->GetDeviceDescription();

  for (int i = 0; i < specs.size(); i++) {
    EntrySpec& spec = specs[i];
    std::unique_ptr<HloModule> module = GetModule(
        spec.dtype_lhs, spec.dtype_rhs, spec.dtype_out, spec.m, spec.n, spec.k);

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
