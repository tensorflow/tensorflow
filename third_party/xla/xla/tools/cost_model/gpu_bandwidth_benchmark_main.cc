/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tools/cost_model/gpu_bandwidth_benchmark.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

int main(int argc, char* argv[]) {
  int device_ordinal = 0;
  int min_exp = 13;
  int max_exp = 33;
  int warmup_runs = 10;
  int measurement_runs = 100;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("device_ordinal", &device_ordinal,
                "0-indexed GPU device ordinal."),
      tsl::Flag("min_exp", &min_exp,
                "Minimum power-of-two transfer size exponent (13 = 8 KiB)."),
      tsl::Flag("max_exp", &max_exp,
                "Maximum power-of-two transfer size exponent (33 = 8 GiB)."),
      tsl::Flag("warmup_runs", &warmup_runs,
                "Number of warmup iterations per transfer size."),
      tsl::Flag("measurement_runs", &measurement_runs,
                "Number of timed iterations per transfer size."),
  };
  const std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  const bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);
  if (!parse_ok || argc > 1) {
    absl::FPrintF(stderr, "%s", usage);
    return EXIT_FAILURE;
  }

  absl::StatusOr<std::vector<xla::gpu::BandwidthEntry>> entries =
      xla::gpu::MeasureBandwidthTable(device_ordinal, min_exp, max_exp,
                                      warmup_runs, measurement_runs);
  if (!entries.ok()) {
    LOG(ERROR) << "Bandwidth table measurement failed: " << entries.status();
    return EXIT_FAILURE;
  }

  absl::PrintF("%s", xla::gpu::FormatBandwidthTable(*entries));
  return EXIT_SUCCESS;
}
