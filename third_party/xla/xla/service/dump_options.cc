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

#include "xla/service/dump_options.h"

#include <functional>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/regexp.h"

namespace xla {

DumpOptions::DumpOptions(const DebugOptions& opts)
    : dump_to(opts.xla_dump_to()),
      dump_as_text(opts.xla_dump_hlo_as_text()),
      dump_as_proto(opts.xla_dump_hlo_as_proto()),
      dump_as_dot(opts.xla_dump_hlo_as_dot()),
      dump_as_html(opts.xla_dump_hlo_as_html()),
      dump_as_url(opts.xla_dump_hlo_as_url()),
      dump_fusion_visualization(opts.xla_dump_fusion_visualization()),
      dump_snapshots(opts.xla_dump_hlo_snapshots()),
      dump_unoptimized_snapshots(opts.xla_dump_hlo_unoptimized_snapshots()),
      dump_include_timestamp(opts.xla_dump_include_timestamp()),
      dump_max_hlo_modules(opts.xla_dump_max_hlo_modules()),
      dump_compress_protos(opts.xla_dump_compress_protos()),
      dump_fdo_profiles(opts.xla_gpu_experimental_dump_fdo_profiles()),
      dump_mlir_pretty_form(opts.xla_dump_enable_mlir_pretty_form()) {
  // This constructor examines the values in `opts` and turns on other flags
  // based on what we think is the user's intent.  To reduce confusion about
  // what was a user-specified value versus an extrapolated value, within this
  // function we treat this struct's members as write-only, and read only from
  // `opts`.

  // Did the user specify an explicit format for dumping?
  bool output_format_other_than_url_specified =
      opts.xla_dump_hlo_as_text() || opts.xla_dump_hlo_as_proto() ||
      opts.xla_dump_hlo_as_dot() || opts.xla_dump_hlo_as_html() ||
      opts.xla_dump_hlo_snapshots() ||
      opts.xla_dump_hlo_unoptimized_snapshots();
  bool output_format_specified =
      output_format_other_than_url_specified || opts.xla_dump_hlo_as_url();

  // If we haven't specified an output format, default to dumping as text.
  if (!output_format_specified) {
    dump_as_text = true;
  }

  // Disable dumping if specified by the user.
  if (!opts.xla_enable_dumping()) {
    dump_to = "";
  }

  // If dump_to is empty, default to dumping to stdout, so long as some dump
  // format other than dump-as-url was specified.  If the user only specified
  // --xla_dump_hlo_as_url, then don't dump to stdout, that is likely noise
  // they don't want.
  if (opts.xla_dump_to().empty() && output_format_other_than_url_specified) {
    dump_to = "-";
  }

  // If we specified a regular expression restricting which modules to dump,
  // respect that.
  //
  // If we didn't specify which modules to dump but we passed some other flag
  // which implies dumping modules, dump all modules.
  //
  // Otherwise, don't dump any HLO modules.
  if (!opts.xla_dump_hlo_module_re().empty()) {
    // RE2 object is not copyable, and we can't capture "by move", so we
    // resort to this hack.
    std::string pattern = opts.xla_dump_hlo_module_re();
    should_dump_module = [pattern](absl::string_view module_name) {
      return RE2::PartialMatch(module_name, pattern);
    };
  } else if (!opts.xla_dump_hlo_pass_re().empty() ||
             !opts.xla_dump_emitter_re().empty() ||
             !opts.xla_dump_to().empty() || output_format_specified) {
    should_dump_module = [](absl::string_view) { return true; };
  } else {
    should_dump_module = [](absl::string_view) { return false; };
  }

  // Initialize should_dump_pass.  This one is easy: We only dump per-pass
  // data if the user asked for it explicitly.
  if (!opts.xla_dump_hlo_pass_re().empty()) {
    std::string pattern = opts.xla_dump_hlo_pass_re();
    should_dump_pass = [pattern](absl::string_view pass_name) {
      return RE2::PartialMatch(pass_name, pattern);
    };
  } else {
    should_dump_pass = [](absl::string_view) { return false; };
  }

  if (!opts.xla_dump_emitter_re().empty()) {
    std::string pattern = opts.xla_dump_emitter_re();
    should_dump_emitter = [pattern](absl::string_view emitter_name) {
      return RE2::PartialMatch(emitter_name, pattern);
    };
  } else {
    should_dump_emitter = [](absl::string_view) { return false; };
  }

  // Initialize should_dump_pipeline. If the option was not specified, dump
  // all pipelines. Otherwise dump only those pipelines that user asked for
  // explicitly.
  if (!opts.xla_dump_hlo_pipeline_re().empty()) {
    std::string pattern = opts.xla_dump_hlo_pipeline_re();
    should_dump_pipeline = [pattern](absl::string_view pipeline_name) {
      return RE2::PartialMatch(pipeline_name, pattern);
    };
  } else {
    should_dump_pipeline = [](absl::string_view) { return true; };
  }

  // Output dirs "sponge" and "test_undeclared_outputs_dir" (case-insensitive)
  // have a special meaning: Dump into the directory specified by the
  // environment variable TEST_UNDECLARED_OUTPUTS_DIR.
  std::string dump_to_lower = absl::AsciiStrToLower(dump_to);
  if (dump_to_lower == "sponge" ||
      dump_to_lower == "test_undeclared_outputs_dir") {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&dump_to)) {
      LOG(ERROR) << "--xla_dump_to=" << opts.xla_dump_to()
                 << ", but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                    "is not set, so cannot dump anywhere.";
      should_dump_module = [](absl::string_view) { return false; };
      should_dump_pass = [](absl::string_view) { return false; };
      should_dump_emitter = [](absl::string_view) { return false; };
      should_dump_pipeline = [](absl::string_view) { return false; };
    }
  }

  // Dumping unoptimized HLO snapshots should not trigger dumping of all
  // available information for the HLO module and pipelines.

  if (dump_unoptimized_snapshots) {
    should_dump_module = [](absl::string_view) { return false; };
    should_dump_pipeline = [](absl::string_view) { return false; };
  }
}

}  // namespace xla
