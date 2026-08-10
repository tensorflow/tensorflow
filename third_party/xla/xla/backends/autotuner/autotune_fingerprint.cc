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

#include "xla/backends/autotuner/autotune_fingerprint.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/xla.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

tsl::Fprint128 GetHloFingerprint(const HloInstruction& instr) {
  auto options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);

  return tsl::Fingerprint128(instr.ToString(options));
}

// TODO(b/444398084): Consider only codegen-relevant fields in DebugOptions.
// TODO(b/444398084): Avoid recomputing the fingerprint for the same
// DebugOptions.
std::string GetCodegenOptionsFingerprint(const DebugOptions& options) {
  std::string proto_str;
  google::protobuf::TextFormat::PrintToString(options, &proto_str);
  uint64_t fprint = tsl::Fingerprint64(proto_str);
  return absl::StrCat(absl::Hex(fprint, absl::kZeroPad16));
}

}  // namespace xla
