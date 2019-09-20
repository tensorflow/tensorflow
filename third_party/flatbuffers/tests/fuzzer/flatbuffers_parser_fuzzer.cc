// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <stddef.h>
#include <stdint.h>
#include <clocale>
#include <string>

#include "flatbuffers/idl.h"
#include "test_init.h"

static constexpr uint8_t flags_strict_json = 0x01;
static constexpr uint8_t flags_skip_unexpected_fields_in_json = 0x02;
static constexpr uint8_t flags_allow_non_utf8 = 0x04;
// static constexpr uint8_t flags_flag_3 = 0x08;
// static constexpr uint8_t flags_flag_4 = 0x10;
// static constexpr uint8_t flags_flag_5 = 0x20;
// static constexpr uint8_t flags_flag_6 = 0x40;
// static constexpr uint8_t flags_flag_7 = 0x80;

// Utility for test run.
OneTimeTestInit OneTimeTestInit::one_time_init_;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Reserve one byte for Parser flags and one byte for repetition counter.
  if (size < 3) return 0;
  const uint8_t flags = data[0];
  // normalize to ascii alphabet
  const int extra_rep_number = data[1] >= '0' ? (data[1] - '0') : 0;
  data += 2;
  size -= 2;  // bypass

  const std::string original(reinterpret_cast<const char *>(data), size);
  auto input = std::string(original.c_str());  // until '\0'
  if (input.empty()) return 0;

  flatbuffers::IDLOptions opts;
  opts.strict_json = (flags & flags_strict_json);
  opts.skip_unexpected_fields_in_json =
      (flags & flags_skip_unexpected_fields_in_json);
  opts.allow_non_utf8 = (flags & flags_allow_non_utf8);

  flatbuffers::Parser parser(opts);

  // Guarantee 0-termination in the input.
  auto parse_input = input.c_str();

  // The fuzzer can adjust the number repetition if a side-effects have found.
  // Each test should pass at least two times to ensure that the parser doesn't
  // have any hidden-states or locale-depended effects.
  for (auto cnt = 0; cnt < (extra_rep_number + 2); cnt++) {
    // Each even run (0,2,4..) will test locale independed code.
    auto use_locale = !!OneTimeTestInit::test_locale() && (0 == (cnt % 2));
    // Set new locale.
    if (use_locale) {
      FLATBUFFERS_ASSERT(setlocale(LC_ALL, OneTimeTestInit::test_locale()));
    }

    // Check Parser.
    parser.Parse(parse_input);

    // Restore locale.
    if (use_locale) { FLATBUFFERS_ASSERT(setlocale(LC_ALL, "C")); }
  }

  return 0;
}
