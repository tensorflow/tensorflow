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

// A small utility to compress data from standard input to standard output
// using Zstd compression at a specified level (default 22).
//
// This is useful specifically in genrules where you cannot rely on the
// standard zstd CLI tool being available in the build environment.
//
// Arguments:
//   [compression_level] (optional): The Zstd compression level to use
//                                   (default 22).
//
// Usage in genrule:
// genrule(
//     name = "src_zst",
//     srcs = [":src"],
//     outs = ["src.zst"],
//     cmd = "$(location :zstd_compressor)\
//         --compression_level=22 < $(location :src) > $@",
//     tools = [":zstd_compressor"],
// )

#include <iostream>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/copy_all.h"
#include "riegeli/bytes/std_io.h"
#include "riegeli/zstd/zstd_writer.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

int main(int argc, char** argv) {
  int compression_level = 22;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("compression_level", &compression_level,
                "Zstd compression level"),
  };
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);

  if (!parse_ok) {
    std::cerr << tsl::Flags::Usage(argv[0], flag_list);
    return 1;
  }

  riegeli::ZstdWriter zstd_writer(
      riegeli::Maker<riegeli::StdOut>(),
      riegeli::ZstdWriterBase::Options().set_compression_level(
          compression_level));

  absl::Status status =
      riegeli::CopyAll(riegeli::StdIn(), std::move(zstd_writer));
  if (!status.ok()) {
    std::cerr << "Compression failed: " << status.message() << std::endl;
    return 1;
  }

  return 0;
}
