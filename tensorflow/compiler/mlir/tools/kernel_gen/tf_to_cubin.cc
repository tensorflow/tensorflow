// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- tf_to_cubin.cc -------------------------------------------*- C++ -*-===//
//
// This file implements the entry point to compile a tf op to a cubin file.
//
//===----------------------------------------------------------------------===//
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/cubin_creator.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {
bool ParseStringList(std::string string_list, std::vector<uint32_t>* result) {
  result->clear();
  uint32_t item;
  auto items = absl::StrSplit(string_list, ',');
  for (const auto& item_str : items) {
    if (!absl::SimpleAtoi(item_str, &item)) {
      LOG(ERROR) << "Expected token " << item_str << " to be an integer";
      return false;
    }
    result->push_back(item);
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  std::string output_file = "foo.bin";
  int32_t architecture = 50;
  std::vector<uint32_t> tile_sizes;
  std::vector<uint32_t> unroll_factors;
  std::vector<uint32_t> same_shape;

  auto parse_tile_sizes = [&tile_sizes](std::string tile_sizes_str) {
    if (!ParseStringList(tile_sizes_str, &tile_sizes)) {
      return false;
    }
    // Initialize with the default.
    if (tile_sizes.empty()) {
      tile_sizes.push_back(16);
      tile_sizes.push_back(64);
    }
    return true;
  };

  auto parse_unroll_factors =
      [&unroll_factors](std::string unroll_factors_str) {
        return ParseStringList(unroll_factors_str, &unroll_factors);
      };

  auto parse_same_shape = [&same_shape](std::string same_shape_str) {
    return ParseStringList(same_shape_str, &same_shape);
  };

  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("output", &output_file, "output file"),
      tensorflow::Flag("arch", &architecture,
                       "target architecture (e.g. 50 for sm_50)"),
      tensorflow::Flag("tile_sizes", parse_tile_sizes, "16,64",
                       "tile sizes to use"),
      tensorflow::Flag("unroll_factors", parse_unroll_factors, "",
                       "factors to unroll by, separated by commas"),
      tensorflow::Flag("same_shape", parse_same_shape, "",
                       "arguments with same shape, separated by commas"),
  };
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain("usage", &argc, &argv);
  if (!parse_ok) {
    return 1;
  }

  std::pair<int32_t, int32_t> compute_capability(architecture / 10,
                                                 architecture % 10);

  auto cubin = tensorflow::kernel_gen::GenerateCubinForTfCode(
      argv[1], compute_capability, tile_sizes, same_shape, unroll_factors);

  if (!cubin.ok()) {
    LOG(ERROR) << cubin.status();
    return 1;
  }

  std::vector<uint8_t> cubin_data = cubin.ConsumeValueOrDie();

  auto status = tensorflow::WriteStringToFile(
      tensorflow::Env::Default(), output_file,
      absl::string_view{reinterpret_cast<char*>(cubin_data.data()),
                        cubin_data.size()});

  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }

  return 0;
}
