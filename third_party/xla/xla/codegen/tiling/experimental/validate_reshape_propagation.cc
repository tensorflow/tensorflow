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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace xla::gpu::experimental {
namespace {

std::vector<int64_t> ParseShape(absl::string_view shape_str) {
  std::vector<int64_t> shape;
  if (shape_str.empty() || shape_str.front() != '[' ||
      shape_str.back() != ']') {
    LOG(FATAL) << "Invalid shape string: " << shape_str;
  }
  absl::string_view inner = shape_str.substr(1, shape_str.size() - 2);
  CHECK(!inner.empty());
  std::vector<absl::string_view> dims = absl::StrSplit(inner, ',');
  for (auto dim_str : dims) {
    int64_t dim;
    CHECK(absl::SimpleAtoi(dim_str, &dim)) << "Failed to parse " << dim_str;
    shape.push_back(dim);
  }
  return shape;
}

std::vector<int64_t> GetPowersOf2And(int64_t limit,
                                     std::vector<int64_t> extra) {
  std::vector<int64_t> res;
  for (int64_t p = 1; p <= limit; p *= 2) {
    res.push_back(p);
  }
  for (int64_t v : extra) {
    if (v > 0 && absl::c_find(res, v) == res.end()) {
      res.push_back(v);
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

std::vector<int64_t> GetOffsets(int64_t limit) {
  std::vector<int64_t> res = {0, 1, 2, limit - 1};
  std::vector<int64_t> filtered;
  for (int64_t v : res) {
    if (v >= 0 && v < limit && absl::c_find(filtered, v) == filtered.end()) {
      filtered.push_back(v);
    }
  }
  std::sort(filtered.begin(), filtered.end());
  return filtered;
}

std::vector<int64_t> GetUpperBounds(int64_t limit) {
  std::vector<int64_t> res = {1, limit - 1, limit};
  std::vector<int64_t> filtered;
  for (int64_t v : res) {
    if (v > 0 && v <= limit && absl::c_find(filtered, v) == filtered.end()) {
      filtered.push_back(v);
    }
  }
  std::sort(filtered.begin(), filtered.end());
  return filtered;
}

struct ValidationStats {
  int64_t total_tested = 0;
  int64_t true_positives = 0;
  int64_t true_negatives = 0;
  int64_t false_positives = 0;
  int64_t false_negatives = 0;
};

struct TileConfig {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::vector<int64_t> offsets;
  std::vector<int64_t> upper_bounds;
};

absl::Status ValidateConfig(const Shape& from_shape, const Shape& to_shape,
                            const HloInstruction* reshape,
                            mlir::MLIRContext* mlir_context,
                            const TileConfig& config, ValidationStats& stats) {
  // To understand if concrete tile propagation is correct or produces
  // false positive or negative we propagate the tile symbolically and
  // see if this tiling passes VerifyTileEquivalence.
  // Then we propagate tile with sizes set. If both results matches (both are OK
  // or both reject the tiling) then we declare a success.
  stats.total_tested++;
  int64_t rank = from_shape.dimensions().size();

  ASSIGN_OR_RETURN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(reshape),
                          mlir_context));

  llvm::SmallVector<DimTile> output_tiles;
  output_tiles.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    SymbolicExpr tid = CreateDimExpr(i, mlir_context);
    SymbolicExpr tile_size = CreateSymbolExpr(i, rank, mlir_context);
    output_tiles.push_back(DimTile{
        /* offset= */
        CreateSymbolicConstant(config.offsets[i], mlir_context) +
            tid * tile_size,
        /* size= */ tile_size,
        /* stride= */
        CreateSymbolicConstant(config.strides[i], mlir_context),
        /* upper_bound= */
        CreateSymbolicConstant(config.upper_bounds[i], mlir_context)});
  }
  Tile output_tile(*tiling_space, std::move(output_tiles));
  ASSIGN_OR_RETURN(
      Tiles input_tiles,
      PropagateTileToInput(*tiling_space, *reshape, output_tile, 0));
  CHECK_EQ(input_tiles.size(), 1);
  Tile input_tile = input_tiles[0];
  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replacement_map =
      GetTileSizeReplacementMap(*tiling_space, config.sizes);
  RETURN_IF_ERROR(tiling_space->AssignTileSizes(config.sizes));

  input_tile.Replace(replacement_map);
  input_tile.Simplify();
  output_tile.Replace(replacement_map);
  output_tile.Simplify();

  absl::Status equivalent = VerifyTileEquivalence(
      output_tile, from_shape, input_tile, to_shape, tiling_space.get());
  absl::StatusOr<Tiles> concrete_propagation_result =
      PropagateTileToInput(*tiling_space, *reshape, output_tile, 0);
  if (equivalent.ok() == concrete_propagation_result.ok()) {
    if (equivalent.ok()) {
      stats.true_positives++;
    } else {
      stats.true_negatives++;
    }
    return absl::OkStatus();
  }

  if (!concrete_propagation_result.ok()) {
    // False negatives are not that interesting.
    LOG(INFO) << absl::StrCat(
        "\nFalse negative:\n", "Config: tile sizes=[",
        absl::StrJoin(config.sizes, ", "), "] strides=[",
        absl::StrJoin(config.strides, ", "), "] static offsets=[",
        absl::StrJoin(config.offsets, ", "), "] upper_bounds=[",
        absl::StrJoin(config.upper_bounds, ", "), "]\n", "From Tile:\n",
        output_tile.ToString(), "\n", "To Tile:\n", input_tile.ToString(), "\n",
        concrete_propagation_result.status().message(), "\n");
    stats.false_negatives++;
  }
  if (!equivalent.ok()) {
    std::cout << absl::StrCat(
        "\nFalse positive:\n", "Config: sizes=[",
        absl::StrJoin(config.sizes, ", "), "] strides=[",
        absl::StrJoin(config.strides, ", "), "] offsets=[",
        absl::StrJoin(config.offsets, ", "), "] upper_bounds=[",
        absl::StrJoin(config.upper_bounds, ", "), "]\n", "From Tile:\n",
        output_tile.ToString(), "\n", "To Tile: \n", input_tile.ToString(),
        "\n", equivalent.message(), "\n");
    stats.false_positives++;
  }
  return absl::OkStatus();
}

void ProcessReshape(const Shape& from_shape, const Shape& to_shape,
                    bool bypass_limit) {
  std::cout << "\n==========================================================="
               "=======\n";
  std::cout << "Validating reshape: " << from_shape.ToString() << " -> "
            << to_shape.ToString() << "\n";
  std::cout << "============================================================="
               "=====\n";

  mlir::MLIRContext mlir_context;
  HloComputation::Builder builder("entry");
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, to_shape, "p0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(from_shape, p0));

  int64_t rank = from_shape.dimensions().size();

  // Generate Cartesian products
  std::vector<std::vector<int64_t>> size_configs(rank);
  std::vector<std::vector<int64_t>> stride_configs(rank);
  std::vector<std::vector<int64_t>> offset_configs(rank);
  std::vector<std::vector<int64_t>> upper_bound_configs(rank);

  int64_t num_configs = 1;
  for (int i = 0; i < rank; ++i) {
    int64_t dim = from_shape.dimensions(i);
    size_configs[i] = GetPowersOf2And(dim, {dim});
    stride_configs[i] = GetPowersOf2And(dim, {dim});
    offset_configs[i] = GetOffsets(dim);
    upper_bound_configs[i] = GetUpperBounds(dim);

    num_configs *= size_configs[i].size() * stride_configs[i].size() *
                   offset_configs[i].size() * upper_bound_configs[i].size();
  }

  std::cout << "Estimated number of configs for this reshape: " << num_configs
            << "\n";
  if (num_configs > 100000 && !bypass_limit) {
    std::cout << "Skipping reshape because it has too many configs ("
              << num_configs << "). Use --bypass_limit to run it.\n";
    return;
  }

  ValidationStats stats;
  TileConfig config;
  config.sizes.resize(rank);
  config.strides.resize(rank);
  config.offsets.resize(rank);
  config.upper_bounds.resize(rank);

  auto run_loops = [&](auto self, int dim_idx) -> void {
    if (dim_idx == rank) {
      absl::Status status = ValidateConfig(from_shape, to_shape, reshape,
                                           &mlir_context, config, stats);
      if (!status.ok()) {
        LOG(ERROR) << status.message() << "\n";
      }
      return;
    }
    for (int64_t size : size_configs[dim_idx]) {
      config.sizes[dim_idx] = size;
      for (int64_t stride : stride_configs[dim_idx]) {
        config.strides[dim_idx] = stride;
        for (int64_t offset : offset_configs[dim_idx]) {
          config.offsets[dim_idx] = offset;
          for (int64_t ub : upper_bound_configs[dim_idx]) {
            if (offset + (size - 1) * stride >= ub) {
              continue;
            }
            config.upper_bounds[dim_idx] = ub;
            self(self, dim_idx + 1);
          }
        }
      }
    }
  };

  run_loops(run_loops, 0);

  std::cout << absl::StrCat(
      "Total configs tested: ", stats.total_tested, "\n",
      "Supported (True Positives): ", stats.true_positives, " (",
      (stats.total_tested ? (stats.true_positives * 100 / stats.total_tested)
                          : 0),
      "%)\n", "Unsupported (True Negatives): ", stats.true_negatives, " (",
      (stats.total_tested ? (stats.true_negatives * 100 / stats.total_tested)
                          : 0),
      "%)\n", "False Positives: ", stats.false_positives, " (",
      (stats.total_tested ? (stats.false_positives * 100 / stats.total_tested)
                          : 0),
      "%)\n", "False Negatives: ", stats.false_negatives, " (",
      (stats.total_tested ? (stats.false_negatives * 100 / stats.total_tested)
                          : 0),
      "%)\n");
}

void RunValidation(const std::vector<std::pair<Shape, Shape>>& reshapes,
                   bool bypass_limit) {
  for (const auto& pair : reshapes) {
    ProcessReshape(pair.first, pair.second, bypass_limit);
  }
}

}  // namespace
}  // namespace xla::gpu::experimental

int main(int argc, char** argv) {
  std::string reshapes_str = "";
  bool bypass_limit = false;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("reshapes", &reshapes_str,
                "List of reshapes, e.g. '[1, 4] [4]; [14] [2, 7]'"),
      tsl::Flag("bypass_limit", &bypass_limit,
                "Bypass the 100,000 combinations limit")};

  const std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (!parse_ok) {
    std::cerr << usage << "\n";
    return 1;
  }

  reshapes_str.erase(
      std::remove_if(reshapes_str.begin(), reshapes_str.end(),
                     [](unsigned char c) { return std::isspace(c); }),
      reshapes_str.end());

  if (reshapes_str.empty()) {
    LOG(INFO) << "No reshapes provided.";
    return 0;
  }

  std::vector<std::string> parts = absl::StrSplit(
      reshapes_str, absl::ByAnyChar(";"), absl::SkipWhitespace());

  std::vector<std::pair<xla::Shape, xla::Shape>> reshapes;
  for (const std::string& part : parts) {
    size_t split_pos = part.find("][");
    if (split_pos == std::string::npos) {
      LOG(FATAL) << "Expected two shapes for each reshape, e.g. [1,4][4], got: "
                 << part;
    }
    std::string str1 = part.substr(0, split_pos + 1);
    std::string str2 = part.substr(split_pos + 1);

    std::vector<int64_t> in_dims = xla::gpu::experimental::ParseShape(str1);
    std::vector<int64_t> out_dims = xla::gpu::experimental::ParseShape(str2);
    reshapes.push_back({xla::ShapeUtil::MakeShape(xla::F32, in_dims),
                        xla::ShapeUtil::MakeShape(xla::F32, out_dims)});
  }

  xla::gpu::experimental::RunValidation(reshapes, bypass_limit);
  return 0;
}
