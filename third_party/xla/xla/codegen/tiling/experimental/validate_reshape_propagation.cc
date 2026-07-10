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
  int64_t false_positives = 0;
  int64_t false_negatives = 0;
  int64_t total_unsupported = 0;
};

struct TileConfig {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::vector<int64_t> offsets;
  std::vector<int64_t> upper_bounds;
};

void ValidateConfig(const Shape& input_shape, const Shape& output_shape,
                    const HloInstruction* reshape, const HloInstruction* p0,
                    TilingSpace* tiling_space_sym,
                    mlir::MLIRContext* mlir_context, const TileConfig& config,
                    ValidationStats& stats) {
  stats.total_tested++;
  int64_t rank = tiling_space_sym->num_dimensions();

  // Propagate symbolically - that does not trigger checks.
  llvm::SmallVector<DimTile> input_dim_tiles_sym;
  input_dim_tiles_sym.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    SymbolicExpr tid = CreateDimExpr(i, mlir_context);
    SymbolicExpr ts = CreateSymbolExpr(i, rank, mlir_context);
    input_dim_tiles_sym.push_back(DimTile{
        /* offset= */ tid * ts, /* size= */ ts,
        /* stride= */
        CreateSymbolicConstant(config.strides[i], mlir_context),
        /* upper_bound= */
        CreateSymbolicConstant(input_shape.dimensions(i), mlir_context)});
  }
  Tile input_tile_sym(*tiling_space_sym, std::move(input_dim_tiles_sym));
  absl::StatusOr<Tiles> output_tiles_sym =
      PropagateTileToOutput(*tiling_space_sym, *reshape, input_tile_sym, 0);

  if (!output_tiles_sym.ok()) {
    VLOG(2) << "Unsupported reshape configuration" << output_tiles_sym.status();
    stats.total_unsupported++;
    return;
  }

  // 2. Concrete propagation
  absl::StatusOr<std::unique_ptr<TilingSpace>> tiling_space_conc_or =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(p0), mlir_context);
  CHECK_OK(tiling_space_conc_or.status());
  std::unique_ptr<TilingSpace> tiling_space_conc =
      std::move(tiling_space_conc_or.value());
  CHECK_OK(tiling_space_conc->AssignTileSizes(config.sizes));

  llvm::SmallVector<DimTile> input_dim_tiles_conc;
  input_dim_tiles_conc.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    SymbolicExpr tid = CreateDimExpr(i, mlir_context);
    SymbolicExpr ts = CreateSymbolicConstant(config.sizes[i], mlir_context);
    input_dim_tiles_conc.push_back(DimTile{
        /* offset= */
        CreateSymbolicConstant(config.offsets[i], mlir_context) + tid * ts,
        /* size= */ ts, /* stride= */
        CreateSymbolicConstant(config.strides[i], mlir_context),
        /* upper_bound= */
        CreateSymbolicConstant(config.upper_bounds[i], mlir_context)});
  }
  Tile input_tile_conc(*tiling_space_conc, std::move(input_dim_tiles_conc));
  absl::StatusOr<Tiles> output_tiles_conc =
      PropagateTileToOutput(*tiling_space_conc, *reshape, input_tile_conc, 0);

  // 3. Evaluate happy case
  llvm::ArrayRef<DimTile> dim_tiles_ref =
      output_tiles_sym.value()[0].dim_tiles();
  llvm::SmallVector<DimTile> output_dim_tiles_sym(dim_tiles_ref.begin(),
                                                  dim_tiles_ref.end());
  Tile eval_output_tile(*tiling_space_conc, std::move(output_dim_tiles_sym));
  VLOG(1) << "Before Replace: " << eval_output_tile.ToString();
  llvm::DenseMap<SymbolicExpr, SymbolicExpr> ts_replacements;
  for (int i = 0; i < rank; ++i) {
    SymbolicExpr ts = CreateSymbolExpr(i, rank, mlir_context);
    ts_replacements[ts] = CreateSymbolicConstant(config.sizes[i], mlir_context);
  }
  eval_output_tile.Replace(ts_replacements);
  VLOG(1) << "After Replace: " << eval_output_tile.ToString();
  eval_output_tile.Simplify();
  VLOG(1) << "After Simplify: " << eval_output_tile.ToString();
  absl::Status equiv_status =
      VerifyTileEquivalence(input_tile_conc, input_shape, eval_output_tile,
                            output_shape, tiling_space_conc.get());
  bool happy_case_correct = equiv_status.ok();
  bool concrete_accepted = output_tiles_conc.ok();

  if (happy_case_correct && !concrete_accepted) {
    VLOG(2) << absl::StrCat(
        "\nFalse negative:\n", "Config: sizes=[",
        absl::StrJoin(config.sizes, ","), "] strides=[",
        absl::StrJoin(config.strides, ","), "] offsets=[",
        absl::StrJoin(config.offsets, ","), "] upper_bounds=[",
        absl::StrJoin(config.upper_bounds, ","), "]\n",
        "Input Tile (Before Prop): \n", input_tile_conc.ToString(), "\n",
        "Output Tile (After Prop): \n", eval_output_tile.ToString(), "\n",
        output_tiles_conc.status().message(), "\n");
    stats.false_negatives++;
  } else if (!happy_case_correct && concrete_accepted) {
    std::cout << absl::StrCat(
        "\nFalse positive:\n", "Config: sizes=[",
        absl::StrJoin(config.sizes, ","), "] strides=[",
        absl::StrJoin(config.strides, ","), "] offsets=[",
        absl::StrJoin(config.offsets, ","), "] upper_bounds=[",
        absl::StrJoin(config.upper_bounds, ","), "]\n",
        "Input Tile (Before Prop): \n", input_tile_conc.ToString(), "\n",
        "Output Tile (After Prop): \n", eval_output_tile.ToString(), "\n",
        equiv_status.message(), "\n");
    stats.false_positives++;
  }
}

void ProcessReshape(const Shape& input_shape, const Shape& output_shape,
                    bool bypass_limit) {
  std::cout << "\n==========================================================="
               "=======\n";
  std::cout << "Validating reshape: " << input_shape.ToString() << " -> "
            << output_shape.ToString() << "\n";
  std::cout << "============================================================="
               "=====\n";

  mlir::MLIRContext mlir_context;
  HloComputation::Builder builder("entry");
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "p0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(output_shape, p0));

  absl::StatusOr<std::unique_ptr<TilingSpace>> tiling_space_sym_or =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(p0), &mlir_context);
  CHECK_OK(tiling_space_sym_or.status());
  std::unique_ptr<TilingSpace> tiling_space_sym =
      std::move(tiling_space_sym_or.value());
  int64_t rank = tiling_space_sym->num_dimensions();

  // Generate Cartesian products
  std::vector<std::vector<int64_t>> size_configs(rank);
  std::vector<std::vector<int64_t>> stride_configs(rank);
  std::vector<std::vector<int64_t>> offset_configs(rank);
  std::vector<std::vector<int64_t>> upper_bound_configs(rank);

  int64_t num_configs = 1;
  for (int i = 0; i < rank; ++i) {
    int64_t dim = input_shape.dimensions(i);
    size_configs[i] = GetPowersOf2And(dim, {dim});
    stride_configs[i] = GetPowersOf2And(dim, {dim});
    offset_configs[i] = GetOffsets(dim);
    upper_bound_configs[i] = GetUpperBounds(dim);

    num_configs *= size_configs[i].size() * stride_configs[i].size() *
                   offset_configs[i].size() * upper_bound_configs[i].size();
  }

  std::cout << "Number of configs for this reshape: " << num_configs << "\n";
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
      ValidateConfig(input_shape, output_shape, reshape, p0,
                     tiling_space_sym.get(), &mlir_context, config, stats);
      return;
    }
    for (int64_t size : size_configs[dim_idx]) {
      config.sizes[dim_idx] = size;
      for (int64_t stride : stride_configs[dim_idx]) {
        config.strides[dim_idx] = stride;
        for (int64_t offset : offset_configs[dim_idx]) {
          config.offsets[dim_idx] = offset;
          for (int64_t ub : upper_bound_configs[dim_idx]) {
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
      "False Positives: ", stats.false_positives, " (",
      (stats.total_tested ? (stats.false_positives * 100.0 / stats.total_tested)
                          : 0),
      " %)\n", "False Negatives: ", stats.false_negatives, " (",
      (stats.total_tested ? (stats.false_negatives * 100.0 / stats.total_tested)
                          : 0),
      " %)\n", "Total Unsupported: ", stats.total_unsupported, " (",
      (stats.total_tested
           ? (stats.total_unsupported * 100.0 / stats.total_tested)
           : 0),
      " %)\n");
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
                "List of reshapes, e.g. '[4] [2,2] [12,1] [1,3,4]'"),
      tsl::Flag("bypass_limit", &bypass_limit,
                "Bypass the 100,000 combinations limit")};

  const std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (!parse_ok) {
    std::cerr << usage << "\n";
    return 1;
  }

  if (reshapes_str.empty()) {
    LOG(INFO) << "No reshapes provided.";
    return 0;
  }

  std::vector<std::string> parts = absl::StrSplit(
      reshapes_str, absl::ByAnyChar(" \n\r\t"), absl::SkipWhitespace());
  if (parts.size() % 2 != 0) {
    LOG(FATAL) << "Expected pairs of reshapes.";
  }

  std::vector<std::pair<xla::Shape, xla::Shape>> reshapes;
  for (size_t i = 0; i < parts.size(); i += 2) {
    std::vector<int64_t> in_dims = xla::gpu::experimental::ParseShape(parts[i]);
    std::vector<int64_t> out_dims =
        xla::gpu::experimental::ParseShape(parts[i + 1]);
    reshapes.push_back({xla::ShapeUtil::MakeShape(xla::F32, in_dims),
                        xla::ShapeUtil::MakeShape(xla::F32, out_dims)});
  }

  xla::gpu::experimental::RunValidation(reshapes, bypass_limit);
  return 0;
}
