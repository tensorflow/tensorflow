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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/debug.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/shape.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/path.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTTODOTPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

using AtomExecutableMap = ::xla::ifrt::AtomExecutableMap;

static const mlir::StringRef kLineStyleLargeTransfer = "solid";
static const mlir::StringRef kLineStyleSmallTransfer = "dashed";
static const mlir::StringRef kShapeNode = "ellipse";

struct ExecutableStats {
  int64_t peak_memory_in_bytes = -1;
  float flops = -1.0;
  int64_t argument_size_in_bytes = -1;
  int64_t output_size_in_bytes = -1;
  int64_t alias_size_in_bytes = -1;

  std::string ToString() const {
    std::string peak_memory_str = "N/A";
    if (peak_memory_in_bytes >= 0) {
      peak_memory_str =
          tsl::strings::HumanReadableNumBytes(peak_memory_in_bytes);
    }
    std::string argument_size_str = "N/A";
    if (argument_size_in_bytes >= 0) {
      argument_size_str =
          tsl::strings::HumanReadableNumBytes(argument_size_in_bytes);
    }
    std::string output_size_str = "N/A";
    if (output_size_in_bytes >= 0) {
      output_size_str =
          tsl::strings::HumanReadableNumBytes(output_size_in_bytes);
    }
    std::string alias_size_str = "N/A";
    if (alias_size_in_bytes >= 0) {
      alias_size_str = alias_size_str =
          tsl::strings::HumanReadableNumBytes(alias_size_in_bytes);
    }
    std::string flops_str = "N/A";
    if (flops >= 0.0) {
      flops_str = absl::StrFormat("%0.1f", flops / 1e9);
    }
    return absl::StrCat("Peak memory ", peak_memory_str, "\nArgument size ",
                        argument_size_str, "\nOutput size ", output_size_str,
                        "\nAlias size ", alias_size_str, "\nGFlops ",
                        flops_str);
  }
};

class IfrtToDotPass : public impl::IfrtToDotPassBase<IfrtToDotPass> {
 public:
  using impl::IfrtToDotPassBase<IfrtToDotPass>::IfrtToDotPassBase;

  explicit IfrtToDotPass(
      IfrtToDotPassOptions options,
      std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map)
      : IfrtToDotPassBase(std::move(options)),
        atom_executable_future_map_(std::move(atom_executable_future_map)) {}

  void runOnOperation() override;

 private:
  // Initializes a map from a devices attribute to a color to use for the dot
  // nodes corresponding to executable using the devices specified by the
  // attribute.
  void initMeshColorMapping(mlir::SymbolTableCollection& symbol_table,
                            mlir::ModuleOp module_op) {
    devices_attr_to_color_.clear();
    llvm::SmallVector<xla::ifrt::IfrtDevicesAttr> devices_attrs;
    module_op.walk([&](xla::ifrt::CallLoadedExecutableOp call_op)
                       -> mlir::WalkResult {
      xla::ifrt::LoadedExecutableOp loaded_exec_op =
          call_op.getCalleeOp(symbol_table);
      auto devices_attr = loaded_exec_op.getDevicesAttr();
      if (devices_attr_to_color_.insert({devices_attr, "0.0 1.0 1.0"}).second) {
        devices_attrs.push_back(devices_attr);
      }
      return mlir::WalkResult::advance();
    });
    for (auto [idx, devices_attr] : llvm::enumerate(devices_attrs)) {
      double hue = static_cast<double>(idx) / devices_attrs.size();
      devices_attr_to_color_[devices_attr] = absl::StrCat(hue, " 1.0 1.0");
    }
  }

  // Returns all values printed onto a stream as a string.
  static std::string strFromOs(
      mlir::function_ref<void(mlir::raw_ostream&)> func) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    func(os);
    return os.str();
  }

  // Escapes special characters such as '\n' and quotation marks.
  static std::string escapeString(std::string str) {
    return strFromOs([&](mlir::raw_ostream& os) { os.write_escaped(str); });
  }

  // Puts quotation marks around a given string.
  static std::string quoteString(const std::string& str) {
    return "\"" + str + "\"";
  }

  // Generates an attribute statement.
  std::string attrStmt(const mlir::Twine& key, const mlir::Twine& value) {
    return (key + " = " + value).str();
  }

  // Emits an attribute list.
  void emitAttrList(mlir::raw_ostream& os,
                    const absl::flat_hash_map<std::string, std::string>& map) {
    os << "[";
    llvm::interleaveComma(map, os, [&](const auto& it) {
      os << this->attrStmt(it.first, it.second);
    });
    os << "]";
  }

  // Emits a node statement.
  int emitNodeStmt(mlir::raw_ostream& os, std::string label,
                   mlir::StringRef shape = kShapeNode,
                   mlir::StringRef background = "") {
    int node_id = ++last_node_id_;
    absl::flat_hash_map<std::string, std::string> attrs;
    attrs["label"] = quoteString(escapeString(std::move(label)));
    attrs["shape"] = shape.str();
    if (!background.empty()) {
      attrs["style"] = "filled";
      attrs["fillcolor"] = ("\"" + background + "\"").str();
    }
    os << llvm::format("v%i ", node_id);
    emitAttrList(os, attrs);
    os << ";\n";
    return node_id;
  }

  // Emits an edge statement.
  void emitEdgeStmt(mlir::raw_ostream& os, int src_node_id, int dst_node_id,
                    std::string label, mlir::StringRef style) {
    absl::flat_hash_map<std::string, std::string> attrs;
    attrs["style"] = style.str();
    attrs["label"] = quoteString(escapeString(std::move(label)));
    edges_.push_back(strFromOs([&](mlir::raw_ostream& os) {
      os << llvm::format("v%i -> v%i ", src_node_id, dst_node_id);
      emitAttrList(os, attrs);
    }));
  }

  // Emits all edges. This function should be called after all node statements
  // have been emitted.
  void emitAllEdgeStmts(mlir::raw_ostream& os) {
    for (const std::string& edge : edges_) {
      os << edge << ";\n";
    }
    edges_.clear();
  }

  // Returns peak memory and flops stats from the executable.
  //
  // In case an executable does not offer such stats (e.g., MpmdReshard), the
  // function returns -1.0 for the corresponding stat.
  ExecutableStats getStatsFromExecutable(
      xla::ifrt::LoadedExecutableRef executable) {
    ExecutableStats stats = {/*peak_memory_in_bytes=*/-1,
                             /*flops=*/-1.0,
                             /*argument_size_in_bytes=*/-1,
                             /*output_size_in_bytes=*/-1,
                             /*alias_size_in_bytes=*/-1};
    // Get peak memory from the compiled memory stas.
    auto memory_stats = executable->GetCompiledMemoryStats();
    if (!memory_stats.ok()) {
      LOG(WARNING) << "Failed to get compiled memory stats for executable "
                   << executable->name() << ": " << memory_stats.status();
    } else {
      stats.argument_size_in_bytes = memory_stats->argument_size_in_bytes;
      stats.output_size_in_bytes = memory_stats->output_size_in_bytes;
      stats.alias_size_in_bytes = memory_stats->alias_size_in_bytes;
      stats.peak_memory_in_bytes = memory_stats->argument_size_in_bytes +
                                   memory_stats->output_size_in_bytes -
                                   memory_stats->alias_size_in_bytes +
                                   memory_stats->temp_size_in_bytes +
                                   memory_stats->generated_code_size_in_bytes;
    }

    // Get flops from the cost analysis.
    if (auto cost_analysis = executable->GetCostAnalysis();
        cost_analysis.ok()) {
      auto flops = cost_analysis->Get<float>("flops");
      if (flops.ok()) {
        stats.flops = *flops;
      } else {
        LOG(WARNING) << "Cost analysis of executable " << executable->name()
                     << " does not contain flops";
      }
    } else {
      LOG(WARNING) << "Failed to get cost analysis from executable "
                   << executable->name() << ": " << cost_analysis.status();
    }
    return stats;
  }

  // Map from atom program name to the executable. It is used to get stats
  // about the executables.
  std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map_;

  // Counter for generating unique node/subgraph identifiers.
  int last_node_id_ = 0;

  // A vector of edge statements. We accumulate them first so that we can emit
  // them after all node statements.
  std::vector<std::string> edges_;

  // Map from devices attribute to a color. This map is used to color
  // differently executable nodes that run on different devices.
  llvm::DenseMap<xla::ifrt::IfrtDevicesAttr, std::string>
      devices_attr_to_color_;
};

void IfrtToDotPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();

  std::string module_name = module_op.getName().value_or("unknown").str();
  // Include the module fingerprint in the file name to avoid exporting a
  // module multiple times.
  std::string file_path =
      tsl::io::JoinPath(dot_graph_dump_to,
                        absl::StrCat("ifrt_", module_name, "_",
                                     MlirModuleFingerprint(module_op), ".dot"));
  std::unique_ptr<tsl::WritableFile> f;
  if (const absl::Status status =
          tsl::Env::Default()->NewWritableFile(file_path, &f);
      !status.ok()) {
    LOG(ERROR) << "Could not create file " << file_path
               << " for writing: " << status;
    signalPassFailure();
    return;
  }

  mlir::SymbolTableCollection symbol_table;
  llvm::DenseMap<mlir::Value, int> val_to_node_id;

  initMeshColorMapping(symbol_table, module_op);

  AppendOnlyFileRawStream os(std::move(f));
  os << "digraph G {\n";
  // Show the executables from left to right.
  os << "graph [rankdir=LR]\n";

  mlir::WalkResult result = module_op.walk([&](mlir::Operation* op)
                                               -> mlir::WalkResult {
    auto result =
        llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<xla::ifrt::CallLoadedExecutableOp>([&](auto& op) {
              xla::ifrt::LoadedExecutableOp loaded_exec_op =
                  op.getCalleeOp(symbol_table);
              std::string atom_program_name = loaded_exec_op.getSymName().str();

              // Get memory and flops stats from the executable.
              auto exec_it =
                  atom_executable_future_map_->find(atom_program_name);
              if (exec_it == atom_executable_future_map_->end()) {
                op.emitOpError(
                    absl::StrCat("No executable found for atom program ",
                                 atom_program_name));
                return mlir::failure();
              }
              absl::StatusOr<LoadedExecutableRef> exec =
                  exec_it->second.Await();
              if (!exec.ok()) {
                op.emitOpError(absl::StrCat("Failed to compile atom program '",
                                            atom_program_name,
                                            "': ", exec.status()));
                return mlir::failure();
              }
              auto stats = getStatsFromExecutable(*std::move(exec));
              auto devices = loaded_exec_op.getDevices();

              // Add a DOT node for the executable only if its flops and peak
              // memory are above the minimum thresholds. This filtering avoids
              // generating large graphs with many irrelevant nodes.
              if (stats.flops < dot_graph_min_executable_flops &&
                  stats.peak_memory_in_bytes <
                      dot_graph_min_executable_peak_memory_bytes) {
                return mlir::success();
              }

              std::string exec_label = absl::StrFormat(
                  "%s\nNum devices %d First device %d Last device %d\n%s",
                  atom_program_name, devices.size(), devices[0],
                  devices[devices.size() - 1], stats.ToString());
              auto node = emitNodeStmt(
                  os, exec_label, kShapeNode,
                  devices_attr_to_color_[loaded_exec_op.getDevicesAttr()]);

              absl::flat_hash_map<int, int64_t>
                  per_device_array_size_between_execs;
              // Calculate per-device size of the arrays that are passed between
              // executables.
              for (const auto& input : op.getInputs()) {
                auto it = val_to_node_id.find(input);
                if (it == val_to_node_id.end()) {
                  // No node was created for the op creating this input because
                  // the op falls below the filtering thresholds.
                  continue;
                }
                const auto array_type =
                    llvm::cast<xla::ifrt::IfrtArrayType>(input.getType());
                if (array_type == nullptr) {
                  op.emitOpError(absl::StrCat("Input ",
                                              mlir::debugString(input),
                                              " is not an array type."));
                  return mlir::failure();
                }
                int64_t per_device_num_bytes = 0;

                auto dtype = xla::ifrt::ToIfrtDType(
                    array_type.getShape().getElementType());
                if (dtype.ok() && dtype->byte_size().has_value()) {
                  per_device_num_bytes = dtype->byte_size().value();
                } else {
                  // The dtype might not have a fixed size (e.g., string).
                  LOG(WARNING)
                      << "Failed to compute size in bytes for array type "
                      << mlir::debugString(array_type);
                }

                auto sharding_param_attr =
                    mlir::dyn_cast_or_null<xla::ifrt::IfrtShardingParamAttr>(
                        array_type.getShardingAttr());
                if (sharding_param_attr != nullptr) {
                  auto local_shape = sharding_param_attr.getSharding()
                                         .LocalShapeFromGlobalShape(
                                             array_type.getShape().getShape());
                  if (local_shape.ok()) {
                    xla::ifrt::Shape shard_shape(*local_shape);
                    per_device_num_bytes *= shard_shape.num_elements();
                    per_device_array_size_between_execs[it->second] +=
                        per_device_num_bytes;
                  } else {
                    op.emitOpError(
                        absl::StrCat("Could not get per shard shape for array ",
                                     mlir::debugString(array_type)));
                    return mlir::failure();
                  }
                } else {
                  op.emitOpError(
                      absl::StrCat("Only arrays with `ShardingParamAttr` "
                                   "are supported, but got array ",
                                   mlir::debugString(array_type)));
                  return mlir::failure();
                }
              }

              // Add edges between the dot nodes corresponding to the
              // executables creating the executable's inputs and its dot
              // node.
              for (const auto& [src_node_id, num_bytes] :
                   per_device_array_size_between_execs) {
                if (num_bytes >= dot_graph_min_per_device_transfer_size_bytes) {
                  auto line_style = (num_bytes < 10e6)
                                        ? kLineStyleSmallTransfer
                                        : kLineStyleLargeTransfer;
                  emitEdgeStmt(os, src_node_id, node,
                               tsl::strings::HumanReadableNumBytes(num_bytes),
                               line_style);
                }
              }

              for (const auto& output : op.getOutputs()) {
                val_to_node_id[output] = node;
              }
              return mlir::success();
            })
            .Case<xla::ifrt::CopyArraysOp>([&](auto& op) {
              for (const auto& [input, output] :
                   llvm::zip(op.getInputs(), op.getOutputs())) {
                if (auto it = val_to_node_id.find(input);
                    it != val_to_node_id.end()) {
                  val_to_node_id[output] = it->second;
                }
              }
              return mlir::success();
            })
            .Case<xla::ifrt::ReshardOp>([&](auto& op) {
              op.emitOpError(
                  "Dot graphs can be generated only after `ReshardOp`s have "
                  "been converted to `CallOp`s.");
              return mlir::failure();
            })
            .Case<xla::ifrt::CallOp>([&](auto& op) {
              op.emitOpError(
                  "Dot graphs can be generated only after atom programs "
                  "have been compiled.");
              return mlir::failure();
            })
            .Default(mlir::success());

    if (mlir::failed(result)) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  emitAllEdgeStmts(os);
  os << "}\n";

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createIfrtToDotPass(
    IfrtToDotPassOptions options,
    std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map) {
  return std::make_unique<IfrtToDotPass>(std::move(options),
                                         std::move(atom_executable_future_map));
}

void registerIfrtToDotPass(
    IfrtToDotPassOptions options,
    std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map) {
  mlir::registerPass(
      [atom_executable_future_map = std::move(atom_executable_future_map),
       options = std::move(options)]() -> std::unique_ptr<mlir::Pass> {
        return createIfrtToDotPass(std::move(options),
                                   std::move(atom_executable_future_map));
      });
}

}  // namespace ifrt
}  // namespace xla
