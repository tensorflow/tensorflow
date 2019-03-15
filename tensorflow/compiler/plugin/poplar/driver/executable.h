/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXECUTABLE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"

#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

// A Poplar executable is a wrapper around and Engine, with
// the execution Sequence program, input tensors and output
// tensor recorded.
class PoplarExecutable : public Executable {
 public:
  PoplarExecutable(std::unique_ptr<HloModule> hlo_module,
                   std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
                   std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
                   std::unique_ptr<poplar::Engine> engine,
                   const InputOutputAliasingMap& input_output_aliasing_map,
                   const bool is_constant_graph,
                   std::vector<std::vector<Literal>> literal_output,
                   const bool is_remap_graph,
                   std::vector<uint64> remaped_output, int replication_count_,
                   const InfeedInfos& infeed_infos,
                   const OutfeedInfos& outfeed_infos);

  ~PoplarExecutable() override;

  StatusOr<ScopedShapedBuffer> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) override;

  static int64 ShapeSizeBytes(const Shape& shape);

  int64 ExecutionCount() const { return execution_count_; }

  void OnEngineLoaded() { execution_count_ = 0; }

  const InputOutputAliasingMap& GetInputOutputAliasingMap() const {
    return input_output_aliasing_map_;
  }

  poplar::Engine* Engine() const { return poplar_engine_.get(); }

  const std::vector<std::vector<Literal>>& LiteralValue() const {
    return literal_output_;
  }

  const InfeedInfos& GetInfeedInfos() const { return infeed_infos_; }

  const OutfeedInfos& GetOutfeedInfos() const { return outfeed_infos_; }

  const bool IsConstantGraph() const { return is_constant_graph_; }

  const std::vector<uint64>& RemapMap() const { return remaped_output_; }

  const bool IsRemapGraph() const { return is_remap_graph_; }

  static StatusOr<PoplarExecutable*> Deserialize(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
      const std::string& filename);

  static Status Serialize(const PoplarExecutable& executable,
                          const std::string& filename);

 private:
  friend class GraphCompileIoMapTest;

  std::unique_ptr<poplar::Engine> poplar_engine_;
  InputOutputAliasingMap input_output_aliasing_map_;
  std::vector<std::vector<Literal>> literal_output_;
  const bool is_constant_graph_;
  std::vector<uint64> remaped_output_;
  const bool is_remap_graph_;
  int64 execution_count_;
  int replication_count_;
  InfeedInfos infeed_infos_;
  OutfeedInfos outfeed_infos_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutable);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
