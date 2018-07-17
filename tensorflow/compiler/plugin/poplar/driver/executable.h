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
                   std::shared_ptr<poplar::Engine> engine,
                   const OutputMap& output_map,
                   std::vector<std::unique_ptr<Literal>> literal_output,
                   const std::vector<bool>& parameter_streamed,
                   const std::vector<bool>& output_streamed);

  ~PoplarExecutable() override;

  StatusOr<ScopedShapedBuffer> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) override;

  static int64 ShapeSizeBytes(const Shape& shape);

  bool DumpReport() const { return first_execution_; }

  const OutputMap& OutputMapping() const { return output_map_; }

  const std::shared_ptr<poplar::Engine>& Engine() const {
    return poplar_engine_;
  }

  const std::vector<bool>& ParameterStreamed() const {
    return parameter_streamed_;
  }

  const std::vector<bool>& OutputStreamed() const { return output_streamed_; }

  const std::vector<std::unique_ptr<Literal>>& LiteralValue() const {
    return literal_output_;
  }

  static StatusOr<PoplarExecutable*> Deserialize(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
      const std::string& filename);

  static Status Serialize(const PoplarExecutable& executable,
                          const std::string& filename);

 private:
  friend class GraphCompileIoMapTest;

  std::shared_ptr<poplar::Engine> poplar_engine_;
  OutputMap output_map_;
  std::vector<std::unique_ptr<Literal>> literal_output_;
  std::vector<bool> parameter_streamed_;
  std::vector<bool> output_streamed_;
  bool first_execution_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutable);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
