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

namespace sep = ::perftools::gputools::poplarplugin;

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
                   const sep::OutputMap& output_map,
                   const std::vector<Shape>& parameter_shapes);
  ~PoplarExecutable() override;

  StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteOnStream(
          const ServiceExecutableRunOptions* run_options,
          tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
          HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteAsyncOnStream(
          const ServiceExecutableRunOptions* run_options,
          tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) override;

  static int64 ShapeSizeBytes(const Shape& shape);

 private:
  friend class GraphCompileIoMapTest;

  std::shared_ptr<poplar::Engine> poplar_engine_;
  sep::OutputMap output_map_;
  std::vector<Shape> parameter_shapes_;
  bool first_execution_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutable);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
