/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_ENTRY_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_ENTRY_VISITOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

/*
 * This visitor handles inputs and outputs of the entry computation in a module.
 */
class EntryVisitor : public FullVisitor {
 public:
  EntryVisitor(poplar::Graph& graph, CompilerResources& resources,
               uint64 num_parameters, uint64 num_outputs)
      : FullVisitor(graph, resources),
        parameter_streamed(num_parameters),
        output_streamed(num_outputs),
        all_outputs_are_parameters(false) {}

  Status HandleParameter(HloInstruction* inst);
  Status FinishVisit(HloInstruction* inst);
  Status Postprocess(HloInstruction* inst);

  const OutputMap& GetOutputMap();
  const std::vector<bool>& GetParameterStreamed();
  const std::vector<bool>& GetOutputStreamed();

  const bool AreAllOutputsParameters();

  const poplar::program::Sequence& GetHostToDevice();
  const poplar::program::Sequence& GetDeviceToHost();

 private:
  Status StreamOutputs(HloInstruction* inst, uint64 start_idx,
                       OutVector outputs);

  std::set<HloInstruction*> non_standard_parameter_layout;

  OutputMap output_map;
  std::vector<bool> parameter_streamed;
  std::vector<bool> output_streamed;

  bool all_outputs_are_parameters;

  poplar::program::Sequence host_to_device;
  poplar::program::Sequence device_to_host;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
