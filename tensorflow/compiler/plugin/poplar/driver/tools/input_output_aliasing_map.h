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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INPUT_OUTPUT_ALIASING_MAP_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INPUT_OUTPUT_ALIASING_MAP_H_

#include <vector>
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {

/*
 * The goal of this class is to categorize the inputs and outputs of the
 * computation into Streamed or ResourceVariable.  Also, if one of the outputs
 * is a resource variable, and it is updated by the computation (and therefore
 * is also an input), then this is recorded too.
 */
class InputOutputAliasingMap {
 public:
  // A class which describes the aliasing information of an input
  class InputInfo {
   public:
    enum class Type {
      // StreamedVariable is a variable which is streamed to the device during
      // every execution of the main sequence
      StreamedVariable,
      // ResourceModified is a variable which is passed to the device before the
      // first execution the main sequence and is mapped to an output variable,
      // which means we want it to stay resident on the device between every
      // execution of the main sequence
      ResourceModified,
      // ResourceNotModified is a variable which is passed to the device before
      // the first execution the main sequence and is *not* mapped to an output
      // variable, which means we want it to stay *unchanged* resident on the
      // device between every execution of the main sequence
      ResourceNotModified,
    };

    InputInfo() = delete;
    InputInfo(const Type type) : type_(type), output_index_(0) {}
    InputInfo(const Type type, const uint64 output_index)
        : type_(type), output_index_(output_index) {}

    const bool IsStreaming() const;
    const bool IsResource() const;
    const bool IsResourceNotModified() const;
    const uint64 GetOutputIndex() const;

   private:
    Type type_;
    uint64 output_index_;
  };

  // A class which describes the aliasing information of an output
  class OutputInfo {
   public:
    enum class Type {
      // StreamedVariable is a variable which is streamed from the device during
      // every execution of the main sequence
      StreamedVariable,
      // ResourceModified is a variable which is passed to the device before the
      // first execution the main sequence and is mapped to an output variable,
      // which means we want it to stay resident on the device between every
      // execution of the main sequence
      ResourceModified,
      // ResourceOutputOnly is a variable usually created for an initialisation
      // graph, indicating that this variable will be either ResourceModified or
      // ResourceNotModified for the next graph and should stay resident on the
      // device (however not supported in Poplar yet).
      ResourceOutputOnly,
    };

    OutputInfo() = delete;
    OutputInfo(const Type& type) : type_(type), input_index_(0) {}
    OutputInfo(const Type& type, const uint64 input_index)
        : type_(type), input_index_(input_index) {}

    const bool IsStreaming() const;
    const bool IsResource() const;
    const bool IsResourceModified() const;
    const uint64 GetInputIndex() const;

   private:
    Type type_;
    uint64 input_index_;
  };

  InputOutputAliasingMap(const HloModule* module);

  const std::vector<InputInfo>& GetEntryInputInfos() const;
  const std::vector<OutputInfo>& GetEntryOutputInfos() const;
  const uint64& GetNumStreamingInputs();
  const uint64& GetNumStreamingOutputs();

  std::string ToString() const;

 private:
  std::vector<InputInfo> entry_input_infos_;
  std::vector<OutputInfo> entry_output_infos_;

  uint64 num_streaming_inputs_;
  uint64 num_streaming_outputs_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
