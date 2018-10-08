#include "tensorflow/compiler/plugin/poplar/driver/input_output_aliasing_map.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <vector>

namespace xla {
namespace poplarplugin {

InputOutputAliasingMap::InputOutputAliasingMap(const HloModule* module) {
  // First go through all the inputs
  // Marked as either streaming or not modified resource - will be changed to a
  // modified resource if the outputs modify it
  const auto& inputs = module->entry_computation()->parameter_instructions();
  uint64 num_resource_inputs = module->config().resource_input_count();
  // TF will mark resources in initilizer graphs as resources, but these are not
  // actually passed in as parameters, this makes sure that the number of those
  // is correct
  if (num_resource_inputs > inputs.size()) {
    num_resource_inputs = inputs.size();
  }
  num_streaming_inputs_ = inputs.size() - num_resource_inputs;

  for (uint64 idx = 0; idx < inputs.size(); ++idx) {
    const InputInfo::Type type = idx < num_streaming_inputs_
                                     ? InputInfo::Type::StreamedVariable
                                     : InputInfo::Type::ResourceNotModified;
    entry_input_infos_.push_back(InputInfo(type));
  }

  // Go through all the outputs
  const auto& root = module->entry_computation()->root_instruction();
  const uint64 num_outputs = ShapeUtil::IsTuple(root->shape())
                                 ? ShapeUtil::TupleElementCount(root->shape())
                                 : 1;

  const auto resource_update_to_input_index =
      module->config().resource_update_to_input_index();
  uint64 num_resource_updates = resource_update_to_input_index.size();
  num_streaming_outputs_ = num_outputs - num_resource_updates;
  for (uint64 idx = 0; idx < num_outputs; ++idx) {
    if (idx < num_streaming_outputs_) {
      entry_output_infos_.push_back(
          OutputInfo(OutputInfo::Type::StreamedVariable));
    } else {
      const uint64 input_index =
          resource_update_to_input_index[idx - num_streaming_outputs_];
      if (num_streaming_inputs_ <= input_index && input_index < inputs.size()) {
        // If the resource input index is in the right range, then map it as
        // Input <-> Output
        entry_output_infos_.push_back(
            OutputInfo(OutputInfo::Type::ResourceModified, input_index));
        // Update the input info to reflect that it is mapped
        entry_input_infos_[input_index] =
            InputInfo(InputInfo::Type::ResourceModified, idx);
      } else {
        // Otherwise it's a resource output
        entry_output_infos_.push_back(
            OutputInfo(OutputInfo::Type::ResourceOutputOnly));
      }
    }
  }
}

const std::vector<InputOutputAliasingMap::InputInfo>&
InputOutputAliasingMap::GetEntryInputInfos() const {
  return entry_input_infos_;
}

const std::vector<InputOutputAliasingMap::OutputInfo>&
InputOutputAliasingMap::GetEntryOutputInfos() const {
  return entry_output_infos_;
}

const uint64& InputOutputAliasingMap::GetNumStreamingInputs() {
  return num_streaming_inputs_;
}

const uint64& InputOutputAliasingMap::GetNumStreamingOutputs() {
  return num_streaming_outputs_;
}

const bool InputOutputAliasingMap::InputInfo::IsStreaming() const {
  return type_ == InputOutputAliasingMap::InputInfo::Type::StreamedVariable;
}

const bool InputOutputAliasingMap::InputInfo::IsResource() const {
  return type_ == InputOutputAliasingMap::InputInfo::Type::ResourceModified ||
         type_ == InputOutputAliasingMap::InputInfo::Type::ResourceNotModified;
}

const bool InputOutputAliasingMap::InputInfo::IsResourceNotModified() const {
  return type_ == InputOutputAliasingMap::InputInfo::Type::ResourceNotModified;
}

const uint64 InputOutputAliasingMap::InputInfo::GetOutputIndex() const {
  return output_index_;
}

const bool InputOutputAliasingMap::OutputInfo::IsStreaming() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::StreamedVariable;
}

const bool InputOutputAliasingMap::OutputInfo::IsResourceModified() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceModified;
}

const uint64 InputOutputAliasingMap::OutputInfo::GetInputIndex() const {
  return input_index_;
}

}  // namespace poplarplugin
}  // namespace xla
