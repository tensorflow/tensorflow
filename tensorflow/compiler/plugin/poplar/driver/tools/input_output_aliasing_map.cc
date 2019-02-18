#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <vector>

namespace xla {
namespace poplarplugin {

InputOutputAliasingMap::InputOutputAliasingMap(const HloModule* module) {
  const auto& inputs = module->entry_computation()->parameter_instructions();
  uint64 num_arguments = module->config().argument_count();
  uint64 num_resource_inputs = module->config().resource_input_count();
  const auto& input_mapping = module->config().input_mapping();
  const auto& resource_update_to_input_index =
      module->config().resource_update_to_input_index();

  /*
   * An XLA entry computation has a set of input parameters.  These map to a
   * combination of the inputs to the _XlaRun TF Op, and the resources which
   * are used by it.
   *
   * The `num_arguments` variable stores the total number of arguments in the
   * original _XlaRun operation.  `input_mapping` contains a map from the XLA
   * computation parameters to the _XlaRun arguments.  The number of entries
   * in the `input_mapping` will be less than `num_arguments` when there are
   * uninitialized ResourceVariables, which are not passed to the XLA
   * Computation.
   *
   * The `num_resource_inputs` gives the total number of resource variables in
   * the original _XlaRun Op.
   */

  num_streaming_inputs_ = num_arguments - num_resource_inputs;

  for (uint64 idx = 0; idx < inputs.size(); ++idx) {
    bool is_resource = idx < input_mapping.size() &&
                       input_mapping[idx] >= num_streaming_inputs_;
    const InputInfo::Type type = is_resource
                                     ? InputInfo::Type::ResourceNotModified
                                     : InputInfo::Type::StreamedVariable;
    entry_input_infos_.push_back(InputInfo(type));
  }

  /*
   * The `resource_update_to_input_index` is a map from the computation output
   * to a _XlaRun input.
   */
  const auto& root = module->entry_computation()->root_instruction();
  const uint64 num_outputs =
      root->shape().IsTuple() ? ShapeUtil::TupleElementCount(root->shape()) : 1;

  uint64 num_resource_updates = resource_update_to_input_index.size();
  num_streaming_outputs_ = num_outputs - num_resource_updates;

  for (uint64 idx = 0; idx < num_outputs; ++idx) {
    if (idx < num_streaming_outputs_) {
      entry_output_infos_.push_back(
          OutputInfo(OutputInfo::Type::StreamedVariable));
    } else {
      const uint64 resource_idx = idx - num_streaming_outputs_;
      const uint64 input_index = resource_update_to_input_index[resource_idx];

      auto input_map_it = absl::c_find(input_mapping, input_index);
      if (input_map_it != input_mapping.end()) {
        int64 parameter_index =
            std::distance(input_mapping.begin(), input_map_it);

        if (num_streaming_inputs_ <= parameter_index) {
          entry_output_infos_.push_back(
              OutputInfo(OutputInfo::Type::ResourceModified, input_index));
          entry_input_infos_[parameter_index] =
              InputInfo(InputInfo::Type::ResourceModified, idx);
        } else {
          entry_output_infos_.push_back(
              OutputInfo(OutputInfo::Type::ResourceOutputOnly));
        }
      } else {
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

const bool InputOutputAliasingMap::OutputInfo::IsResource() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceModified ||
         type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceOutputOnly;
}

const bool InputOutputAliasingMap::OutputInfo::IsResourceModified() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceModified;
}

const uint64 InputOutputAliasingMap::OutputInfo::GetInputIndex() const {
  return input_index_;
}

std::string InputOutputAliasingMap::ToString() const {
  std::stringstream ss;
  ss << "== Input information ==\n";
  for (int i = 0; i < entry_input_infos_.size(); i++) {
    auto& ip = entry_input_infos_[i];
    ss << " " << i << ":\n";
    ss << " -IsStreaming=" << ip.IsStreaming() << "\n";
    ss << " -IsResource=" << ip.IsResource() << "\n";
    ss << " -IsResourceNotModified=" << ip.IsResourceNotModified() << "\n";
    ss << " -GetOutputIndex=" << ip.GetOutputIndex() << "\n";
  }
  ss << "== Output information ==\n";
  for (int i = 0; i < entry_output_infos_.size(); i++) {
    auto& op = entry_output_infos_[i];
    ss << " " << i << ":\n";
    ss << " -IsStreaming=" << op.IsStreaming() << "\n";
    ss << " -IsResourceModified=" << op.IsResourceModified() << "\n";
    ss << " -GetInputIndex=" << op.GetInputIndex() << "\n";
  }

  return ss.str();
}

}  // namespace poplarplugin
}  // namespace xla
