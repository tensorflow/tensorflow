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

#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "tensorflow/contrib/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace toco {

bool ParseTocoFlagsFromCommandLineFlags(
    int* argc, char* argv[], string* msg,
    ParsedTocoFlags* parsed_toco_flags_ptr) {
  using tensorflow::Flag;
  ParsedTocoFlags& parsed_flags = *parsed_toco_flags_ptr;
  std::vector<tensorflow::Flag> flags = {
      Flag("input_file", parsed_flags.input_file.bind(),
           parsed_flags.input_file.default_value(),
           "Input file (model of any supported format). For Protobuf "
           "formats, both text and binary are supported regardless of file "
           "extension."),
      Flag("savedmodel_directory", parsed_flags.savedmodel_directory.bind(),
           parsed_flags.savedmodel_directory.default_value(),
           "Full path to the directory containing the SavedModel."),
      Flag("output_file", parsed_flags.output_file.bind(),
           parsed_flags.output_file.default_value(),
           "Output file. "
           "For Protobuf formats, the binary format will be used."),
      Flag("input_format", parsed_flags.input_format.bind(),
           parsed_flags.input_format.default_value(),
           "Input file format. One of: TENSORFLOW_GRAPHDEF, TFLITE."),
      Flag("output_format", parsed_flags.output_format.bind(),
           parsed_flags.output_format.default_value(),
           "Output file format. "
           "One of TENSORFLOW_GRAPHDEF, TFLITE, GRAPHVIZ_DOT."),
      Flag("savedmodel_tagset", parsed_flags.savedmodel_tagset.bind(),
           parsed_flags.savedmodel_tagset.default_value(),
           "Comma-separated set of tags identifying the MetaGraphDef within "
           "the SavedModel to analyze. All tags in the tag set must be "
           "specified."),
      Flag("default_ranges_min", parsed_flags.default_ranges_min.bind(),
           parsed_flags.default_ranges_min.default_value(),
           "If defined, will be used as the default value for the min bound "
           "of min/max ranges used for quantization."),
      Flag("default_ranges_max", parsed_flags.default_ranges_max.bind(),
           parsed_flags.default_ranges_max.default_value(),
           "If defined, will be used as the default value for the max bound "
           "of min/max ranges used for quantization."),
      Flag("inference_type", parsed_flags.inference_type.bind(),
           parsed_flags.inference_type.default_value(),
           "Target data type of arrays in the output file (for input_arrays, "
           "this may be overridden by inference_input_type). "
           "One of FLOAT, QUANTIZED_UINT8."),
      Flag("inference_input_type", parsed_flags.inference_input_type.bind(),
           parsed_flags.inference_input_type.default_value(),
           "Target data type of input arrays. "
           "If not specified, inference_type is used. "
           "One of FLOAT, QUANTIZED_UINT8."),
      Flag("input_type", parsed_flags.input_type.bind(),
           parsed_flags.input_type.default_value(),
           "Deprecated ambiguous flag that set both --input_data_types and "
           "--inference_input_type."),
      Flag("input_types", parsed_flags.input_types.bind(),
           parsed_flags.input_types.default_value(),
           "Deprecated ambiguous flag that set both --input_data_types and "
           "--inference_input_type. Was meant to be a "
           "comma-separated list, but this was deprecated before "
           "multiple-input-types was ever properly supported."),

      Flag("drop_fake_quant", parsed_flags.drop_fake_quant.bind(),
           parsed_flags.drop_fake_quant.default_value(),
           "Ignore and discard FakeQuant nodes. For instance, to "
           "generate plain float code without fake-quantization from a "
           "quantized graph."),
      Flag(
          "reorder_across_fake_quant",
          parsed_flags.reorder_across_fake_quant.bind(),
          parsed_flags.reorder_across_fake_quant.default_value(),
          "Normally, FakeQuant nodes must be strict boundaries for graph "
          "transformations, in order to ensure that quantized inference has "
          "the exact same arithmetic behavior as quantized training --- which "
          "is the whole point of quantized training and of FakeQuant nodes in "
          "the first place. "
          "However, that entails subtle requirements on where exactly "
          "FakeQuant nodes must be placed in the graph. Some quantized graphs "
          "have FakeQuant nodes at unexpected locations, that prevent graph "
          "transformations that are necessary in order to generate inference "
          "code for these graphs. Such graphs should be fixed, but as a "
          "temporary work-around, setting this reorder_across_fake_quant flag "
          "allows TOCO to perform necessary graph transformaitons on them, "
          "at the cost of no longer faithfully matching inference and training "
          "arithmetic."),
      Flag("allow_custom_ops", parsed_flags.allow_custom_ops.bind(),
           parsed_flags.allow_custom_ops.default_value(),
           "If true, allow TOCO to create TF Lite Custom operators for all the "
           "unsupported TensorFlow ops."),
      Flag(
          "drop_control_dependency",
          parsed_flags.drop_control_dependency.bind(),
          parsed_flags.drop_control_dependency.default_value(),
          "If true, ignore control dependency requirements in input TensorFlow "
          "GraphDef. Otherwise an error will be raised upon control dependency "
          "inputs."),
      Flag("debug_disable_recurrent_cell_fusion",
           parsed_flags.debug_disable_recurrent_cell_fusion.bind(),
           parsed_flags.debug_disable_recurrent_cell_fusion.default_value(),
           "If true, disable fusion of known identifiable cell subgraphs into "
           "cells. This includes, for example, specific forms of LSTM cell."),
  };
  bool asked_for_help =
      *argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-help"));
  if (asked_for_help) {
    *msg += tensorflow::Flags::Usage(argv[0], flags);
    return false;
  } else {
    return tensorflow::Flags::Parse(argc, argv, flags);
  }
}

namespace {

// Defines the requirements for a given flag. kUseDefault means the default
// should be used in cases where the value isn't specified by the user.
enum class FlagRequirement {
  kNone,
  kMustBeSpecified,
  kMustNotBeSpecified,
  kUseDefault,
};

// Enforces the FlagRequirements are met for a given flag.
template <typename T>
void EnforceFlagRequirement(const T& flag, const string& flag_name,
                            FlagRequirement requirement) {
  if (requirement == FlagRequirement::kMustBeSpecified) {
    QCHECK(flag.specified()) << "Missing required flag " << flag_name;
  }
  if (requirement == FlagRequirement::kMustNotBeSpecified) {
    QCHECK(!flag.specified())
        << "Given other flags, this flag should not have been specified: "
        << flag_name;
  }
}

// Gets the value from the flag if specified. Returns default if the
// FlagRequirement is kUseDefault.
template <typename T>
absl::optional<T> GetFlagValue(const Arg<T>& flag,
                               FlagRequirement requirement) {
  if (flag.specified()) return flag.value();
  if (requirement == FlagRequirement::kUseDefault) return flag.default_value();
  return absl::optional<T>();
}

}  // namespace

void ReadTocoFlagsFromCommandLineFlags(const ParsedTocoFlags& parsed_toco_flags,
                                       TocoFlags* toco_flags) {
  namespace port = toco::port;
  port::CheckInitGoogleIsDone("InitGoogle is not done yet");

#define READ_TOCO_FLAG(name, requirement)                                \
  do {                                                                   \
    EnforceFlagRequirement(parsed_toco_flags.name, #name, requirement);  \
    auto flag_value = GetFlagValue(parsed_toco_flags.name, requirement); \
    if (flag_value.has_value()) {                                        \
      toco_flags->set_##name(flag_value.value());                        \
    }                                                                    \
  } while (false)

#define PARSE_TOCO_FLAG(Type, name, requirement)                         \
  do {                                                                   \
    EnforceFlagRequirement(parsed_toco_flags.name, #name, requirement);  \
    auto flag_value = GetFlagValue(parsed_toco_flags.name, requirement); \
    if (flag_value.has_value()) {                                        \
      Type x;                                                            \
      QCHECK(Type##_Parse(flag_value.value(), &x))                       \
          << "Unrecognized " << #Type << " value "                       \
          << parsed_toco_flags.name.value();                             \
      toco_flags->set_##name(x);                                         \
    }                                                                    \
  } while (false)

  PARSE_TOCO_FLAG(FileFormat, input_format, FlagRequirement::kUseDefault);
  PARSE_TOCO_FLAG(FileFormat, output_format, FlagRequirement::kUseDefault);
  PARSE_TOCO_FLAG(IODataType, inference_type, FlagRequirement::kNone);
  PARSE_TOCO_FLAG(IODataType, inference_input_type, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_ranges_min, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_ranges_max, FlagRequirement::kNone);
  READ_TOCO_FLAG(drop_fake_quant, FlagRequirement::kNone);
  READ_TOCO_FLAG(reorder_across_fake_quant, FlagRequirement::kNone);
  READ_TOCO_FLAG(allow_custom_ops, FlagRequirement::kNone);
  READ_TOCO_FLAG(drop_control_dependency, FlagRequirement::kNone);

  // Deprecated flag handling.
  if (parsed_toco_flags.input_type.specified()) {
    LOG(WARNING)
        << "--input_type is deprecated. It was an ambiguous flag that set both "
           "--input_data_types and --inference_input_type. If you are trying "
           "to complement the input file with information about the type of "
           "input arrays, use --input_data_type. If you are trying to control "
           "the quantization/dequantization of real-numbers input arrays in "
           "the output file, use --inference_input_type.";
    toco::IODataType input_type;
    QCHECK(toco::IODataType_Parse(parsed_toco_flags.input_type.value(),
                                  &input_type));
    toco_flags->set_inference_input_type(input_type);
  }
  if (parsed_toco_flags.input_types.specified()) {
    LOG(WARNING)
        << "--input_types is deprecated. It was an ambiguous flag that set "
           "both --input_data_types and --inference_input_type. If you are "
           "trying to complement the input file with information about the "
           "type of input arrays, use --input_data_type. If you are trying to "
           "control the quantization/dequantization of real-numbers input "
           "arrays in the output file, use --inference_input_type.";
    std::vector<string> input_types =
        absl::StrSplit(parsed_toco_flags.input_types.value(), ',');
    QCHECK(!input_types.empty());
    for (int i = 1; i < input_types.size(); i++) {
      QCHECK_EQ(input_types[i], input_types[0]);
    }
    toco::IODataType input_type;
    QCHECK(toco::IODataType_Parse(input_types[0], &input_type));
    toco_flags->set_inference_input_type(input_type);
  }

#undef READ_TOCO_FLAG
#undef PARSE_TOCO_FLAG
}
}  // namespace toco
