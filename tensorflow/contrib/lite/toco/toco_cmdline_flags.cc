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
      Flag("output_file", parsed_flags.output_file.bind(),
           parsed_flags.output_file.default_value(),
           "Output file. "
           "For Protobuf formats, the binary format will be used."),
      Flag("input_format", parsed_flags.input_format.bind(),
           parsed_flags.input_format.default_value(),
           "Input file format. One of: tensorflow_graphdef, "),
      Flag("output_format", parsed_flags.output_format.bind(),
           parsed_flags.output_format.default_value(), "Output file format."),
      Flag("default_ranges_min", parsed_flags.default_ranges_min.bind(),
           parsed_flags.default_ranges_min.default_value(),
           "If defined, will be used as the default value for the min bound "
           "of min/max ranges used for quantization."),
      Flag("default_ranges_max", parsed_flags.default_ranges_max.bind(),
           parsed_flags.default_ranges_max.default_value(),
           "If defined, will be used as the default value for the max bound "
           "of min/max ranges used for quantization."),
      Flag("input_type", parsed_flags.input_type.bind(),
           parsed_flags.input_type.default_value(),
           "Data type of the input array in the "
           "output file. "),
      Flag("input_types", parsed_flags.input_types.bind(),
           parsed_flags.input_types.default_value(),
           "Data types of the input arrays in the "
           "output file. "
           "Comma-separated list matching the enumeration order of "
           "input_arrays."),
      Flag("inference_type", parsed_flags.inference_type.bind(),
           parsed_flags.inference_type.default_value(),
           "Data type, in the output file, of internal and output arrays "
           "that are FLOAT in the input file. Thus, the value FLOAT means "
           "keep doing floating-point inference, while the value "
           "QUANTIZED_UINT8 means replace all internal floating-point "
           "arithmetic by integer arithmetic producing 8-bit integer "
           "activations instead of float activations --- which we call "
           "\'quantized inference\'."),
      Flag("drop_fake_quant", parsed_flags.drop_fake_quant.bind(),
           parsed_flags.drop_fake_quant.default_value(),
           "Ignore and discard FakeQuant nodes. For instance, that can be used "
           "to "
           "generate plain float code without fake-quantization from a "
           "quantized "
           "graph."),
      Flag(
          "reorder_across_fake_quant",
          parsed_flags.reorder_across_fake_quant.bind(),
          parsed_flags.reorder_across_fake_quant.default_value(),
          "Normally, FakeQuant nodes must be strict boundaries for graph "
          "transformations, in order to ensure that quantized inference has "
          "the "
          "exact same arithmetic behavior as quantized training --- which is "
          "the "
          "whole point of quantized training and of FakeQuant nodes in the "
          "first "
          "place. However, that entails subtle requirements on where exactly "
          "FakeQuant nodes must be placed in the graph. Some quantized graphs "
          "have FakeQuant nodes at unexpected locations, that prevent graph "
          "transformations that are necessary in order to generate inference "
          "code for these graphs. Such graphs should be fixed, but as a "
          "temporary work-around, setting this reorder_across_fake_quant flag "
          "allows toco to perform necessary graph transformaitons on them, "
          "at the cost of no longer faithfully matching inference and training "
          "arithmetic."),
      Flag("allow_custom_ops", parsed_flags.allow_custom_ops.bind(),
           parsed_flags.allow_custom_ops.default_value(),
           "If true, allow TOCO to create TF Lite Custom operators for all the"
           "unsupported Tensorflow ops."),
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

void ReadTocoFlagsFromCommandLineFlags(const ParsedTocoFlags& parsed_toco_flags,
                                       TocoFlags* toco_flags) {
  namespace port = toco::port;
  port::CheckInitGoogleIsDone("InitGoogle is not done yet");

  enum class FlagRequirement { kNone, kMustBeSpecified, kMustNotBeSpecified };

#define ENFORCE_FLAG_REQUIREMENT(name, requirement)                          \
  do {                                                                       \
    if (requirement == FlagRequirement::kMustBeSpecified) {                  \
      QCHECK(parsed_toco_flags.name.specified())                             \
          << "Missing required flag: " << #name;                             \
    }                                                                        \
    if (requirement == FlagRequirement::kMustNotBeSpecified) {               \
      QCHECK(!parsed_toco_flags.name.specified())                            \
          << "Given other flags, this flag should not have been specified: " \
          << #name;                                                          \
    }                                                                        \
  } while (false)

#define READ_TOCO_FLAG(name, requirement)                     \
  ENFORCE_FLAG_REQUIREMENT(name, requirement);                \
  do {                                                        \
    if (parsed_toco_flags.name.specified()) {                 \
      toco_flags->set_##name(parsed_toco_flags.name.value()); \
    }                                                         \
  } while (false)

#define PARSE_TOCO_FLAG(Type, name, requirement)               \
  ENFORCE_FLAG_REQUIREMENT(name, requirement);                 \
  do {                                                         \
    if (parsed_toco_flags.name.specified()) {                  \
      Type x;                                                  \
      QCHECK(Type##_Parse(parsed_toco_flags.name.value(), &x)) \
          << "Unrecognized " << #Type << " value "             \
          << parsed_toco_flags.name.value();                   \
      toco_flags->set_##name(x);                               \
    }                                                          \
  } while (false)

  PARSE_TOCO_FLAG(FileFormat, input_format, FlagRequirement::kMustBeSpecified);
  PARSE_TOCO_FLAG(FileFormat, output_format, FlagRequirement::kMustBeSpecified);
  FlagRequirement tflite_flags_requirement =
      toco_flags->output_format() == TFLITE
          ? FlagRequirement::kMustBeSpecified
          : FlagRequirement::kMustNotBeSpecified;
  PARSE_TOCO_FLAG(IODataType, inference_type, tflite_flags_requirement);
  READ_TOCO_FLAG(default_ranges_min, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_ranges_max, FlagRequirement::kNone);
  READ_TOCO_FLAG(drop_fake_quant, FlagRequirement::kNone);
  READ_TOCO_FLAG(reorder_across_fake_quant, FlagRequirement::kNone);
  READ_TOCO_FLAG(allow_custom_ops, FlagRequirement::kNone);

#undef READ_TOCO_FLAG
#undef PARSE_TOCO_FLAG

  const bool input_type_specified = parsed_toco_flags.input_type.specified();
  const bool input_types_specified = parsed_toco_flags.input_types.specified();
  if (toco_flags->output_format() == TFLITE) {
    QCHECK(input_type_specified || input_types_specified)
        << "When output_format=TFLITE, either input_type or input_types needs "
           "to be specified.";
  } else {
    QCHECK(!input_type_specified && !input_types_specified)
        << "With this output_format, neither input_type nor input_types must "
           "be specified.";
  }
  QCHECK(!(input_type_specified && input_types_specified))
      << "input_type and input_types are mutually exclusive";
  if (input_type_specified) {
    IODataType type;
    QCHECK(IODataType_Parse(parsed_toco_flags.input_type.value(), &type))
        << "Unrecognized input_type: " << parsed_toco_flags.input_type.value();
    toco_flags->add_input_types(type);
  }
  if (input_types_specified) {
    std::vector<string> input_types =
        absl::StrSplit(parsed_toco_flags.input_types.value(), ',');
    for (const string& t : input_types) {
      IODataType type;
      QCHECK(IODataType_Parse(t, &type))
          << "Unrecognized input_types value " << t
          << " in input_types=" << parsed_toco_flags.input_types.value();
      toco_flags->add_input_types(type);
    }
  }
}
}  // namespace toco
