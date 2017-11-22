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
#include "tensorflow/contrib/lite/toco/model_cmdline_flags.h"

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/contrib/lite/toco/args.h"
#include "tensorflow/contrib/lite/toco/toco_graphviz_dump_options.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
// "batch" flag only exists internally
#ifdef PLATFORM_GOOGLE
#include "base/commandlineflags.h"
#endif

namespace toco {

bool ParseModelFlagsFromCommandLineFlags(
    int* argc, char* argv[], string* msg,
    ParsedModelFlags* parsed_model_flags_ptr) {
  ParsedModelFlags& parsed_flags = *parsed_model_flags_ptr;
  using tensorflow::Flag;
  std::vector<tensorflow::Flag> flags = {
      Flag("input_array", parsed_flags.input_array.bind(),
           parsed_flags.input_array.default_value(),
           "Deprecated: use --input_arrays instead. Name of the input array. "
           "If not specified, will try to read "
           "that information from the input file."),
      Flag("input_arrays", parsed_flags.input_arrays.bind(),
           parsed_flags.input_arrays.default_value(),
           "Names of the output arrays, comma-separated. If not specified, "
           "will try to read that information from the input file."),
      Flag("output_array", parsed_flags.output_array.bind(),
           parsed_flags.output_array.default_value(),
           "Deprecated: use --output_arrays instead. Name of the output array, "
           "when specifying a unique output array. "
           "If not specified, will try to read that information from the "
           "input file."),
      Flag("output_arrays", parsed_flags.output_arrays.bind(),
           parsed_flags.output_arrays.default_value(),
           "Names of the output arrays, comma-separated. "
           "If not specified, will try to read "
           "that information from the input file."),
      Flag("input_shape", parsed_flags.input_shape.bind(),
           parsed_flags.input_shape.default_value(),
           "Deprecated: use --input_shapes instead. Input array shape. For "
           "many models the shape takes the form "
           "batch size, input array height, input array width, input array "
           "depth."),
      Flag("input_shapes", parsed_flags.input_shapes.bind(),
           parsed_flags.input_shapes.default_value(),
           "Shapes corresponding to --input_arrays, colon-separated. For "
           "many models each shape takes the form batch size, input array "
           "height, input array width, input array depth."),
      Flag("input_data_type", parsed_flags.input_data_type.bind(),
           parsed_flags.input_data_type.default_value(),
           "Deprecated: use --input_data_types instead. Input array type, if "
           "not already provided in the graph. "
           "Typically needs to be specified when passing arbitrary arrays "
           "to --input_array."),
      Flag("input_data_types", parsed_flags.input_data_types.bind(),
           parsed_flags.input_data_types.default_value(),
           "Input arrays types, comma-separated, if not already provided in "
           "the graph. "
           "Typically needs to be specified when passing arbitrary arrays "
           "to --input_arrays."),
      Flag("mean_value", parsed_flags.mean_value.bind(),
           parsed_flags.mean_value.default_value(),
           "Deprecated: use --mean_values instead. mean_value parameter for "
           "image models, used to compute input "
           "activations from input pixel data."),
      Flag("mean_values", parsed_flags.mean_values.bind(),
           parsed_flags.mean_values.default_value(),
           "mean_values parameter for image models, comma-separated list of "
           "doubles, used to compute input activations from input pixel "
           "data. Each entry in the list should match an entry in "
           "--input_arrays."),
      Flag("std_value", parsed_flags.std_value.bind(),
           parsed_flags.std_value.default_value(),
           "Deprecated: use --std_values instead. std_value parameter for "
           "image models, used to compute input "
           "activations from input pixel data."),
      Flag("std_values", parsed_flags.std_values.bind(),
           parsed_flags.std_values.default_value(),
           "std_value parameter for image models, comma-separated list of "
           "doubles, used to compute input activations from input pixel "
           "data. Each entry in the list should match an entry in "
           "--input_arrays."),
      Flag("variable_batch", parsed_flags.variable_batch.bind(),
           parsed_flags.variable_batch.default_value(),
           "If true, the model accepts an arbitrary batch size. Mutually "
           "exclusive "
           "with the 'batch' field: at most one of these two fields can be "
           "set."),
      Flag("rnn_states", parsed_flags.rnn_states.bind(),
           parsed_flags.rnn_states.default_value(), ""),
      Flag("model_checks", parsed_flags.model_checks.bind(),
           parsed_flags.model_checks.default_value(),
           "A list of model checks to be applied to verify the form of the "
           "model.  Applied after the graph transformations after import."),
      Flag("graphviz_first_array", parsed_flags.graphviz_first_array.bind(),
           parsed_flags.graphviz_first_array.default_value(),
           "If set, defines the start of the sub-graph to be dumped to "
           "GraphViz."),
      Flag(
          "graphviz_last_array", parsed_flags.graphviz_last_array.bind(),
          parsed_flags.graphviz_last_array.default_value(),
          "If set, defines the end of the sub-graph to be dumped to GraphViz."),
      Flag("dump_graphviz", parsed_flags.dump_graphviz.bind(),
           parsed_flags.dump_graphviz.default_value(),
           "Dump graphviz during LogDump call. If string is non-empty then "
           "it defines path to dump, otherwise will skip dumping."),
      Flag("dump_graphviz_video", parsed_flags.dump_graphviz_video.bind(),
           parsed_flags.dump_graphviz_video.default_value(),
           "If true, will dump graphviz at each "
           "graph transformation, which may be used to generate a video."),
  };
  bool asked_for_help =
      *argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-help"));
  if (asked_for_help) {
    *msg += tensorflow::Flags::Usage(argv[0], flags);
    return false;
  } else {
    if (!tensorflow::Flags::Parse(argc, argv, flags)) return false;
  }
  auto& dump_options = *GraphVizDumpOptions::singleton();
  dump_options.graphviz_first_array = parsed_flags.graphviz_first_array.value();
  dump_options.graphviz_last_array = parsed_flags.graphviz_last_array.value();
  dump_options.dump_graphviz_video = parsed_flags.dump_graphviz_video.value();
  dump_options.dump_graphviz = parsed_flags.dump_graphviz.value();

  return true;
}

void ReadModelFlagsFromCommandLineFlags(
    const ParsedModelFlags& parsed_model_flags, ModelFlags* model_flags) {
  toco::port::CheckInitGoogleIsDone("InitGoogle is not done yet");

// "batch" flag only exists internally
#ifdef PLATFORM_GOOGLE
  CHECK(!((base::SpecifiedOnCommandLine("batch") &&
           parsed_model_flags.variable_batch.specified())))
      << "The --batch and --variable_batch flags are mutually exclusive.";
#endif
  CHECK(!(parsed_model_flags.output_array.specified() &&
          parsed_model_flags.output_arrays.specified()))
      << "The --output_array and --vs flags are mutually exclusive.";

  if (parsed_model_flags.output_array.specified()) {
    model_flags->add_output_arrays(parsed_model_flags.output_array.value());
  }

  if (parsed_model_flags.output_arrays.specified()) {
    std::vector<string> output_arrays =
        absl::StrSplit(parsed_model_flags.output_arrays.value(), ',');
    for (const string& output_array : output_arrays) {
      model_flags->add_output_arrays(output_array);
    }
  }

  const bool uses_single_input_flags =
      parsed_model_flags.input_array.specified() ||
      parsed_model_flags.mean_value.specified() ||
      parsed_model_flags.std_value.specified() ||
      parsed_model_flags.input_shape.specified();

  const bool uses_multi_input_flags =
      parsed_model_flags.input_arrays.specified() ||
      parsed_model_flags.mean_values.specified() ||
      parsed_model_flags.std_values.specified() ||
      parsed_model_flags.input_shapes.specified();

  QCHECK(!(uses_single_input_flags && uses_multi_input_flags))
      << "Use either the singular-form input flags (--input_array, "
         "--input_shape, --mean_value, --std_value) or the plural form input "
         "flags (--input_arrays, --input_shapes, --mean_values, --std_values), "
         "but not both forms within the same command line.";

  if (parsed_model_flags.input_array.specified()) {
    QCHECK(uses_single_input_flags);
    model_flags->add_input_arrays()->set_name(
        parsed_model_flags.input_array.value());
  }
  if (parsed_model_flags.input_arrays.specified()) {
    QCHECK(uses_multi_input_flags);
    for (const auto& input_array :
         absl::StrSplit(parsed_model_flags.input_arrays.value(), ',')) {
      model_flags->add_input_arrays()->set_name(string(input_array));
    }
  }
  if (parsed_model_flags.mean_value.specified()) {
    QCHECK(uses_single_input_flags);
    model_flags->mutable_input_arrays(0)->set_mean_value(
        parsed_model_flags.mean_value.value());
  }
  if (parsed_model_flags.mean_values.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<string> mean_values =
        absl::StrSplit(parsed_model_flags.mean_values.value(), ',');
    QCHECK(mean_values.size() == model_flags->input_arrays_size());
    for (int i = 0; i < mean_values.size(); ++i) {
      char* last = nullptr;
      model_flags->mutable_input_arrays(i)->set_mean_value(
          strtod(mean_values[i].data(), &last));
      CHECK(last != mean_values[i].data());
    }
  }
  if (parsed_model_flags.std_value.specified()) {
    QCHECK(uses_single_input_flags);
    model_flags->mutable_input_arrays(0)->set_std_value(
        parsed_model_flags.std_value.value());
  }
  if (parsed_model_flags.std_values.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<string> std_values =
        absl::StrSplit(parsed_model_flags.std_values.value(), ',');
    QCHECK(std_values.size() == model_flags->input_arrays_size());
    for (int i = 0; i < std_values.size(); ++i) {
      char* last = nullptr;
      model_flags->mutable_input_arrays(i)->set_std_value(
          strtod(std_values[i].data(), &last));
      CHECK(last != std_values[i].data());
    }
  }
  if (parsed_model_flags.input_data_type.specified()) {
    QCHECK(uses_single_input_flags);
    IODataType type;
    QCHECK(IODataType_Parse(parsed_model_flags.input_data_type.value(), &type));
    model_flags->mutable_input_arrays(0)->set_data_type(type);
  }
  if (parsed_model_flags.input_data_types.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<string> input_data_types =
        absl::StrSplit(parsed_model_flags.input_data_types.value(), ',');
    QCHECK(input_data_types.size() == model_flags->input_arrays_size());
    for (int i = 0; i < input_data_types.size(); ++i) {
      IODataType type;
      QCHECK(IODataType_Parse(input_data_types[i], &type));
      model_flags->mutable_input_arrays(i)->set_data_type(type);
    }
  }
  if (parsed_model_flags.input_shape.specified()) {
    QCHECK(uses_single_input_flags);
    if (model_flags->input_arrays().empty()) {
      model_flags->add_input_arrays();
    }
    auto* shape = model_flags->mutable_input_arrays(0)->mutable_shape();
    shape->Clear();
    const IntList& list = parsed_model_flags.input_shape.value();
    for (auto& dim : list.elements) {
      shape->Add(dim);
    }
  }
  if (parsed_model_flags.input_shapes.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<string> input_shapes =
        absl::StrSplit(parsed_model_flags.input_shapes.value(), ':');
    QCHECK(input_shapes.size() == model_flags->input_arrays_size());
    for (int i = 0; i < input_shapes.size(); ++i) {
      auto* shape = model_flags->mutable_input_arrays(i)->mutable_shape();
      shape->Clear();
      if (input_shapes[i].empty()) {
        // empty i.e. 0-dimensional input shape.
        // Unfortunately, the current toco::InputArray
        // proto does not allow to distinguish between a known 0-D shape,
        // and an unknown shape. Indeed, shape is currently a plain array,
        // and it being empty means unknown shape. So here, we import a
        // 0-D shape as a 1-D shape of size.
        // TODO(benoitjacob): fix toco::InputArray to allow 0-D shape,
        // probably by making shape an optional message,
        // encapsulating the array.
        shape->Add(1);
      } else {
        for (const auto& dim_str : absl::StrSplit(input_shapes[i], ',')) {
          int size;
          CHECK(absl::SimpleAtoi(dim_str, &size))
              << "Failed to parse input_shape: " << input_shapes[i];
          shape->Add(size);
        }
      }
    }
  }

#define READ_MODEL_FLAG(name)                                   \
  do {                                                          \
    if (parsed_model_flags.name.specified()) {                  \
      model_flags->set_##name(parsed_model_flags.name.value()); \
    }                                                           \
  } while (false)

  READ_MODEL_FLAG(variable_batch);

#undef READ_MODEL_FLAG

  for (const auto& element : parsed_model_flags.rnn_states.value().elements) {
    auto* rnn_state_proto = model_flags->add_rnn_states();
    for (const auto& kv_pair : element) {
      const string& key = kv_pair.first;
      const string& value = kv_pair.second;
      if (key == "state_array") {
        rnn_state_proto->set_state_array(value);
      } else if (key == "back_edge_source_array") {
        rnn_state_proto->set_back_edge_source_array(value);
      } else if (key == "size") {
        int32 size = 0;
        CHECK(absl::SimpleAtoi(value, &size));
        CHECK_GT(size, 0);
        rnn_state_proto->set_size(size);
      } else if (key == "manually_create") {
        CHECK_EQ(absl::AsciiStrToLower(value), "true");
        rnn_state_proto->set_manually_create(true);
      } else {
        LOG(FATAL) << "Unknown key '" << key << "' in --rnn_states";
      }
    }
    CHECK(rnn_state_proto->has_state_array() &&
          rnn_state_proto->has_back_edge_source_array() &&
          rnn_state_proto->has_size())
        << "--rnn_states must include state_array, back_edge_source_array and "
           "size.";
  }

  for (const auto& element : parsed_model_flags.model_checks.value().elements) {
    auto* model_check_proto = model_flags->add_model_checks();
    for (const auto& kv_pair : element) {
      const string& key = kv_pair.first;
      const string& value = kv_pair.second;
      if (key == "count_type") {
        model_check_proto->set_count_type(value);
      } else if (key == "count_min") {
        int32 count = 0;
        CHECK(absl::SimpleAtoi(value, &count));
        CHECK_GE(count, -1);
        model_check_proto->set_count_min(count);
      } else if (key == "count_max") {
        int32 count = 0;
        CHECK(absl::SimpleAtoi(value, &count));
        CHECK_GE(count, -1);
        model_check_proto->set_count_max(count);
      } else {
        LOG(FATAL) << "Unknown key '" << key << "' in --model_checks";
      }
    }
  }
}

ParsedModelFlags* UncheckedGlobalParsedModelFlags(bool must_already_exist) {
  static auto* flags = [must_already_exist]() {
    if (must_already_exist) {
      fprintf(stderr, __FILE__
              ":"
              "GlobalParsedModelFlags() used without initialization\n");
      fflush(stderr);
      abort();
    }
    return new toco::ParsedModelFlags;
  }();
  return flags;
}

ParsedModelFlags* GlobalParsedModelFlags() {
  return UncheckedGlobalParsedModelFlags(true);
}

void ParseModelFlagsOrDie(int* argc, char* argv[]) {
  // TODO(aselle): in the future allow Google version to use
  // flags, and only use this mechanism for open source
  auto* flags = UncheckedGlobalParsedModelFlags(false);
  string msg;
  bool model_success =
      toco::ParseModelFlagsFromCommandLineFlags(argc, argv, &msg, flags);
  if (!model_success || !msg.empty()) {
    // Log in non-standard way since this happens pre InitGoogle.
    fprintf(stderr, "%s", msg.c_str());
    fflush(stderr);
    abort();
  }
}

}  // namespace toco
