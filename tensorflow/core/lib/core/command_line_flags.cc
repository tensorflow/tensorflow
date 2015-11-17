#include "tensorflow/core/lib/core/command_line_flags.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace {

// Templated function to convert a string to target values.
// Return true if the conversion is successful. Otherwise, return false.
template <typename T>
bool StringToValue(const string& content, T* value);

template <>
bool StringToValue<int32>(const string& content, int* value) {
  return str_util::NumericParse32(content, value);
}

template <>
bool StringToValue<string>(const string& content, string* value) {
  *value = content;
  return true;
}

// Parse a single argument by linearly searching through the command table.
// The input format is: --argument=value.
// Return OK if the argument is used. It store the extracted value into the
// matching flag.
// Return NOT_FOUND if the argument is not recognized.
// Retrun INVALID_ARGUMENT if the command is recognized, but fails to extract
// its value.
template <typename T>
Status ParseArgument(const string& argument) {
  for (auto& command :
       internal::CommandLineFlagRegistry<T>::Instance()->commands) {
    string prefix = strings::StrCat("--", command.name, "=");
    if (tensorflow::StringPiece(argument).starts_with(prefix)) {
      string content = argument.substr(prefix.length());
      if (StringToValue<T>(content, command.value)) {
        return Status::OK();
      }
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("Cannot parse integer in: ", argument));
    }
  }
  return Status(error::NOT_FOUND,
                strings::StrCat("Unknown command: ", argument));
}

// A specialization for booleans. The input format is:
//   "--argument" or "--noargument".
// Parse a single argument by linearly searching through the command table.
// Return OK if the argument is used. The value is stored in the matching flag.
// Return NOT_FOUND if the argument is not recognized.
template <>
Status ParseArgument<bool>(const string& argument) {
  for (auto& command :
       internal::CommandLineFlagRegistry<bool>::Instance()->commands) {
    if (argument == strings::StrCat("--", command.name)) {
      *command.value = true;
      return Status::OK();
    } else if (argument == strings::StrCat("--no", command.name)) {
      *command.value = false;
      return Status::OK();
    }
  }
  return Status(error::NOT_FOUND,
                strings::StrCat("Unknown command: ", argument));
}

}  // namespace

Status ParseCommandLineFlags(int* argc, char* argv[]) {
  int unused_argc = 1;
  for (int index = 1; index < *argc; ++index) {
    Status s;
    // Search bool commands.
    s = ParseArgument<bool>(argv[index]);
    if (s.ok()) {
      continue;
    }
    if (s.code() != error::NOT_FOUND) {
      return s;
    }
    // Search int32 commands.
    s = ParseArgument<int32>(argv[index]);
    if (s.ok()) {
      continue;
    }
    // Search string commands.
    s = ParseArgument<string>(argv[index]);
    if (s.ok()) {
      continue;
    }
    if (s.code() != error::NOT_FOUND) {
      return s;
    }
    // Pointer swap the unused argument to the front.
    std::swap(argv[unused_argc++], argv[index]);
  }
  *argc = unused_argc;
  return Status::OK();
}

}  // namespace tensorflow
