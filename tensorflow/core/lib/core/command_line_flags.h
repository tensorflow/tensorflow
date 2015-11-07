#ifndef TENSORFLOW_LIB_CORE_COMMAND_LINE_FLAGS_H_
#define TENSORFLOW_LIB_CORE_COMMAND_LINE_FLAGS_H_

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
namespace internal {

template <typename T>
struct CommandLineFlagRegistry {
  static CommandLineFlagRegistry* Instance() {
    static CommandLineFlagRegistry instance_;
    return &instance_;
  }
  struct Command {
    string name;
    T* value;
    string text;
  };
  std::vector<Command> commands;

 private:
  CommandLineFlagRegistry() {}
  TF_DISALLOW_COPY_AND_ASSIGN(CommandLineFlagRegistry);
};

template <typename T>
struct CommandLineFlagRegister {
  CommandLineFlagRegister(const string& name, T* val, const string& text) {
    CommandLineFlagRegistry<T>::Instance()->commands.push_back(
        {name, val, text});
  }
};

#define TF_DEFINE_variable(type, name, default_value, text)     \
  type FLAGS_##name = default_value;                            \
  namespace TF_flags_internal {                                 \
  tensorflow::internal::CommandLineFlagRegister<type>           \
      TF_flags_internal_var_##name(#name, &FLAGS_##name, text); \
  }  // namespace TF_flags_internal

}  // namespace internal

#define TF_DEFINE_int32(name, default_value, text) \
  TF_DEFINE_variable(int32, name, default_value, text);

#define TF_DEFINE_bool(name, default_value, text) \
  TF_DEFINE_variable(bool, name, default_value, text);

// Parse argv[1]..argv[*argc-1] to options. Remove used arguments from the argv.
// Returned the number of unused arguments in *argc.
// Return error Status if the parsing encounters errors.
// TODO(opensource): switch to a command line argument parser that can be
// shared with other tests.
Status ParseCommandLineFlags(int* argc, char* argv[]);

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_COMMAND_LINE_FLAGS_H_
