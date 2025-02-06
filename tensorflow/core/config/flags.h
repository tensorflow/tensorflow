/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_CONFIG_FLAGS_H_
#define TENSORFLOW_CORE_CONFIG_FLAGS_H_

#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace config {

// Container class for a single feature flag.
// Note: this class is not thread safe.
class Flag {
 public:
  explicit Flag(absl::string_view flag_name, bool default_value);
  bool value() { return value_; }
  void reset(bool value) { value_ = value; }

 private:
  bool value_;
};

// Macro to declare new flags. Declare all flags in core/config/flag_defs.h
// These flags can be overridden by setting the associated environment variable
// TF_FLAG_* flag to true or false. E.g. setting TF_FLAG_MY_FLAG=false will
// override the default value for a flag named `my_flag` to false.
#define TF_DECLARE_FLAG(flag_name, default_value, doc) \
  ::tensorflow::config::Flag flag_name =               \
      ::tensorflow::config::Flag("TF_FLAG_" #flag_name, default_value);

}  // namespace config
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_CONFIG_FLAGS_H_
