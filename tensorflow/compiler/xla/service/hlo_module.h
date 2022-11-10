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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/dynamic_parameter_binding.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_clone_context.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module_metadata.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/service/compilation_environments.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/lib/gtl/iterator_range.h"
#include "tensorflow/tsl/platform/logging.h"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
