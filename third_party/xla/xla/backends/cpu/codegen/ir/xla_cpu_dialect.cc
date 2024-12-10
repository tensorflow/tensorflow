/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/ir/xla_cpu_dialect.h"

#include "xla/backends/cpu/codegen/ir/xla_cpu_ops.h"  // IWYU pragma: keep
#include "xla/backends/cpu/codegen/ir/xla_cpu_types.h"  // IWYU pragma: keep

// Include the auto-generated implementation file.
#include "xla/backends/cpu/codegen/ir/xla_cpu_dialect.cc.inc"

namespace xla::cpu {

void XlaCpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/backends/cpu/codegen/ir/xla_cpu_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/backends/cpu/codegen/ir/xla_cpu_types.cc.inc"
      >();
}

}  // namespace xla::cpu
