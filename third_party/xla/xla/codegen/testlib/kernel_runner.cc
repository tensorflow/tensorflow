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

#include "xla/codegen/testlib/kernel_runner.h"

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {

absl::Status KernelRunner::Call(absl::Span<Literal*> literals) {
  std::vector<KernelRunner::Argument> arguments;
  arguments.reserve(literals.size());
  for (Literal* literal : literals) {
    if (literal == nullptr) {
      return InvalidArgument("Literal is null");
    }

    arguments.push_back(KernelRunnerUtil::CreateArgument(*literal));
  }
  return Call(arguments);
}

KernelRunner::Argument KernelRunnerUtil::CreateArgument(
    Literal& literal, const ShapeIndex& index) {
  const Shape& shape = ShapeUtil::GetSubshape(literal.shape(), index);
  PrimitiveType element_type = shape.element_type();

  return primitive_util::PrimitiveTypeSwitch<KernelRunner::Argument>(
      [&](auto type) {
        if constexpr (primitive_util::IsArrayType(type)) {
          using T = typename primitive_util::PrimitiveTypeToNative<type>::type;
          absl::Span<T> data = literal.data<T>(index);
          return KernelRunner::Argument{
              reinterpret_cast<std::byte*>(data.data()),
              data.size() * sizeof(T)};
        }
        LOG(FATAL) << "Unsupported primitive type: " << type;
        return KernelRunner::Argument();
      },
      element_type);
}

}  // namespace xla
