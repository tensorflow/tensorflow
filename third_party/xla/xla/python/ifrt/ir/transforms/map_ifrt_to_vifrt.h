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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_MAP_IFRT_TO_VIFRT_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_MAP_IFRT_TO_VIFRT_H_

#include <type_traits>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"  // IWYU pragma: export

namespace xla {
namespace ifrt {

// Templates used to be able to define templated op converters for
// IFRT <--> VIFRT.
template <typename VifrtOpTy>
struct VifrtToIfrtOpImpl {
  using Type = std::false_type;
};
template <typename VifrtOpTy>
using VifrtToIfrtOp = typename VifrtToIfrtOpImpl<VifrtOpTy>::Type;

template <typename IfrtOpTy>
struct IfrtToVifrtOpImpl {
  using Type = std::false_type;
};
template <typename IfrtOpTy>
using IfrtToVifrtOp = typename IfrtToVifrtOpImpl<IfrtOpTy>::Type;

#define MAP_IFRT_TO_VIFRT(OpName, OpVer)               \
  template <>                                          \
  struct IfrtToVifrtOpImpl<xla::ifrt::OpName> {        \
    using Type = xla::ifrt::OpName##OpVer;             \
  };                                                   \
  template <>                                          \
  struct VifrtToIfrtOpImpl<xla::ifrt::OpName##OpVer> { \
    using Type = xla::ifrt::OpName;                    \
  };

// Mappings between IFRT and current VIFRT ops.
MAP_IFRT_TO_VIFRT(CallOp, V1)
MAP_IFRT_TO_VIFRT(ReshardOp, V1)
MAP_IFRT_TO_VIFRT(CopyArraysOp, V1)
MAP_IFRT_TO_VIFRT(AssembleOp, V1)
MAP_IFRT_TO_VIFRT(DisassembleOp, V1)
MAP_IFRT_TO_VIFRT(RemapArraysOp, V1)
MAP_IFRT_TO_VIFRT(AfterOp, V1)
MAP_IFRT_TO_VIFRT(CallLoadedExecutableOp, V1)
MAP_IFRT_TO_VIFRT(LoadedExecutableOp, V1)

#undef MAP_IFRT_TO_VIFRT

// Mappings from dialects other than IFRT.
#define MAP_OTHER_TO_VIFRT(UpstreamOpName, VifrtOpName, OpVer) \
  template <>                                                  \
  struct IfrtToVifrtOpImpl<UpstreamOpName> {                   \
    using Type = VifrtOpName##OpVer;                           \
  };                                                           \
  template <>                                                  \
  struct VifrtToIfrtOpImpl<VifrtOpName##OpVer> {               \
    using Type = UpstreamOpName;                               \
  };

MAP_OTHER_TO_VIFRT(::mlir::func::FuncOp, ::xla::ifrt::FuncOp, V1)
MAP_OTHER_TO_VIFRT(::mlir::func::CallOp, ::xla::ifrt::CallFuncOp, V1)
MAP_OTHER_TO_VIFRT(::mlir::func::ReturnOp, ::xla::ifrt::ReturnOp, V1)

#undef MAP_OTHER_TO_VIFRT

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_MAP_IFRT_TO_VIFRT_H_
