/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.h"  // IWYU pragma: keep

#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace xla::gpu {

using namespace mlir;  // NOLINT

static ParseResult parseGraphDispatchRegion(OpAsmParser &parser, Region &body) {
  OpAsmParser::Argument arg;
  if (parser.parseKeyword("graph") || parser.parseLParen() ||
      parser.parseArgument(arg, /*allowType=*/true) || parser.parseRParen())
    return failure();

  return parser.parseRegion(body, /*arguments=*/{arg});
}

static void printGraphDispatchRegion(OpAsmPrinter &p, Operation *op,
                                     Region &body) {
  auto arg = body.getArgument(0);
  p << "graph"
    << "(" << arg << ": " << arg.getType() << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

}  // namespace xla::gpu

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.cc.inc"
