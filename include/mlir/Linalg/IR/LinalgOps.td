//===- LinalgOps.td - Linalg dialect ops -------------------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This is the operation definition file for linear algebra operations.
//
//===----------------------------------------------------------------------===//

#ifdef LINALG_OPS
#else

#ifdef LINALG_BASE
#else
include "mlir/Linalg/IR/LinalgBase.td"
#endif // LINALG_BASE

def BufferSizeOp :
    Op<Linalg_Dialect, "buffer_size", [NoSideEffect]>,
    Arguments<(ins Buffer)>,
    Results<(outs Index)>
{
  let parser = [{ return parseBufferSizeOp(parser, result); }];
  let printer = [{ return printBufferSizeOp(p, *this); }];
}

def DimOp : Op<Linalg_Dialect, "dim", [NoSideEffect]>,
    Arguments<(ins View:$view, APIntAttr:$index)>,
    Results<(outs Index)> {
  let summary = "dimension index operation";
  let description = [{
    The "linalg.dim" operation takes a linalg.view and returns an
    "index". It requires a single integer attribute named "index". It
     returns the size of the specified dimension. For example:

      %1 = linalg.dim %0, 2 : view<?x?x?xf32>
  }];

  let parser = [{ return parseDimOp(parser, result); }];
  let printer = [{ return printDimOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *view," "unsigned index",
    [{
      result->addOperands(view);
      result->addAttribute(
        "index", builder->getIntegerAttr(builder->getIndexType(), index));
      result->types.push_back(builder->getIndexType());
    }]>];

  let extraClassDeclaration = [{
    unsigned getIndex() {
      return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    }
  }];
}

#endif // LINALG_OPS