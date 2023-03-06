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

// This is the operation definition file for ST ops.

#ifndef GML_ST_OPS
#define GML_ST_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "gml_st/IR/gml_st_ops_base.td"

///////////////////////////////////////////////////////////////////////////////
// Ops
///////////////////////////////////////////////////////////////////////////////

def GMLST_FusionOp : GMLST_Op<"fusion", [
      DestinationStyleOpInterface,
      SingleBlockImplicitTerminator<"gml_st::YieldOp">
    ]> {
  let summary = "A cluster of operations to be tiled and fused.";

  let arguments = (ins Variadic<AnyType>:$inputs,
                       OptionalAttr<I64ArrayAttr>:$parallel_tile_sizes,
                       OptionalAttr<I64ArrayAttr>:$reduction_tile_sizes);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  code extraClassDeclaration = [{
    /// Return terminator of the region body.
    YieldOp getTerminator();

    // Implement method necessary for DestinationStyleOpInterface.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t numOperands = this->getNumOperands();
      int64_t numResults = this->getNumResults();
      return {numOperands - numResults, numOperands};
    }
  }];
}

def GMLST_YieldOp : GMLST_Op<"yield", [Pure, ReturnLike, Terminator,
    HasParent<"::mlir::gml_st::FusionOp">]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Yield operation";
  let description = [{
    `gml_st.yield` is a special terminator operation for accumulator regions of
    `gml_st.set_yield` and `gml_st.fusion` region.

    Example:

    ```mlir
    gml_st.yield %f0: tensor<f32>
    ```
  }];
  let assemblyFormat = "attr-dict $values `:` type($values)";
}

#endif // GML_ST_OPS
