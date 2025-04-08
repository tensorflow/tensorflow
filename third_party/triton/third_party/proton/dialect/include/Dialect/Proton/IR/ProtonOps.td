#ifndef PROTON_OPS
#define PROTON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "ProtonDialect.td"
include "ProtonAttrDefs.td"

class TT_Proton_Op<string mnemonic, list<Trait> traits = []> :
    Op<Proton_Dialect, mnemonic, !listconcat(traits, [])> {
}

// Proton profiling metric.
def MetricAttr : I32EnumAttr<
    "Metric", "",
    [
        I32EnumAttrCase<"CYCLE", 0, "cycle">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

// Proton profiling granularity.
def GranularityAttr : I32EnumAttr<
    "Granularity", "",
    [
        I32EnumAttrCase<"WARPGROUP", 0, "warpgroup">,
        I32EnumAttrCase<"WARP", 1, "warp">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

def TT_RecordOp : TT_Proton_Op<"record", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Record a GPU hardware event";

  let description = [{
    The operator records GPU events from performance counters.
    Currently only cycle counter is supported.

    Example:

    ```mlir
    proton.record() {isStart = true, regionId = 4 : i32}
    ...
    proton.record() {isStart = false, regionId = 4 : i32}
    ...
    proton.record() {isStart = true, regionId = 1 : i32, granularity = 1 : i32}
    ...
    proton.record() {isStart = false, regionId = 1 : i32, granularity = 1 : i32}
    ```
  }];
  let arguments = (
    ins BoolAttr: $isStart,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$regionId,
    DefaultValuedAttr<MetricAttr, "Metric::CYCLE">:$metric,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARPGROUP">:$granularity
  );
  let assemblyFormat = " `(` operands `)` attr-dict";
}

#endif // PROTON_OPS
