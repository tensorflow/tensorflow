#ifndef TRITON_CONVERSION_PASSES
#define TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPU: Pass<"convert-triton-to-tritongpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU";
    let description = [{
      This pass converts the Triton Dialect into the TritonGPU Dialect.
      This is a partial conversion that also affects other dialects
      (namely `Arith`, `Math`, `SCF` and `CF`).
      For these dialects, and many Triton dialect operations the conversions
      mainly consists of enhancing the tensor type and the `tt.ptr<tensor<>>`
      type with an appropriate layout encoding (these encodings generally
      include information on `numWarps`, `threadsPerWarp` and `numCTAs`).
    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">,

       Option<"threadsPerWarp", "threads-per-warp",
              "int32_t", /*default*/"32",
              "number of threads per warp">,
        Option<"numCTAs", "num-ctas",
              "int32_t", /*default*/"1",
              "number of ctas in a cga">,
        Option<"target", "target",
              "std::string", /*default*/"\"\"",
              "the GPU target, e.g., cuda:80, hip:gfx942">
   ];
}

#endif
