#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "ttg";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);
    LinearEncodingAttr toLinearEncoding(ArrayRef<int64_t> shape, Attribute layout);

    static int getNumCTAs(ModuleOp mod);
    static int getThreadsPerWarp(ModuleOp mod);

    private:
      LinearLayoutCache llCache;
      LinearEncodingCache leCache;
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
