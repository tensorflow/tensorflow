#ifndef TRITONGPU_TYPES
#define TRITONGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"

class TTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def TTG_AsyncToken : TTG_TypeDef<"AsyncToken", "async.token", []> {
  let summary = "async token type";
  let description = [{
    `ttg.async.token` is a type returned by an asynchronous operation.
    It is used to establish an SSA-based link between async operations
    and operations that group or synchronize the async operations.
  }];
}

// Memory descriptor type.
def TTG_MemDescType : TTG_TypeDef<"MemDesc", "memdesc", [ShapedTypeInterface]> {
    let summary = "memory descriptor type (`::mlir::triton::gpu::MemDescType`) in Triton IR type system";

    let description = [{
        Memory descriptor contains a base pointer (scalar) and a descriptor of the memory.
        If mutable memory is false that means the memory is constant and can only be allocated and stored once.
        A constant memory allocation is different than a tensor as it can have multiple views and the descriptor
        can be changed without changing the underlying memory.
    }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Attribute":$encoding,
    "Attribute":$memorySpace,
    "bool":$mutableMemory,
    ArrayRefParameter<"int64_t">:$allocShape
  );

  let extraClassDeclaration = [{
    MemDescType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                          Type elementType) const {
      return MemDescType::get(shape.value_or(getShape()), elementType, getEncoding(), getMemorySpace(), getMutableMemory(), getAllocShape());
    }

    bool hasRank() const { return true; }
  }];

  let builders = [
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "Attribute":$encoding,
            "Attribute":$memorySpace
        ), [{
            return $_get(elementType.getContext(), shape, elementType, encoding, memorySpace, /*mutableMemory=*/false, /*allocShape=*/shape);
        }]>,
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "Attribute":$encoding,
            "Attribute":$memorySpace,
            "bool":$mutableMemory
        ), [{
            return $_get(elementType.getContext(), shape, elementType, encoding, memorySpace, mutableMemory, /*allocShape=*/shape);
        }]>,
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "Attribute":$encoding,
            "Attribute":$memorySpace,
            "bool":$mutableMemory,
            "llvm::ArrayRef<int64_t>":$allocShape
        ), [{
            return $_get(elementType.getContext(), shape, elementType, encoding, memorySpace, mutableMemory, allocShape);
        }]>

    ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


#endif
