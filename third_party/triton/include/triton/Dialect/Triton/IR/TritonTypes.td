#ifndef TRITON_TYPES
#define TRITON_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"

//
// Types
//
class TritonTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<Triton_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

// Floating-point Type
def TT_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def TT_FloatTensor : RankedTensorOf<[TT_Float]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;

// Boolean Type
// TT_Bool -> I1
def TT_BoolTensor : RankedTensorOf<[I1]>;
def TT_BoolLike : AnyTypeOf<[I1, TT_BoolTensor]>;

// Integer Type
def I4 : I<4>;
def TT_Int : AnyTypeOf<[I1, I4, I8, I16, I32, I64], "integer">;
def TT_IntTensor : RankedTensorOf<[TT_Int]>;
def TT_IntLike : AnyTypeOf<[TT_Int, TT_IntTensor]>;

// I32 Type
// TT_I32 -> I32
// TT_I32Tensor -> I32Tensor
def TT_I32Like : AnyTypeOf<[I32, I32Tensor]>;

// I64 Type
// TT_I64 -> I64
// TT_I64Tensor -> I64Tensor
def TT_I64Like : AnyTypeOf<[I64, I64Tensor]>;

// Pointer Type in TableGen
class TT_PtrOf<list<Type> pointeeTypes> :
    DialectType<Triton_Dialect,
                And<[CPred<"::mlir::isa<::mlir::triton::PointerType>($_self)">,
                     Concat<"[](::mlir::Type pointeeType) { return ",
                            SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                                        "; }(::mlir::cast<::mlir::triton::PointerType>($_self).getPointeeType())">]>,
                "ptr", "::mlir::triton::PointerType">;

// Pointer Type in C++ (corresponding to `TT_PtrOf`)
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
    let summary = "Pointer type (`::mlir::triton::PointerType`) in Triton IR type system";

    let description = [{
        Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
    }];

    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

// Scalar Pointer Type: `ptr<>`
def TT_Ptr : TT_PtrOf<[AnyType]>;

// Tensor of Pointer Type: `tensor<ptr<>>`
def TT_PtrTensor : RankedTensorOf<[TT_Ptr]>;

// Tensor of Pointer Type or Pointer type: `tensor<ptr<>>` or `ptr<>`
def TT_PtrLike : AnyTypeOf<[TT_Ptr, TT_PtrTensor]>;

// Tensor Type
def TT_FpIntTensor : RankedTensorOf<[TT_Float, TT_Int]>;
def TT_Tensor : RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>;

// Pointer Type to Tensor Type: `ptr<tensor<>>`
def TT_TensorPtr : TT_PtrOf<[TT_Tensor]>;

// Any Type in Triton IR
def TT_Type : AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike, TT_TensorPtr]>;

// Result type of ExperimentalMakeTensorDescriptor
def TT_TensorDescType : TritonTypeDef<"TensorDesc", "tensordesc", []> {
  let summary = "Tensor descriptor type (`::mlir::triton::TensorDescType`) in Triton IR type system";

  let description = [{
      A portable abstraction for nvidia-TMA descriptors.
  }];

  let parameters = (ins "RankedTensorType":$blockType);
  let assemblyFormat = "`<` $blockType `>`";
}

#endif
