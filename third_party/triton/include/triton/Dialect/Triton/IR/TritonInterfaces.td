#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def VerifyTensorLayoutsTrait : NativeOpTrait<"VerifyTensorLayoutsTrait">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameLoadStoreOperandsShape : NativeOpTrait<"SameLoadStoreOperandsShape">;
def SameLoadStoreOperandsAndResultShape : NativeOpTrait<"SameLoadStoreOperandsAndResultShape">;
def SameLoadStoreOperandsEncoding : NativeOpTrait<"SameLoadStoreOperandsEncoding">;
def SameLoadStoreOperandsAndResultEncoding : NativeOpTrait<"SameLoadStoreOperandsAndResultEncoding">;
def AsyncRegions : NativeOpTrait<"AsyncRegions">;

// A trait equivalent to InferTypeOpAdaptor, but that checks for structural
// equivalence of the layouts of the result rather than just layout equality.
def InferTypeOpWithLayoutEquivalence : InferTypeOpAdaptorBase<[{
  static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
    if (lhs.size() != rhs.size())
      return false;
    return llvm::all_of(llvm::zip(lhs, rhs), [](auto tup) {
      auto [lhs, rhs] = tup;
      return succeeded(OpTrait::impl::verifyEquivalentType(lhs, rhs));
    });
  }
}]>;

#endif // TRITON_INTERFACES
