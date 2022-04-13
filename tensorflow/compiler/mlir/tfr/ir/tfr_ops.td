/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation definition file for TFR

#ifndef DIALECT_TFR_OPS_
#define DIALECT_TFR_OPS_

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def TFR_Dialect : Dialect {
  let name = "tfr";

  let description = [{
    The TensorFlow Composition dialect.
  }];

  let cppNamespace = "::mlir::TFR";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

//===----------------------------------------------------------------------===//
// Type classes
//===----------------------------------------------------------------------===//

// tensor argument types
class TFR_Type<string name> : DialectType<TFR_Dialect,
    CPred<"$_self.isa<mlir::TFR::" # name # "Type>()">,
    "TFR " # name #" type">,
    BuildableType<"$_builder.getType<mlir::TFR::" # name # "Type>()">;
def TFR_TensorType : TFR_Type<"TFRTensor">;
def TFR_TensorListType : TFR_Type<"TFRTensorList">;
def TFR_AllTensorTypes : Type<Or<[
    TFR_TensorType.predicate,
    TFR_TensorListType.predicate]>, "all tensor related types">;

// attribute argument types
def TFR_AttrType : TFR_Type<"TFRAttr">;
def TFR_AttrScalarType: TypeAlias<TF_ElementType, "scalar attribute">;
def TFR_AttrVectorType : VectorOf<[TF_ElementType, TFR_AttrType]>;
def TFR_AllAttrTypes : Type<Or<[
    TFR_AttrType.predicate,
    Index.predicate,
    TFR_AttrScalarType.predicate,
    TFR_AttrVectorType.predicate]>, "all attribute related types">;

// all allowed arguments types
def TFR_allowedArgType : Type<Or<[
    TFR_AllTensorTypes.predicate,
    TFR_AllAttrTypes.predicate]>, "allowed tfr.call operand types">;

def TFR_allowedConstValues : Attr<Or<[
    FlatSymbolRefAttr.predicate,
    TypeAttr.predicate,
    StrAttr.predicate,
    ArrayAttr.predicate]>, "allowed tfr.constant value"> {
  let storageType = "Attribute";
  let returnType = "Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

// all allowed result types
def TFR_allowedResultType : TypeAlias<TFR_AllTensorTypes,
  "allowed tfr.call result types">;

// standard tensor type and tfr.tensor types can be casted to each other.
def TFR_singleTensorType : Type<Or<[
    TFR_TensorType.predicate,
    TF_Tensor.predicate,
    TensorOf<[quant_QuantizedType]>.predicate]>, "single tensor or tfr.tensor type">;

// all allowed build list input types
def TFR_allowedBuiltListType : Type<Or<[
    TFR_TensorType.predicate,
    TF_ElementType.predicate,
    TFR_AttrType.predicate]>, "single tfr.tensor or tensor element type">;

// all allowed build list result types
def TFR_allowedListResultType : Type<Or<[
    TFR_TensorListType.predicate,
    TFR_AttrType.predicate]>, "tfr.tensor_list or tfr.attr type">;

//===----------------------------------------------------------------------===//
// Op classes
//===----------------------------------------------------------------------===//

class TFR_Op<string mnemonic, list<Trait> traits> :
    Op<TFR_Dialect, mnemonic, traits>;

def TFR_CallOp : TFR_Op<"call", [CallOpInterface]> {
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the callee. The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = tfr.call @my_add(%0, %1) : (tfr.tensor, f32) -> tfr.tensor_list
    ```

    Note that the operands of the `call` operation can only be with tfr.tensor,
    tfr.tensor_list, tfr.attr and mlir float and integer types. The results of
    the `call` operation can only be with tfr.tensor and tfr.tensor_list types.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<TFR_allowedArgType>:$args);

  let results = (outs
    Variadic<TFR_allowedResultType>:$outs);

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }

    // Get the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return calleeAttr(); }
  }];

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}

def TFR_CastOp : TFR_Op<"cast", [NoSideEffect]> {
  let description = [{
    The `cast` operation converts the operand with built-in tensor type to
    tfr.tensor type, or vice versa.

    Example:

    ```mlir
    %1 = tfr.cast(%0) : tensor<f32> -> !tfr.tensor
    %3 = tfr.cast(%1) : !tfr.tensor -> tensor<f32>
    ```
  }];

  let arguments = (ins TFR_singleTensorType:$arg);

  let results = (outs TFR_singleTensorType:$out);

  let extraClassDeclaration = [{
    // Return element type of the input tensor type. Only available when the
    // input is a MLIR built-in tensor type.
    Attribute getInputElementType() {
      if (auto ty = arg().getType().dyn_cast<TensorType>()) {
        return TypeAttr::get(ty.getElementType());
      }
      return {};
    }
  }];

  let hasCanonicalizer = 1;
}

def TFR_GetShapeOp : TFR_Op<"get_shape", [NoSideEffect]> {
  let description = [{
    The `get_shape` operation gets the shape of a tfr.tensor and returns
    !shape.shape type.

    Example:

    ```mlir
    %1 = "tfr.get_shape"(%0) : !tfr.tensor -> !shape.shape
    %1 = tfr.get_shape %0 -> !shape.shape
    ```
  }];

  let arguments = (ins TFR_TensorType:$arg);

  let results = (outs Shape_ShapeType:$out);

  let assemblyFormat = "$arg attr-dict `->` type($out)";

  let hasCanonicalizer = 1;
}

def TFR_GetElementTypeOp : TFR_Op<"get_element_type", [NoSideEffect]> {
  let description = [{
    The `get_element_type` operation gets the element type of a tfr.tensor and
    returns !tfr.attr.

    Example:

    ```mlir
    %1 = "tfr.get_element_type"(%0) : !tfr.tensor -> !tfr.attr
    %1 = tfr.get_element_type %0 -> !tfr.attr
    ```
  }];

  let arguments = (ins TFR_TensorType:$arg);

  let results = (outs TFR_AttrType:$out);

  let assemblyFormat = "$arg attr-dict `->` type($out)";
}

def TFR_EqualOp : TFR_Op<"equal", [NoSideEffect, SameTypeOperands]> {
  let description = [{
    The `equal` operation compares the values of the tfr.attr type arguments.
    The operation returns an i1 boolean indicating if the two values are the
    same.
    Example:

    ```mlir
    %x = tfr.equal %lhs, %rhs -> i1
    %x = "tfr.equal"(%lhs, %rhs) : (!tfr.attr, !tfr.attr) -> i1
    ```
  }];

  let arguments = (ins
      TFR_AttrType:$lhs,
      TFR_AttrType:$rhs
  );
  let results = (outs BoolLike:$result);

  let hasFolder = 1;

  let assemblyFormat = "$lhs `,` $rhs attr-dict `->` type($result)";
}

def TFR_ConstOp : TFR_Op<"constant", [ConstantLike, NoSideEffect]> {
  let description = [{
    The `attr` operation stores TF op's attribute, which doesn't support
    arithmetic operations.

    Example:

    ```mlir
    %1 = "tfr.constant"() { value: i32 } : () -> !tfr.attr
    %2 = "tfr.constant"() { value: [i32, f32] } : () -> !tfr.attr
    %3 = tfr.constant [i32, f32] -> !tfr.attr
    %4 = tfr.constant f32 -> !tfr.attr
    ```
  }];

  let arguments = (ins TFR_allowedConstValues:$value);

  let results = (outs TFR_AttrType:$out);

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{
      auto* ctx = value.getContext();
      $_state.addAttribute("value", value);
      $_state.addTypes(TFRAttrType::get(ctx));
    }]>
  ];

  let assemblyFormat = [{
    $value attr-dict `->` type($out)
  }];
}

def TFR_ConstantTensorOp : TFR_Op<"constant_tensor", [NoSideEffect]> {
  let description = [{
    The `constant_tensor` operation converts the operand with non-built-in
    tensor type to built-in tensor type or tfr.tensor type. If it is built-in
    tensor type, the shape shouldn't be changed during the conversion.

    Example:

    ```mlir
    %1 = tfr.constant_tensor(%0) : f32 -> tensor<f32>
    %3 = tfr.constant_tensor(%2) : vector<1xf32> -> tensor<1xf32>
    ```
  }];

  let arguments = (ins TFR_AllAttrTypes:$arg);

  let results = (outs TFR_singleTensorType:$out);

  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

def TFR_GetElementOp : TFR_Op<"get_element", [NoSideEffect]> {
  let description = [{
    The `get_element` operation extracts one tfr.tensor element from a
    tfr.tensor_list.

    Example:

    ```mlir
    %2 = tfr.get_element %1[%0] : (tfr.tensor, index) -> tfr.tensor
    ```
  }];

  let arguments = (ins
    TFR_TensorListType:$tensor_list,
    Index:$index);

  let results = (outs TFR_TensorType:$out);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $tensor_list `[` $index `]` attr-dict `:`
      `(` type($tensor_list) `,` type($index) `)` `->` type($out)
  }];
}

def TFR_BuildListOp : TFR_Op<"build_list", [NoSideEffect]> {
  let description = [{
   The `build_list` operation builds a tensor list from a list of tensors, or
   an tfr.attr from a list of scalars.

    Example:

    ```mlir
    %3 = tfr.build_list(%2, %1, %0) :
      (tfr.tensor, tfr.tensor, tfr.tensor) -> tfr.tensor_list
    %3 = tfr.build_list(%2, %1, %0) : (i32, i32, i32) -> tfr.attr
    ```
  }];

  let arguments = (ins Variadic<TFR_allowedBuiltListType>:$tensors);

  let results = (outs TFR_allowedListResultType:$out);

  let hasCanonicalizer = 1;
}

def TFR_GetLengthOp : TFR_Op<"get_length", [NoSideEffect]> {
  let description = [{
    The `get_length` operation returns the number of tensors for a
    tfr.tensor_list.

    Example:

    ```mlir
    %2 = tfr.get_length(%1) : tfr.tensor -> index
    %2 = tfr.get_length %1 -> index
    ```
  }];

  let arguments = (ins TFR_TensorListType:$tensor_list);

  let results = (outs Index:$out);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $tensor_list attr-dict `->` type($out)
  }];
}

//===----------------------------------------------------------------------===//
// Function related classes
//===----------------------------------------------------------------------===//

def TFR_TFRFuncOp : TFR_Op<"func", [HasParent<"ModuleOp">,
                                    DeclareOpInterfaceMethods<CallableOpInterface>,
                                    FunctionOpInterface,
                                    IsolatedFromAbove, Symbol]> {
  let summary = "TFR Function defines a composition of other ops";

  let description = [{
    Defines a function that can be used to decompose an TF function call to
    the invocation of a set of other TF ops.

    Syntax:

    ```
    op ::= `tfr.func` visibility? symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)? function-attributes? region
    ```

    Example:

    ```mlir
    tfr.func @foo(%arg0: !tfr.tensor, %arg1: !tfr.tensor_list<T>,
                  %arg2: int {tfr.name="T", tfr.default=1})
        attributes {qux: "quux"} {
      tfr.return
    }
    ```

    Note the arguments are ordered by the following rule:
      tfr.tensor > tfr.tensor_list > tfr.attr/i32/...,
    and only one trfr.tensor_list argument is allowed.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    StrAttr:$sym_name
  );

  let results = (outs);

  // When the regions is empty, the tfr.func is an external function and used
  // to model the element type constraints of the tf op. Otherwise, there is one
  // region containing the composition.
  let regions = (region VariadicRegion<AnyRegion>:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of this function.
    /// FIXME: We should drive this via the ODS `type` param.
    FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue().cast<FunctionType>();
    }
    LogicalResult verifyType() { return success(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    // Get the names of all defined attributes, including both derived and
    // non-derived ones.
    llvm::StringSet<> getDefinedAttributeNames() {
      llvm::StringSet<> all_attrs;
      for (auto& attr : (*this)->getAttrs()) {
        all_attrs.insert(attr.getName().strref());
      }
      for (const auto& operand : llvm::enumerate(getArgumentTypes())) {
        if (auto attr_name = getArgAttrOfType<StringAttr>(
            operand.index(), kAttrArgumentNameAttr)) {
          all_attrs.insert(attr_name.getValue());
        }
      }
      return all_attrs;
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TFR_TFRReturnOp : TFR_Op<"return", [HasParent<"TFRFuncOp">, NoSideEffect,
                                        ReturnLike, Terminator]> {
  let description = [{
    A terminator operation for regions that appear in the body of  `tfr.func`
    functions. The operands to the `tfr.return` are the result values returned
    by an invocation of the `tfr.func`.

    Note that only the tfr.tensor and tfr.tensor_list can be returned.
  }];

  let arguments = (ins Variadic<TFR_allowedResultType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Quantization related operations
//===----------------------------------------------------------------------===//

def TFR_TFRQuantActRangeOp : TFR_Op<"quant_act_range", [NoSideEffect]> {
  let description = [{
   The `quant_act_range` returns the a pair of integers to indicate the fixed
   range for the fused activation `act` with the quantization defined by the
   `scale` and `zero point`. Currently, the allowed activations are
   `NONE`, `RELU`, `RELU6` and `RELU_N1_TO_1`.

    Example:

    ```mlir
    %3, %4 = tfr.quant_act_range(%2, %1, %0) :
        (tfr.attr, float, i64) -> (tfr.tensor, tfr.tensor)
    ```
  }];

  let arguments = (ins
      TFR_AttrType:$act,
      F32:$scale,
      I64:$zp);

  let results = (outs TFR_TensorType:$min, TFR_TensorType:$max);

  let assemblyFormat = [{
    `(` $act `,` $scale `,` $zp `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TFR_TFRQuantRescaleOp : TFR_Op<"quant_rescale", [NoSideEffect]> {
  let description = [{
   The `quant_rescale` rescales the elements of the integer tensor by the the
   floating-point rescale factor. This op needs to be legalized to the preferred
   operations of the backends.

    Example:

    ```mlir
    %3 = tfr.quant_rescale(%2, %1, %0) :
        (tfr.tensor, tfr.tensor, i64) -> (tfr.tensor)
    ```
  }];

  let arguments = (ins
      TFR_TensorType:$input,
      TFR_TensorType:$scale,
      I64:$zp);

  let results = (outs TFR_TensorType:$output);

  let assemblyFormat = [{
    `(` $input `,` $scale `,` $zp `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def TFR_TFRQuantRawDataOp : TFR_Op<"quant_raw_data", [
    NoSideEffect,
    SameOperandsAndResultType]> {
  let description = [{
   The `quant_raw_data` removes the quantization parameter from the intput
   tensor(s).

    Example:

    ```mlir
    %3 = tfr.quant_raw_data(%0) : (tfr.tensor) -> (tfr.tensor)
    ```
  }];

  let arguments = (ins TFR_AllTensorTypes:$input);

  let results = (outs TFR_AllTensorTypes:$output);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, results)
  }];

  let hasCanonicalizer = 1;
}

def TFR_TFRQuantQParamsOp : TFR_Op<"quant_qparam", [NoSideEffect]> {
  let description = [{
   The `quant_qparam` returns the quantization parameter of the input
   tensors.

    Example:

    ```mlir
    %3 = tfr.quant_qparam(%0) : (tfr.tensor) -> (float, tfr.tensor)
    ```
  }];

  let arguments = (ins TFR_TensorType:$input);

  let results = (outs TFR_TensorType:$scale, TFR_TensorType:$zp);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, results)
  }];

  let hasCanonicalizer = 1;
}


def TFR_TFRQuantScaleFactorOp : TFR_Op<"quant_scale_factor", [NoSideEffect]> {
  let description = [{
   The `quant_scale_factor` computes the effective scale factor according to the
   output scale and input scales.

    Example:

    ```mlir
    %3 = tfr.quant_scale_factor(%0) : (f32, tfr.tensor_list) -> (tfr.tensor)
    ```
  }];

  let arguments = (ins
      F32:$out_scale,
      TFR_TensorListType:$in_scales);

  let results = (outs TFR_TensorType:$scale_factor);

  let assemblyFormat = [{
    `(` $out_scale `,` $in_scales `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

#endif // DIALECT_TFR_OPS_
