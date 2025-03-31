#ifndef TRITON_PATTERNS
#define TRITON_PATTERNS

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"
include "mlir/IR/PatternBase.td"


// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)

// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
def CombineDotAddIPattern : Pat<
        (Arith_AddIOp $d, (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $overflow),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;
def CombineDotAddFPattern : Pat<
        (Arith_AddFOp $d, (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $fastmath),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"::llvm::cast<::mlir::IntegerAttr>($0).getInt() == 0">> $maxNumImpreciseAcc),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;

def CombineDotAddIRevPattern : Pat<
        (Arith_AddIOp (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $d, $overflow),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;
def CombineDotAddFRevPattern : Pat<
        (Arith_AddFOp (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $d, $fastmath),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"::llvm::cast<::mlir::IntegerAttr>($0).getInt() == 0">> $maxNumImpreciseAcc),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;

// addptr(addptr(%ptr, %idx0), %idx1) => addptr(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithDialect
//         (ref: ArithCanonicalization.td)
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;
def CombineAddPtrPattern : Pat<
        (TT_AddPtrOp (TT_AddPtrOp $ptr, $idx0), $idx1),
        (TT_AddPtrOp $ptr, (Arith_AddIOp $idx0, $idx1, DefOverflow)),
        [(Constraint<CPred<"isAddPtrOffsetCombinable($0, $1)">> $idx0, $idx1)]>;

#endif
