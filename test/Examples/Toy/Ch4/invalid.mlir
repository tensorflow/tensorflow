// RUN: not toyc-ch4 %s -emit=mlir 2>&1


// This IR is not "valid":
// - toy.print should not return a value.
// - toy.print should take an argument.
// - There should be a block terminator.
// This all round-trip since this is opaque for MLIR.
func @main() {
  %0 = "toy.print"()  : () -> !toy<"array<2, 3>">
}
