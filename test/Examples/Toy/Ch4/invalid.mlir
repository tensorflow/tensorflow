// RUN: not toyc-ch4 %s -emit=mlir 2>&1

// The following IR is not "valid":
// - toy.print should not return a value.
// - toy.print should take an argument.
// - There should be a block terminator.
func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
