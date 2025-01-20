// RUN: xla-opt --int4-to-packed-int4-rewrite %s

module {
  tt.func @dot_test(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}) -> tensor<16x16xi8> {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16: i64
    %0 = tt.make_tensor_ptr %arg0, [%c16, %c16], [%c16, %c16], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<16x16xi4>>
    %1 = tt.load %0 : !tt.ptr<tensor<16x16xi4>>
    %2 = arith.extsi %1 : tensor<16x16xi4> to tensor<16x16xi8>
    tt.return %2 : tensor<16x16xi8>
  }
}
