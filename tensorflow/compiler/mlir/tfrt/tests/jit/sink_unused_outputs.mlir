// RUN: tf-tfrt-opt -split-input-file -tf-cpurt-sink-unused-outputs %s |\
// RUN: FileCheck %s

func @unused_output(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %1 = linalg.init_tensor [%0] : tensor<?xf32>
  %2 = linalg.init_tensor [%0] : tensor<?xf32>
  %output = linalg.fill(%cst, %1) : f32, tensor<?xf32> -> tensor<?xf32>
  %4 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %5 = tensor.dim %input, %c1 : tensor<?x?xf32>
  %6:2 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%4, %5)
           step (%c4, %c4)
           ins (%input_ = %input: tensor<?x?xf32>)
           outs (%output_ = %output: tensor<?xf32>,
                 %unused_out = %2: tensor<?xf32>)
           iterators["parallel", "reduction"] {
    %7 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%i)[%4]
    %8 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%j)[%5]
    %10 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%i)[%4]

    %input_sub =  tensor.extract_slice %input_[%i, %j] [%7, %8] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
    %output_sub = tensor.extract_slice %output_[%i] [%10] [1]
            : tensor<?xf32> to tensor<?xf32>
    %unused_out_sub = tensor.extract_slice %unused_out[%i] [%10] [1]
            : tensor<?xf32> to tensor<?xf32>

    %fill = linalg.fill(%cst, %unused_out_sub)
            : f32, tensor<?xf32> -> tensor<?xf32>
    %partial_result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input_sub : tensor<?x?xf32>)
        outs(%fill : tensor<?xf32>) {
      ^bb0(%arg6: f32, %arg7: f32):
        %18 = arith.addf %arg6, %arg7 : f32
        linalg.yield %18 : f32
      } -> tensor<?xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%partial_result : tensor<?xf32>)
        outs(%output_sub : tensor<?xf32>) {
    ^bb0(%arg6: f32, %arg7: f32):
      %18 = arith.addf %arg6, %arg7 : f32
      linalg.yield %18 : f32
    } -> tensor<?xf32>
    %16 = tensor.insert_slice %partial_result into %unused_out[%i] [%10] [1]
            : tensor<?xf32> into tensor<?xf32>
    %17 = tensor.insert_slice %result into %output_[%i] [%10] [1]
            : tensor<?xf32> into tensor<?xf32>
    linalg.yield %17, %16 : tensor<?xf32>, tensor<?xf32>
  }
  return %6#0 : tensor<?xf32>
}
// CHECK-LABEL: func @unused_output

// CHECK:      %[[SINGLE_RESULT:.*]] = linalg.tiled_loop
// CHECK-SAME:   ins (%[[INPUT_:.*]] = %{{.*}}: tensor<?x?xf32>)
// CHECK-SAME:   outs (%[[OUTPUT_:.*]] = %{{.*}}: tensor<?xf32>)
// CHECK:          tensor.extract_slice %[[INPUT_]]
// CHECK-NEXT:     tensor.extract_slice %[[OUTPUT_]]
// CHECK-NEXT:     %[[INIT:.*]] = linalg.init_tensor
// CHECK-NEXT:     linalg.fill(%{{.*}}, %[[INIT]])
// CHECK-NEXT:     linalg.generic
// CHECK:          linalg.generic
// CHECK:          tensor.insert_slice %{{.*}} into %[[OUTPUT_]]
// CHECK:       return %[[SINGLE_RESULT:.*]] : tensor<?xf32>
