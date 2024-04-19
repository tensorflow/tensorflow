// RUN: hlo_to_kernel --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70

func.func @AddV2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> attributes {llvm.emit_c_interface, tf_entry} {
  %0 = shape.const_shape [1, 1, 1, 1, 1] : tensor<5xindex>
  %c5 = arith.constant 5 : index
  %1 = shape.const_shape [1, 1, 1, 1] : tensor<4xindex>
  %c4 = arith.constant 4 : index
  %2 = shape.const_shape [1, 1, 1] : tensor<3xindex>
  %c3 = arith.constant 3 : index
  %3 = shape.const_shape [1, 1] : tensor<2xindex>
  %c2 = arith.constant 2 : index
  %4 = shape.const_shape [1] : tensor<1xindex>
  %c1 = arith.constant 1 : index
  %5 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %6 = shape.shape_of %arg1 : tensor<*xf32> -> tensor<?xindex>
  %7 = shape.num_elements %5 : tensor<?xindex> -> index
  %8 = arith.cmpi eq, %7, %c1 : index
  %9 = scf.if %8 -> (tensor<*xf32>) {
    %14 = shape.num_elements %6 : tensor<?xindex> -> index
    %from_elements = tensor.from_elements %14 : tensor<1xindex>
    %15 = mhlo.reshape %arg0 : (tensor<*xf32>) -> tensor<f32>
    %16 = mhlo.dynamic_reshape %arg1, %from_elements : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
    %17 = chlo.broadcast_add %15, %16 : (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
    %cast = tensor.cast %17 : tensor<?xf32> to tensor<*xf32>
    scf.yield %cast : tensor<*xf32>
  } else {
    %14 = shape.num_elements %6 : tensor<?xindex> -> index
    %15 = arith.cmpi eq, %14, %c1 : index
    %16 = scf.if %15 -> (tensor<*xf32>) {
      %17 = shape.num_elements %5 : tensor<?xindex> -> index
      %from_elements = tensor.from_elements %17 : tensor<1xindex>
      %18 = mhlo.dynamic_reshape %arg0, %from_elements : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
      %19 = mhlo.reshape %arg1 : (tensor<*xf32>) -> tensor<f32>
      %20 = chlo.broadcast_add %18, %19 : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
      %cast = tensor.cast %20 : tensor<?xf32> to tensor<*xf32>
      scf.yield %cast : tensor<*xf32>
    } else {
      %17 = shape.shape_eq %5, %6 : tensor<?xindex>, tensor<?xindex>
      %18 = scf.if %17 -> (tensor<*xf32>) {
        %19 = shape.any %5, %6 : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
        %20 = shape.num_elements %19 : tensor<?xindex> -> index
        %from_elements = tensor.from_elements %20 : tensor<1xindex>
        %21 = mhlo.dynamic_reshape %arg0, %from_elements : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
        %22 = mhlo.dynamic_reshape %arg1, %from_elements : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
        %23 = chlo.broadcast_add %21, %22 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
        %cast = tensor.cast %23 : tensor<?xf32> to tensor<*xf32>
        scf.yield %cast : tensor<*xf32>
      } else {
        %19:2 = chlo.minimum_broadcast_shapes %5, %6 : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
        %20 = shape.rank %19#0 : tensor<?xindex> -> index
        %21 = shape.rank %19#1 : tensor<?xindex> -> index
        %22 = arith.cmpi sgt, %20, %21 : index
        %23 = arith.select %22, %20, %21 : index
        %24 = arith.cmpi ule, %23, %c1 : index
        %25 = scf.if %24 -> (tensor<*xf32>) {
          %26 = shape.broadcast %19#0, %4 : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
          %cast = tensor.cast %26 : tensor<?xindex> to tensor<1xindex>
          %27 = mhlo.dynamic_reshape %arg0, %cast : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
          %28 = shape.broadcast %19#1, %4 : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
          %cast_0 = tensor.cast %28 : tensor<?xindex> to tensor<1xindex>
          %29 = mhlo.dynamic_reshape %arg1, %cast_0 : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
          %30 = chlo.broadcast_add %27, %29 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
          %cast_1 = tensor.cast %30 : tensor<?xf32> to tensor<*xf32>
          scf.yield %cast_1 : tensor<*xf32>
        } else {
          %26 = arith.cmpi ule, %23, %c2 : index
          %27 = scf.if %26 -> (tensor<*xf32>) {
            %28 = shape.broadcast %19#0, %3 : tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
            %cast = tensor.cast %28 : tensor<?xindex> to tensor<2xindex>
            %29 = mhlo.dynamic_reshape %arg0, %cast : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
            %30 = shape.broadcast %19#1, %3 : tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
            %cast_0 = tensor.cast %30 : tensor<?xindex> to tensor<2xindex>
            %31 = mhlo.dynamic_reshape %arg1, %cast_0 : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
            %32 = chlo.broadcast_add %29, %31 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
            %cast_1 = tensor.cast %32 : tensor<?x?xf32> to tensor<*xf32>
            scf.yield %cast_1 : tensor<*xf32>
          } else {
            %28 = arith.cmpi ule, %23, %c3 : index
            %29 = scf.if %28 -> (tensor<*xf32>) {
              %30 = shape.broadcast %19#0, %2 : tensor<?xindex>, tensor<3xindex> -> tensor<?xindex>
              %cast = tensor.cast %30 : tensor<?xindex> to tensor<3xindex>
              %31 = mhlo.dynamic_reshape %arg0, %cast : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
              %32 = shape.broadcast %19#1, %2 : tensor<?xindex>, tensor<3xindex> -> tensor<?xindex>
              %cast_0 = tensor.cast %32 : tensor<?xindex> to tensor<3xindex>
              %33 = mhlo.dynamic_reshape %arg1, %cast_0 : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
              %34 = chlo.broadcast_add %31, %33 : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
              %cast_1 = tensor.cast %34 : tensor<?x?x?xf32> to tensor<*xf32>
              scf.yield %cast_1 : tensor<*xf32>
            } else {
              %30 = arith.cmpi ule, %23, %c4 : index
              %31 = scf.if %30 -> (tensor<*xf32>) {
                %32 = shape.broadcast %19#0, %1 : tensor<?xindex>, tensor<4xindex> -> tensor<?xindex>
                %cast = tensor.cast %32 : tensor<?xindex> to tensor<4xindex>
                %33 = mhlo.dynamic_reshape %arg0, %cast : (tensor<*xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
                %34 = shape.broadcast %19#1, %1 : tensor<?xindex>, tensor<4xindex> -> tensor<?xindex>
                %cast_0 = tensor.cast %34 : tensor<?xindex> to tensor<4xindex>
                %35 = mhlo.dynamic_reshape %arg1, %cast_0 : (tensor<*xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
                %36 = chlo.broadcast_add %33, %35 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
                %cast_1 = tensor.cast %36 : tensor<?x?x?x?xf32> to tensor<*xf32>
                scf.yield %cast_1 : tensor<*xf32>
              } else {
                %32 = arith.cmpi ule, %23, %c5 : index
                cf.assert %32, "Input for dynamic binary or n-ary op lowering was of a rank greater than 5"
                %33 = shape.broadcast %19#0, %0 : tensor<?xindex>, tensor<5xindex> -> tensor<?xindex>
                %cast = tensor.cast %33 : tensor<?xindex> to tensor<5xindex>
                %34 = mhlo.dynamic_reshape %arg0, %cast : (tensor<*xf32>, tensor<5xindex>) -> tensor<?x?x?x?x?xf32>
                %35 = shape.broadcast %19#1, %0 : tensor<?xindex>, tensor<5xindex> -> tensor<?xindex>
                %cast_0 = tensor.cast %35 : tensor<?xindex> to tensor<5xindex>
                %36 = mhlo.dynamic_reshape %arg1, %cast_0 : (tensor<*xf32>, tensor<5xindex>) -> tensor<?x?x?x?x?xf32>
                %37 = chlo.broadcast_add %34, %36 : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
                %cast_1 = tensor.cast %37 : tensor<?x?x?x?x?xf32> to tensor<*xf32>
                scf.yield %cast_1 : tensor<*xf32>
              }
              scf.yield %31 : tensor<*xf32>
            }
            scf.yield %29 : tensor<*xf32>
          }
          scf.yield %27 : tensor<*xf32>
        }
        scf.yield %25 : tensor<*xf32>
      }
      scf.yield %18 : tensor<*xf32>
    }
    scf.yield %16 : tensor<*xf32>
  }
  %10 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %11 = shape.shape_of %arg1 : tensor<*xf32> -> tensor<?xindex>
  %12 = shape.broadcast %10, %11 : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  %13 = mhlo.dynamic_reshape %9, %12 : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
  return %13 : tensor<*xf32>
}
