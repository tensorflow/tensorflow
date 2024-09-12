// RUN: hlo_to_kernel --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70

func.func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf_entry} {
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = shape.num_elements %0 : tensor<?xindex> -> index
  %from_elements = tensor.from_elements %1 : tensor<1xindex>
  %2 = mhlo.dynamic_reshape %arg0, %from_elements : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  %3 = mhlo.tanh %2 : tensor<?xf32>
  %4 = mhlo.dynamic_reshape %3, %0 : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
