// RUN: mlir-hlo-opt %s | mlir-hlo-opt
// RUN: diff <(mlir-hlo-opt %s) <(mlir-hlo-opt -emit-bytecode %s | mlir-hlo-opt)
// RUN: (! mlir-hlo-opt -debug %s) || mlir-hlo-opt -emit-bytecode -debug-only=mhlo-bytecode %s 2>&1 | (! grep 'Not Implemented')

// Test all attributes and types in MHLO
// Use round trip testing to validate both serialization and deserialization
func.func @test_bytecode_customizations(
  %arg0 : !mhlo.token,
  %arg1 : !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
) -> () attributes {
  enum_attrs  = [#mhlo.rng_algorithm<PHILOX>,
                 #mhlo.rng_distribution<NORMAL>,
                 #mhlo<comparison_direction LT>,
                 #mhlo<comparison_type FLOAT>,
                 #mhlo<fft_type RFFT>,
                 #mhlo<fusion_kind kCustom>,
                 #mhlo<kind sharding>,
                 #mhlo<precision DEFAULT>,
                 #mhlo<transpose NO_TRANSPOSE>],
  channel = #mhlo.channel_handle<handle = 0, type = 0>,
  conv = #mhlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
  dot = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>,
  scatter = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 2], index_vector_dim = 1>,
  gather = #mhlo.gather<collapsed_slice_dims = [0, 1], index_vector_dim = 2, offset_dims = [2], start_index_map = [0, 1]>,
  arg_alias = #mhlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = [1]>,
  res_alias = [#mhlo.result_alias<result_index = [2]>,
               #mhlo.result_alias<tuple_indices = [1, 1], result_index = [2, 0, 1], must_alias>],
  ext = #mhlo.type_extensions<bounds = [4]>
} {
  func.return
}
