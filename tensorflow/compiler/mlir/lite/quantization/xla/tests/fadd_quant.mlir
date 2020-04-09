# RUN: not tfcompile --graph=%s.pbtxt --config=%s.config.pbtxt --experimental_quantize --cpp_class="::test::fadd_quant"  2>&1 | FileCheck %s -dump-input-on-failure

# TODO(fengliuai): update this file with the progress of the implementation

// CHECK: "quant.region"
// CHECK: ^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>):	// no predecessors
// CHECK:   xla_hlo.add %arg0, %arg1
// CHECK:   "quant.return"
// CHECK: }) {input_specs = [!quant.uniform<i8:f32, 0.49803921568627452:-128>, !quant.uniform<i8:f32, 0.49803921568627452:-128>],
// CHECK-SAME: logical_kernel = "generic.add", output_specs = [!quant.uniform<i8:f32, 0.49803921568627452:-128>]}
