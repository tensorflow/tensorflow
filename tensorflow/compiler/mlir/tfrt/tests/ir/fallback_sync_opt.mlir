// RUN: tf-tfrt-opt %s | tf-tfrt-opt | FileCheck %s --dump-input=fail

func.func @test(%arg: !t.tensor) attributes {tfrt.sync} {
  // CHECK: tfrt_fallback_sync.executeop "tf.Relu"(%{{.*}}) {T = f32} : 1
  %0 = tfrt_fallback_sync.executeop "tf.Relu"(%arg) {T = f32} : 1
  tfrt.return
}
