module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:2", "/job:localhost/replica:0/task:0/device:TPU:3", "/job:localhost/replica:0/task:0/device:TPU:4", "/job:localhost/replica:0/task:0/device:TPU:5", "/job:localhost/replica:0/task:0/device:TPU:6", "/job:localhost/replica:0/task:0/device:TPU:7"]} {
  func.func @main(%arg0: tensor<*xf32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) {
    "tf_device.cluster_func"(%arg0) <{func = @empty_func}> {_dynamic_arg_index = [], _replication_info = "cluster", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], input_sharding_configuration = ["{devices=[2,1]0,1}"], num_cores_per_replica = 2 : i64, output_sharding_configuration = [""], padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<*xf32>) -> (tensor<*xf32>)
    func.return
  }
  func.func @empty_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}
