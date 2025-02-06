module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'num_cores_per_replica'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}