// RUN: tf-tfrt-opt -tfrt-dist-remote-run-encapsulate %s | FileCheck %s


func private @init(%arg0 : !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle {tfrt.device = "/job:worker1/task:0/device:CPU:0"}) attributes {host = "/job:worker1/task:0"} {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %arg1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 65536], values = [1.0 : f32] } : 1
  %result = corert.executeop(%cpu) "tf.AddV2"(%arg0, %arg1) : 1
  tfrt.return %ch0, %result : !tfrt.chain, !corert.tensorhandle
}

func private @print(%chain : !tfrt.chain, %tensor_handle : !corert.tensorhandle) -> (!tfrt.chain) attributes {host = "/job:worker1/task:0"} {
  %ch2 = "corert.print_tensorhandle"(%tensor_handle, %chain) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain
  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: func @remote_execute
func @remote_execute(%arg0 : !corert.tensorhandle) -> (!tfrt.chain, !tfrt.chain, !corert.tensorhandle) {
  %c0 = tfrt.new.chain
  // CHECK: %[[CONFIGS:.*]]:2 = "tfrt_dist.test_create_configurations"()
  %configs:2 = tfrt_dist.test_create_configurations : 2
  // CHECK-NEXT: %[[CLIENT_CTX:.*]] = tfrt_dist.test_create_distributed_context %[[CONFIGS]]#0
  %client_context = tfrt_dist.test_create_distributed_context %configs#0 : (!tfrt_dist.dist_context_configuration) -> !tfrt_dist.dist_context
  // CHECK-NEXT: %[[WORKER_TASK:.*]] = tfrt_dist.get_task_handle %[[CLIENT_CTX]] {task_name = "/job:worker/task:1"}
  %worker_task = tfrt_dist.get_task_handle %client_context {task_name = "/job:worker/task:1"}
  // This is the remote invocation of the @print and @init functions, check that
  // we correctly serialize and encapsulate them.
  // CHECK-NEXT: %[[REGISTER_CHAIN_0:.*]] = tfrt_dist.register_tfrt_function(%[[IN_CHAIN:.*]], %[[CLIENT_CTX]], %[[WORKER_TASK]]) "init" {{.*}}func @init(%[[ARG_0:.*]]: !corert.tensorhandle loc({{.*}}) -> (!tfrt.chain, !corert.tensorhandle {{.*}}
  // CHECK-NEXT: %[[SPEC_0:.*]] = tfrt_dist.create_remote_execute_spec
  // CHECK-SAME: {output_devices = ["/job:worker1/task:0/device:CPU:0", "/job:worker1/task:0/device:CPU:0"]}
  // CHECK-NEXT: %[[OBJECT_ID_0:.*]] = tfrt_dist.get_remote_object_id_from_th %[[ARG_1:.*]]
  // CHECK-NEXT: %[[EXEC_CHAIN_0:.*]], %[[RESULTS_0:.*]]:3 = tfrt_dist.remote_execute_th[%[[REGISTER_CHAIN_0]], %[[CLIENT_CTX]], %[[WORKER_TASK]], %[[SPEC_0]], 1] "init"(%[[OBJECT_ID_0]]) : (!tfrt_dist.remote_object_id) -> (!tfrt_dist.remote_object_id, !tfrt_dist.remote_object_id, !corert.tensorhandle)
  %execute_chain, %remote_chain, %remote_tensor = tfrt_dist.remote_execute_func [%c0, %client_context, %worker_task] @init(%arg0) : (!corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)

  // CHECK-NEXT: %[[REGISTER_CHAIN_1:.*]] = tfrt_dist.register_tfrt_function(%[[EXEC_CHAIN_0]], %[[CLIENT_CTX]], %[[WORKER_TASK]]) "print" {{.*}}@print(%[[ARG_2:.*]]: !tfrt.chain loc({{.*}}), %[[ARG_3:.*]]: !corert.tensorhandle loc({{.*}})) -> !tfrt.chain {{.*}}
  // CHECK-NEXT: %[[SPEC_1:.*]] = tfrt_dist.create_remote_execute_spec
  // CHECK-SAME: {output_devices = ["/job:worker1/task:0/device:CPU:0"]}
  // CHECK-NEXT: %[[OBJECT_ID_1:.*]] = tfrt_dist.get_remote_object_id_from_th %[[RESULTS_0]]#2
  // CHECK-NEXT: %[[EXEC_CHAIN_1:.*]], %[[RESULTS_1:.*]] = tfrt_dist.remote_execute_th[%[[REGISTER_CHAIN_1]], %[[CLIENT_CTX]], %[[WORKER_TASK]], %[[SPEC_1]], 0] "print"(%[[RESULTS_0]]#0, %[[OBJECT_ID_1]]) : (!tfrt_dist.remote_object_id, !tfrt_dist.remote_object_id) -> !tfrt_dist.remote_object_id
  %execute_chain2, %remote_chain2 = tfrt_dist.remote_execute_func[%execute_chain, %client_context, %worker_task] @print(%remote_chain, %remote_tensor) : (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain)


  // CHECK-NEXT: tfrt.return %[[EXEC_CHAIN_0]], %[[EXEC_CHAIN_1]], %[[RESULTS_0]]#2 : !tfrt.chain, !tfrt.chain, !corert.tensorhandle
  tfrt.return %execute_chain, %execute_chain2, %remote_tensor : !tfrt.chain, !tfrt.chain, !corert.tensorhandle
}

