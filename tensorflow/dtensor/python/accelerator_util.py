from typing import List, Optional
import logging
from tensorflow.core.protobuf import cluster_pb2, tensorflow_server_pb2
from tensorflow.dtensor.python import config, tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.platform import remote_utils
from tensorflow.python.util.tf_export import tf_export

# Use an Enum to define accelerator types for better clarity
class AcceleratorType:
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"

class AcceleratorSystem:
    def __init__(self):
        self._initialized = None

    def is_initialized(self) -> bool:
        return bool(self._initialized)

    def set_initialized(self, value):
        self._initialized = value

    def initialize_multi_client_cluster(self, job_name: str,
                                        dtensor_jobs: List[str],
                                        client_id: int,
                                        collective_leader: str,
                                        port: Optional[int] = None,
                                        gpu_use_nccl_communication: bool = False,
                                        enable_coordination_service: bool = True):
        assert context.executing_eagerly()

        if not collective_leader.startswith("/job:"):
            collective_leader = "/job:" + collective_leader

        context.context().configure_collective_ops(
            use_nccl_communication=gpu_use_nccl_communication,
            collective_leader=collective_leader)
        if enable_coordination_service:
            context.context().configure_coordination_service(
                service_type="standalone", service_leader=collective_leader)

        config_proto = context.get_config()

        cluster_def = cluster_pb2.ClusterDef()
        cluster_def.job.add(name=job_name, tasks=dict(enumerate(dtensor_jobs))
        server_def = tensorflow_server_pb2.ServerDef(
            cluster=cluster_def,
            default_session_config=config_proto,
            job_name=job_name,
            task_index=client_id,
            protocol=remote_utils.get_default_communication_protocol(),
            port=port)
        server_def.default_session_config.rpc_options.num_channels_per_target = 4
        server_def.default_session_config.experimental.recv_buf_max_chunk = -1

        logging.info("Enabling collectives with server_def: %s", server_def)

        context.context().enable_collective_ops(server_def)
        context.ensure_initialized()

    @tf_export("experimental.dtensor.initialize_accelerator_system", v1=[])
    def initialize_accelerator_system(self, device_type: Optional[str] = None,
                                       enable_coordination_service: Optional[bool] = True,
                                       num_logical_cpu_devices: Optional[int] = None,
                                       experimental_reset_context: Optional[bool] = False) -> str:
        assert context.executing_eagerly()

        if self.is_initialized():
            raise ValueError("Accelerator system has already been initialized. Call tf.experimental.dtensor.shutdown_accelerator_system() first.")

        if experimental_reset_context:
            if context.context()._initialized:    # pylint: disable=protected-access
                logging.warn("experimental_reset_context is True. Resetting TensorFlow context. Existing TensorFlow objects (e.g. Tensors and resources) are invalidated.")
                context.context().ensure_uninitialized()

        if context.context()._initialized:  # pylint: disable=protected-access
            raise ValueError("TensorFlow has already been initialized. tf.experimental.dtensor.initialize_accelerator_system() must be called before TensorFlow is initialized.")

        context.context()._clear_caches()  # pylint: disable=protected-access

        if device_type is None:
            device_type = config.preferred_device_type()

        device_type = device_type.upper()
        if device_type not in {AcceleratorType.CPU, AcceleratorType.GPU, AcceleratorType.TPU}:
            raise ValueError(f"Unknown device_type {device_type}. Allowed values are CPU, GPU, or TPU")

        if config.gpu_use_nccl_communication():
            logical_gpu_count = config.num_local_devices(AcceleratorType.GPU)
            physical_gpu_count = len(tf_config.list_physical_devices(AcceleratorType.GPU))
            if logical_gpu_count > physical_gpu_count:
                raise ValueError("DTENSOR_GPU_USE_NCCL_COMMUNICATION is set for using NCCL. NCCL Collectives require one to one mapping between logical and physical GPUs. The number of logical GPU ({logical_gpu_count}) is more than the number of physical GPU ({physical_gpu_count}).")

        if device_type in (AcceleratorType.GPU, AcceleratorType.TPU):
            num_local_devices = config.num_local_devices(device_type)
            if num_logical_cpu_devices is None:
                num_logical_cpu_devices = max(config.num_local_devices(AcceleratorType.CPU), num_local_devices)
            else:
                if num_logical_cpu_devices < num_local_devices:
                    raise ValueError("If set, `num_logical_cpu_devices` (={num_logical_cpu_devices}) must be greater than or equal to the number of local {device_type} devices (={num_local_devices})")

        if num_logical_cpu_devices is not None:
            tf_config.set_logical_device_configuration(tf_config.list_physical_devices(AcceleratorType.CPU)[0], [context.LogicalDeviceConfiguration()] * num_logical_cpu_devices)

        if not config.is_local_mode():
            self.initialize_multi_client_cluster(
                job_name=config.job_name(),
                dtensor_jobs=config.jobs(),
                client_id=config.client_id(),
                collective_leader=config.full_job_name(task_id=0),
                gpu_use_nccl_communication=config.gpu_use_nccl_communication(),
                enable_coordination_service=enable_coordination_service)
        else:
            if device_type == AcceleratorType.GPU:
                context.context()._collective_use_nccl_communication = config.gpu_use_nccl_communication()

        if device_type == AcceleratorType.TPU and not config.backend_is_pw():
            tpu_util.initialize_tpu_system()

        self.set_initialized(device_type)

        return device_type

    @tf_export("experimental.dtensor.shutdown_accelerator_system", v1=[])
    def shutdown_accelerator_system(self):
        try:
            context.async_wait()
        finally:
            if not self.is_initialized():
                raise ValueError("Accelerator system is not initialized. Call tf.experimental.dtensor.initialize_accelerator_system first.")

            device_type = self._initialized

            if not config.is_local_mode():
                raise ValueError("Shutting down accelerator system under multi-client mode is not supported.")

            if device_type == AcceleratorType.TPU and not config.backend_is_pw():
                tpu_util.shutdown_tpu_system()

            context._reset_context()  # pylint: disable=protected-access
            context.context()._clear_caches()  # pylint: disable=protected-access
            self.set_initialized(None)
