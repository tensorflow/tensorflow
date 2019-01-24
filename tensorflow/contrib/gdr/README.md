Introduction
===

This is an implementation of GDR out-of-band transport for TensorFlow distributed runtime, complementary to current gRPC transport. It uses gRPC as control plane to setup rendezvous for each tensor transmission, and utilizes [GPU Direct RDMA](https://developer.nvidia.com/gpudirect) whenever possible to transmit tensors in remote GPU memory through network interface card (NIC), bypassing host memory and CPU entirely. It gracefully falls back to ordinary RDMA or even gRPC when GDR is not available.

Design
===

The GDR out-of-band transport is designed to avoid any unnecessary memory copies, especially for large tensors (>100MB). That typically requires registration of tensor buffers to NIC in an ad-hoc manner, which is rather slow as described in the design trade-off of the verbs runtime. The verbs runtime thus chooses to manage its own NIC-registered buffers and copy the tensors from/to those buffers for every single tensor transfer.

We show that, however, such design trade-off is not always relevant. In this patch, we manage both computation and communication buffers in a unified manner. By pre-registration of large buffers to NIC and allocating small tensors from the buffer pool using a BFC allocator, it is possible to avoid both ad-hoc buffer registration and memory copies all together.

For the actual tensor transport, we rely on gRPC to transmit the [remote buffer information](gdr.proto). This greatly simplifies our design, and there are only 2 types of RDMA messages: a single READ to retrieve the tensor data (bypassing remote CPU), and another invalidate using WRITE with IMM to release the tensor buffer on the remote side. The remote side will only be polling the invalidate message and `Unref` the tensor buffers that read by its peer.

Environment
===

To fully utilize GDR, the target environment has to meet 3 conditions:

1. There is an RDMA capable device with corresponding [OFED package](https://www.openfabrics.org/index.php/overview.html) installed (detailed information is available from your [Infiniband/RoCE](http://www.mellanox.com/page/products_dyn?product_family=116)/[iWarp](http://www.chelsio.com/gpudirect-rdma/) vendor), which could be verified through `ibv_devinfo`, e.g.

```
$ ibv_devinfo
hca_id:	mlx4_0
	transport:			InfiniBand (0)
	fw_ver:				2.40.7000
	node_guid:			248a:0703:00f6:3370
	sys_image_guid:			248a:0703:00f6:3370
	vendor_id:			0x02c9
	vendor_part_id:			4099
	hw_ver:				0x1
	board_id:			MT_1090110023
	phys_port_cnt:			2
	Device ports:
		port:	1
			state:			PORT_ACTIVE (4)
			max_mtu:		4096 (5)
			active_mtu:		1024 (3)
			sm_lid:			0
			port_lid:		0
			port_lmc:		0x00
			link_layer:		Ethernet

		port:	2
			state:			PORT_ACTIVE (4)
			max_mtu:		4096 (5)
			active_mtu:		1024 (3)
			sm_lid:			0
			port_lid:		0
			port_lmc:		0x00
			link_layer:		Ethernet
```

2. There is a GDR capable GPU, i.e. of Fermi, Kepler or later architecture with [corresponding driver](http://docs.nvidia.com/cuda/gpudirect-rdma/index.html) installed. The PCI-e topology could be confirmed by `nvidia-smi topo -m`. For example, in the following topology, `GPU2` and `GPU3` are adjacent to `mlx4_0`, and tensors on these devices could benefit from GDR in current implementation.

```
$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	mlx4_0	CPU Affinity
GPU0	 X 	PHB	SOC	SOC	SOC	0-5
GPU1	PHB	 X 	SOC	SOC	SOC	0-5
GPU2	SOC	SOC	 X 	PHB	PHB	6-11
GPU3	SOC	SOC	PHB	 X 	PHB	6-11
mlx4_0	SOC	SOC	PHB	PHB	 X

Legend:

  X   = Self
  SOC  = Connection traversing PCIe as well as the SMP link between CPU sockets(e.g. QPI)
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
```

3. The [`nv_peer_mem`](https://github.com/Mellanox/nv_peer_memory) kernel module is installed.

How to build and run in GDR mode
===

To test it out on a GDR capable environment, choose to enable GDR in your configure script.

```
Do you wish to build TensorFlow with GDR support? [y/N]: y
GDR support will be enabled for TensorFlow.
```

Change your `protocol` to `grpc+gdr` to enable GDR in your deployment.

```
server = tf.train.Server(cluster, job_name="local", task_index=0, protocol='grpc+gdr') # default protocol is 'grpc'
```

Currently the out-of-band transport service listens to the same IP and port address as specified in gRPC.

A successful initialization looks like this:

```
2017-08-05 19:10:38.601718: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K40m, pci bus id: 0000:02:00.0)
2017-08-05 19:10:38.601728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla K40m, pci bus id: 0000:03:00.0)
2017-08-05 19:10:38.601736: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:2) -> (device: 2, name: Tesla K40m, pci bus id: 0000:82:00.0)
2017-08-05 19:10:38.601742: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:3) -> (device: 3, name: Tesla K40m, pci bus id: 0000:83:00.0)
2017-08-05 19:10:39.591026: I tensorflow/contrib/gdr/gdr_memory_manager.cc:235] RDMA server is listening on 10.40.2.200:5001
2017-08-05 19:10:39.591071: I tensorflow/contrib/gdr/gdr_memory_manager.cc:285] Instrumenting CPU allocator cuda_host_bfc
2017-08-05 19:10:39.591083: I tensorflow/contrib/gdr/gdr_memory_manager.cc:285] Instrumenting CPU allocator cpu_pool
2017-08-05 19:10:39.591095: I tensorflow/contrib/gdr/gdr_memory_manager.cc:285] Instrumenting CPU allocator cpu_rdma_bfc
2017-08-05 19:10:39.591278: I tensorflow/contrib/gdr/gdr_memory_manager.cc:78] NUMA node for device: mlx4_0 is 1
2017-08-05 19:10:39.740253: I tensorflow/contrib/gdr/gdr_memory_manager.cc:296] Instrumenting GPU allocator with bus_id 2
```

The last line suggests that the GPUs with bus id 2 (mapped to pci bus id prefixed 0000:8) will benefit from GDR and host memory bypass, which is `/gpu:2` and `/gpu:3` in this case.

Caveats
===

In current implementation, only tensors that reside in host memory or in GPU memory such that the GPU is adjacent to an RDMA capable NIC will use direct RDMA as its transport. When RDMA is available but not GDR, a temporary tensor copy on host memory will be used as RDMA source/destination (and copied from/to the target device). When there is no RDMA device present, it can even fallback to the original gRPC runtime. While it is theoretically possible to mix GDR enabled TF with non-GDR deployments in the same job, make sure the environment is properly setup so the GDR mode is enabled whenever possible (i.e. do not fall back to gRPC when it is not absolutely necessary).

In the original design (as in the reference), tensor buffers are only registered to NIC when we could determine that the tensor will be either a source of Send or a sink of Recv across physical machine boundary. However, to implement the precise allocations, we need to change all the devices to possibly return a NIC compatible allocator. As GDR is currently in contrib, we would like to avoid the unnecessary code disruption to the TF core, so we allocate all tensors from NIC-registered buffers using a BFC allocator. This behavior is similar to the effect of enabling the extra GPU option `force_gpu_compatible`, which allocate all host tensors in GPU-registered buffers no matter they will be transferred from/to GPUs or not.

Reference
===

Bairen Yi, Jiacheng Xia, Li Chen, and Kai Chen. 2017. Towards Zero Copy Dataflows using RDMA. In Proceedings of SIGCOMM Posters and Demos'17, Los Angeles, CA, USA, August 22-24, 2017, 3 pages. https://doi.org/10.1145/3123878.3131975
