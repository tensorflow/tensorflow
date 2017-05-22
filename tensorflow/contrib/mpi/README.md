## How to compile and use MPI-enabled TensorFlow

1. Follow the regular TF compilation instructions. During configure step, if you want MPI support, answer yes to this question:

    ```Do you wish to build TensorFlow with MPI support [y/N]```

2. To turn on the MPI connection, add the protocol "grpc+mpi" in the server definition:

    ```server = tf.train.Server(cluster, job_name="local", task_index=0, protocol='grpc+mpi') # default protocol is 'grpc'```

## Overview

By using this protocol TensorFlow can take advantage of the high performance networking primitives that are offered via the MPI API. This enables TensorFlow to take advantage of high performance low latency networks such as Infiniband. These changes are largely transparent to the user who only has to change the offered protocol and launch the script using the 'mpirun'  launcher. For example:
    ```mpirun -np 2 python my_neuralnet.py ```





## Runtime options

The following environment variables can be set to modify the behavior at runtime:

**MPI_DISABLED=[0,1]**

This environment variable allows you to disable the MPI path before launch (e.g. for performance or correctness testing). 

**MPI_OPTIMAL_PATH=[0,1]**

When set to 0 it will use the default path where tensors are encoded to ProtoText before being copied to a remote process. When set to 1 a more optimal path will be taken where only the tensor description is encoded while the actual tensor data is transferred directly from the source buffer to the destination buffer.
This path is disabled by default as it requires that the MPI library can directly access the pointer to the data. For CPU backed buffers this is no problem, however for GPU backed buffers this requires MPI libraries that are built with CUDA support (CUDA Aware). When using non-CUDA aware MPI libraries and GPU buffers you will get segmentation faults.



## Known problems

For certain complex neural nets the implementation sometimes crashes inside the MPI libraries. This seems to be related to memory allocations/routines that register the memory for the Infiniband transfers. (The crashes do not happen when all MPI processes are within the same physical machine). 

**MVAPICH**
- The problem manifests itself with a segmentation fault inside a memory copy routine and during startup you will get the following warning: "WARNING: Error in initializing MVAPICH2 ptmalloc library. Continuing without InfiniBand registration cache support." 

**OpenMPI**
- With OpenMPI corrupt data will be received resulting in an assertion or the MPI library will print an error and exit. The error is "Attempt to free memory that is still in use by an ongoing MPI communication.  MPI job will now abort."

## Implementation details


The implementation takes over the responsibility for sending and receiving tensors between separate processes. This is facilitated by TensorFlow's ability to support different protocols. In this particular implementation, the standard gRPC library is used for all administrative operations while the MPI functions take over the tensor exchanges. On the sending side the tensors are placed in the standard waiting tables and nothing is changed there. On the receiving side the RecvFromRemoteAsync function is newly implemented and instead of requesting the data via gRPC the data is now requested via MPI calls.

To this end once the code is loaded a dedicated thread will be launched that handles all MPI operations. This thread will loop through a set of operations:

* Send requests placed on the request queue to the sending process
Once a request for a tensor is received two callbacks are created. The first one is to request the tensor and the second one is executed once the requested data has arrived. To this end the request is placed in a queue and will be sent once the MPI thread services the queue. This sending is done using non-blocking MPI_Isend operations.

* Send tensor data in response to a request call
Once a request has arrived from a remote process the request is forwarded to the original TensorFlow code which looks up the tensor in the waiting table. Once the tensor has been found a callback is executed which places the found tensor on the sendQueue for the MPI thread. Once the sendQueue is served the tensor data will be send using non-blocking send operations (MP_Isend) to the remote process.

* Receive tensor request
The MPI thread will check if there are any incoming tensor request messages on the communication lines using MPI_Iprobe. Once a request has been received it will be passed on to the standard TensorFlow code and eventually will be placed on the sendQueue.

* Receive tensor 
At some point after a request has been sent the remote process will transmit the tensor. This tensor will be received and we look-up the callback that is associated with this tensor in our request table and execute the callback on the received data.


In the implementation all send operations are non-blocking, all probe operations are non-blocking and all receive-operations are blocking. The receive-operations are only executed after the probe has determined that there is something to receive. 
The MPI processes identify each other using an MPI process ID. The TensorFlow gRPC processes identify each other using a name. During launch we create a mapping between the TensorFlow process name and the MPI process ID to allow the processes to communicate with the correct destinations when using MPI operations.




























