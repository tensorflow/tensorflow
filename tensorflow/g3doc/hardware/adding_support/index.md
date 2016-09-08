# Adding Support to New Hardware in TensorFlow

New compute hardware is be release frequently and 
it is of great interest to do machine learning on 
this hardware. Much of this hardware is designed 
with machine learning in mind and had features 
beyond the normal processor. Taking advantage of
these novel features is important to make the 
processor be used to its fullest. For example, the
[Intel Xeon Phi](http://www.intel.com/content/www/us/en/processors/xeon/xeon-phi-detail.html),
[KnuPath Hermosa](https://www.knupath.com/products/hermosa-processors/),
[IBM TrueNorth](http://researchweb.watson.ibm.com/articles/brain-chip.shtml), etc.
Custom support for each of these different processors
is needed to make TensorFlow capable of using them.

This is a walk through of how to add support for 
new hardware to TensorFlow. It is assumed that you
can link C/C++ code that uses your particular hardware
into TensorFlow. For simplicity, I will link
code that implement standard BLAS calls.

## TensorFlow Class Structure

The class structure for Devices,
* [class DeviceBase](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/device_base.h)
  * [class Device: public DeviceBase](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/device.h)
    * [class LocalDevice: public Device](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h)
      * [class BaseGPUDevice: public LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_device.h)
      	* [class GPUDevice: public BaseGPUDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_device_factory.cc)
      * [class ThreadPoolDevice: public LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/threadpool_device.h)
    * [class RemoteDevice: public Device](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/distributed_runtime/remote_device.cc)


There is also a structure referred to as a Device Factory,
* [class DeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/device_factory.h)
  * [class ThreadPoolDeviceFactory: public Device Factory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/threadpool_device_factory.cc)
  * [class BaseGPUDeviceFactory : public DeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_device.h)
    * [class GPUDeviceFactory: public BaseGPUDeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_device_factory.cc)
  * [class GPUCompatibleCPUDeviceFactory: public DeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_device_factory.cc)


It is clear that TensorFlow is currently built to support 3 major 
types of devices: CPU, GPU, Distributed. We will walk throught the 
creation of a new kind of device for each of these classes and leave 
it to the reader to decide which general type of device his/her 
hardware falls into best.


## New ThreadPool

We would like to create a new hardware interface that we can use in
standard TensorFlow operations. For example, we might want a ficticious
device called an SPU that I can use like,

```python
import tensorflow as tf

# Creates a graph.                                                              
with tf.device('/spu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.                      
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.                                                                  
print sess.run(c)
```

with output,
```bash
$ python test.py 
Device mapping: no known devices.
I tensorflow/core/common_runtime/direct_session.cc:175] Device mapping:

MatMul: /job:localhost/replica:0/task:0/spu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] MatMul: /job:localhost/replica:0/task:0/spu:0
b: /job:localhost/replica:0/task:0/spu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] b: /job:localhost/replica:0/task:0/spu:0
a: /job:localhost/replica:0/task:0/spu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] a: /job:localhost/replica:0/task:0/spu:0
[[ 22.  28.]
 [ 49.  64.]]
```

To do so, we need to implement a few classes and modify the C++
source to TensorFlow. Detail of this are found [here](fake_device.md),
but as a high level overview, you need to create or modify the following,

* `/configure` - Modify build parameters
* `/tensorflow/core/common_runtime/device_factory.cc` - Add bindings for the SPU device factory
* `/tensorflow/core/common_runtime/device_set.cc` - Define a priority ordering for the devices
* `/tensorflow/core/common_runtime/threadpiscine_device.{cc,h}` - Code for SPU support. Basically a copy of ThreadPool
* `/tensorflow/core/common_runtime/threadpiscine_device_factory.cc` - Add code that creates new devices as they are available. Here, it just creates threads as they are requested with arbitrary amount of memory.
* `/tensorflow/core/framework/types.{cc,h}` - Add a name for the SPU type.
* `/tensorflow/core/util/device_name_utils.cc` - Add parsing support for the SPU name
* `/tensorflow/python/framework/device.py` - Python bindings to SPU name

The [minimal changes](fake_device.md) to these files will create 
a fake device that runs on the CPU just as would a normal
CPU TensorFlow device specification, but allows a separation of code
so that we can begin replacing the underlying elements of TensorFlow for
the computation and make it run on our hardware.

## Creating a Fake GPU-style Device

We will now engage in a slightly more complicated,
but more useful, exercise because most new devices
will sit on the PCI Bus and have a similar interface
as GPUs. Also, because their memory is off the host
memory, it needs better management.

For this section, I will refer to my device as a JPU
(because J is next to G!). Just and FYI, I am leaving
my SPU code intact from the previous section. I
doubt this will cause problems, but it is left to 
the reader to adapt as needed if not anything arises.

Much more has to be done for a GPU kind of device.

* [Memory Allocator](jpu_allocator.md)
* [Device and Factory](jpu_device.md)