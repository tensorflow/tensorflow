# Constructing a Fake Device in TensorFlow

A good starting point to adding new hardware is to add 
existing hardware under a different name! So here we add
the standard CPU under the name SPU by constructing a 
set of classes called ThreadPiscine (piscine means pool
in French).

We will create a new ThreadPool under a different name in 
the C++ code. Throughout this section, I will refer to it 
as ThreadPiscine (ie. pool -> piscine) and the hardware as
 a SPU (ie. C -> S). I will reimplement the code from 
TensorFlow with these names.

The diff for the minimal changes is broken up into parts here

```bash
diff --git a/configure b/configure
index 20bbf12..a96ffec 100755
--- a/configure
+++ b/configure
@@ -80,7 +80,17 @@ while [ "$TF_NEED_CUDA" == "" ]; do
   esac
 done
 
-if [ "$TF_NEED_CUDA" == "0" ]; then
+while [ "$TF_NEED_SPU" == "" ]; do
+  read -p "Do you wish to build TensorFlow with SPU support? [y/N] " INPUT
+  case $INPUT in
+    [Yy]* ) echo "SPU support will be enabled for TensorFlow"; TF_NEED_SPU=1;;
+    [Nn]* ) echo "No SPU support will be enabled for TensorFlow"; TF_NEED_SPU=0;;
+    "" ) echo "No SPU support will be enabled for TensorFlow"; TF_NEED_SPU=0;;
+    * ) echo "Invalid selection: " $INPUT;;
+  esac
+done
+
+if [[ "$TF_NEED_CUDA" == "0" ]] && [[ "$TF_NEED_SPU" == "0" ]]; then
   echo "Configuration finished"
   exit
 fi
@@ -111,6 +121,7 @@ done
 # Find out where the CUDA toolkit is installed
 OSNAME=`uname -s`
 
+if [[ "$TF_NEED_CUDA" == "1" ]]; then
 while true; do
   # Configure the Cuda SDK version to use.
   if [ -z "$TF_CUDA_VERSION" ]; then
@@ -314,5 +325,8 @@ fi
 
 # Invoke the cuda_config.sh and set up the TensorFlow's canonical view of the Cuda libraries
 (cd third_party/gpus/cuda; ./cuda_config.sh;) || exit -1
+fi
+
+# Configuration Prompting for SPUs
 
 echo "Configuration finished"
```

Add a binding to the Device Factory for the SPU
```C++
diff --git a/tensorflow/core/common_runtime/device_factory.cc b/tensorflow/core/common_runtime/device_factory.cc
index 7d0a238..6d122dd 100644
--- a/tensorflow/core/common_runtime/device_factory.cc
+++ b/tensorflow/core/common_runtime/device_factory.cc
@@ -95,6 +95,12 @@ void DeviceFactory::AddDevices(const SessionOptions& options,
     gpu_factory->CreateDevices(options, name_prefix, devices);
   }
 
+  // Then SPU
+  auto spu_factory = GetFactory("SPU");
+  if (spu_factory) {
+    spu_factory->CreateDevices(options, name_prefix, devices);
+  }
+
   // Then the rest.
   mutex_lock l(*get_device_factory_lock());
   for (auto& p : device_factories()) {

diff --git a/tensorflow/core/common_runtime/device_set.cc b/tensorflow/core/common_runtime/device_set.cc
index 98c6c38..60c88ef 100644
--- a/tensorflow/core/common_runtime/device_set.cc
+++ b/tensorflow/core/common_runtime/device_set.cc
@@ -56,6 +56,8 @@ int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
     return 3;
   } else if (StringPiece(d.type()) == DEVICE_GPU) {
     return 2;
+  } else if (StringPiece(d.type()) == DEVICE_SPU) {
+    return 4;
   } else {
     return 1;
   }
```

Implementation of ThreadPiscine by basically copying ThreadPool (note that piscine is French for pool ;)) and making simple substitutions to the code,
```C++
tensorflow/core/common_runtime$ diff thread{pool,piscine}_device.cc
16c16
< #include "tensorflow/core/common_runtime/threadpool_device.h"
---
> #include "tensorflow/core/common_runtime/threadpiscine_device.h"
32,35c32,35
< ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
<                                    const string& name, Bytes memory_limit,
<                                    BusAdjacency bus_adjacency,
<                                    Allocator* allocator)
---
> ThreadPiscineDevice::ThreadPiscineDevice(const SessionOptions& options,
>					   const string& name, Bytes memory_limit,
> 					   BusAdjacency bus_adjacency,
> 					   Allocator* allocator)
41c41
< ThreadPoolDevice::~ThreadPoolDevice() {}
---
> ThreadPiscineDevice::~ThreadPiscineDevice() {}
43c43
< void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
---
> void ThreadPiscineDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
55c55
< Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
---
> Allocator* ThreadPiscineDevice::GetAllocator(AllocatorAttributes attr) {
59c59
< Status ThreadPoolDevice::MakeTensorFromProto(
---
> Status ThreadPiscineDevice::MakeTensorFromProto(
```
```C++
tensorflow/core/common_runtime$ diff thread{pool,piscine}_device.h
16,17c16,17
< #ifndef TENSORFLOW_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
< #define TENSORFLOW_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
---
> #ifndef TENSORFLOW_COMMON_RUNTIME_THREADPISCINE_DEVICE_H_
> #define TENSORFLOW_COMMON_RUNTIME_THREADPISCINE_DEVICE_H_
25c25
< class ThreadPoolDevice : public LocalDevice {
---
> class ThreadPiscineDevice : public LocalDevice {
27c27
<   ThreadPoolDevice(const SessionOptions& options, const string& name,
---
>   ThreadPiscineDevice(const SessionOptions& options, const string& name,
30c30
<   ~ThreadPoolDevice() override;
---
>   ~ThreadPiscineDevice() override;
46c46
< #endif  // TENSORFLOW_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
---
> #endif  // TENSORFLOW_COMMON_RUNTIME_THREADPISCINE_DEVICE_H_
```
```C++
tensorflow/core/common_runtime$ diff thread{pool,piscine}_device_factory.cc
17c17
< #include "tensorflow/core/common_runtime/threadpool_device.h"
---
> #include "tensorflow/core/common_runtime/threadpiscine_device.h"
27c27
< class ThreadPoolDeviceFactory : public DeviceFactory {
---
> class ThreadPiscineDeviceFactory : public DeviceFactory {
34c34
<     auto iter = options.config.device_count().find("CPU");
---
>     auto iter = options.config.device_count().find("SPU");
39,41c39,41
<       string name = strings::StrCat(name_prefix, "/cpu:", i);
<       devices->push_back(new ThreadPoolDevice(options, name, Bytes(256 << 20),
<                                               BUS_ANY, cpu_allocator()));
---
>       string name = strings::StrCat(name_prefix, "/spu:", i);
>       devices->push_back(new ThreadPiscineDevice(options, name, Bytes(256 << 20),
> 			       				    	  	        BUS_ANY, cpu_allocator()));
45c45
< REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory);
---
> REGISTER_LOCAL_DEVICE_FACTORY("SPU", ThreadPiscineDeviceFactory);
```


Adding bindings for the processor type
```C++
diff --git a/tensorflow/core/framework/types.cc b/tensorflow/core/framework/types.cc
index d1e4d57..7f710ce 100644
--- a/tensorflow/core/framework/types.cc
+++ b/tensorflow/core/framework/types.cc
@@ -37,6 +37,7 @@ std::ostream& operator<<(std::ostream& os, const DeviceType& d) {
 
 const char* const DEVICE_CPU = "CPU";
 const char* const DEVICE_GPU = "GPU";
+const char* const DEVICE_SPU = "SPU";
 
 string DataTypeString(DataType dtype) {
   if (IsRefType(dtype)) {
diff --git a/tensorflow/core/framework/types.h b/tensorflow/core/framework/types.h
index b9690bd..2cdfe51 100644
--- a/tensorflow/core/framework/types.h
+++ b/tensorflow/core/framework/types.h
@@ -69,6 +69,7 @@ std::ostream& operator<<(std::ostream& os, const DeviceType& d);
 // Convenient constants that can be passed to a DeviceType constructor
 extern const char* const DEVICE_CPU;  // "CPU"
 extern const char* const DEVICE_GPU;  // "GPU"
+extern const char* const DEVICE_SPU;
 
 typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
 typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;
diff --git a/tensorflow/core/util/device_name_utils.cc b/tensorflow/core/util/device_name_utils.cc
index 5816dbd..c2873d7 100644
--- a/tensorflow/core/util/device_name_utils.cc
+++ b/tensorflow/core/util/device_name_utils.cc
@@ -162,6 +162,16 @@ bool DeviceNameUtils::ParseFullName(StringPiece fullname, ParsedName* p) {
       }
       progress = true;
     }
+    if (str_util::ConsumePrefix(&fullname, "/spu:") ||
+        str_util::ConsumePrefix(&fullname, "/SPU:")) {
+      p->has_type = true;
+      p->type = "SPU";  // Treat '/gpu:..' as uppercase '/device:GPU:...'
+      p->has_id = !str_util::ConsumePrefix(&fullname, "*");
+      if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
+        return false;
+      }
+      progress = true;
+    }
 
     if (!progress) {
       return false;
diff --git a/tensorflow/python/framework/device.py b/tensorflow/python/framework/device.py
index 8f5125d..fb2f2fb 100644
--- a/tensorflow/python/framework/device.py
+++ b/tensorflow/python/framework/device.py
@@ -155,7 +155,9 @@ class DeviceSpec(object):
         elif ly == 2 and y[0] == "task":
           self.task = y[1]
         elif ((ly == 1 or ly == 2) and
-              ((y[0].upper() == "GPU") or (y[0].upper() == "CPU"))):
+              ((y[0].upper() == "GPU") or 
+               (y[0].upper() == "CPU") or 
+               (y[0].upper() == "SPU"))):
           if self.device_type is not None:
             raise ValueError("Cannot specify multiple device types: %s" % spec)
           self.device_type = y[0].upper()
```


After making these changes, you can construct standard TensorFlow code that uses my SPUs!!

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