# JPU Device and Factory

In the GPU code, there are six classes that handle the device.

* `class BaseGPUDevice : public LocalDevice`
* `class BaseGPUDeviceFactory : public DeviceFactory`
* `class GPUDevice : public BaseGPUDevice`
* `class GPUDeviceFactory : public BaseGPUDeviceFactory`
* `class GPUCompatibleCPUDevice : public ThreadPoolDevice`
* `class GPUCompatibleCPUDeviceFactory : public DeviceFactory`

## Structure

The `BaseGPUDevice` and `BaseGPUDeviceFactory` implement a 
good portion of the GPU specific Device code and interface
as a `LocalDevice`. `GPUDevice` basically only makes specific
choices of the more general `BaseGPUDevice` options. Same for
`GPUDeviceFactory`, very little is changed.

An interesting line is found,
```C++
REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory);
```
which was modified when constructing the `ThreadPiscine`. These
macros register the device factory with Python via SWIG.

An interesting design choice is shown with `GPUCompatibleCPUDevice`,
these replace the normal CPU interface to enable CUDA DMA and 
other optimizations.

The GPU compatible devices get registered with a higher priority than
the non-compatible devices via the macro expansion of

```C++
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 50);
```
We want `GPUCompatibleCPUDevice` with higher priority so that CPU 
kernels can pass data faster to the GPU devices for processing. With
this macro, all the CPU devices will be run with the GPU compatible
code for optimized preformance.

which expands to,
```C++
namespace dfactory {

template <class Factory>
class Registrar {
 public:

  explicit Registrar(const string& device_type, int priority = 0) {
    DeviceFactory::Register(device_type, new Factory(), priority);
  }
};

}
static ::tensorflow::dfactory::Registrar<GPUCompatibleCPUDeviceFactory> ___0__object_("CPU", 50);
```

That way, all the GPU compatible devices are used up first and we 
get better IO with the GPUs.

## Extension

When adding a new device, it should be taken into consideration
if you wish to have GPU DMA working with your new device. For
most applications, due to the maturity of GPUs, GPU computation
with your new device is probably highly desirable. As such,
you probably want to make a new CPU device that is a sub-class of
`GPUCompatibleCPUDevice` so that you get the optimized GPU interface
along with an optimized interface for your device. You should set
the priority higher than 50 as well so the registration system
actually uses your new CPU device.

