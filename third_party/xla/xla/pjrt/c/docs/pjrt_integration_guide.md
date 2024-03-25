# PJRT integration guide

[jieying@google.com](mailto:jieying@google.com), [skyewm@google.com](mailto:skyewm@google.com)

## Background

[PJRT](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h) is the uniform Device API that we want to add to the ML ecosystem. The long term vision is that: (1) frameworks (JAX, TF, etc.) will call PJRT, which has device-specific implementations that are opaque to the frameworks; (2) each device focuses on implementing PJRT APIs, and can be opaque to the frameworks.

This doc focuses on the recommendations about how to integrate with PJRT, and how to test PJRT integration with JAX.

## How to integrate with PJRT

### Step 1: Implement [PJRT C API interface](https://github.com/openxla/xla/blob/71a4e6e6e4e9f0f8b8f25c07a32ad489aff19239/xla/pjrt/c/pjrt_c_api.h)

**Option A**: You can implement the PJRT C API directly.

**Option B**: If you're able to build against C++ code in the [xla repo](https://github.com/openxla/xla) (via forking or bazel), you can also implement the PJRT C++ API and use the C→C++ wrapper:

1. Implement a C++ PJRT client inheriting from the [base PJRT client](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h) (and related PJRT classes). Here are some examples of C++ PJRT client: [pjrt\_stream\_executor\_client.h](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/pjrt_stream_executor_client.h), [tfrt\_cpu\_pjrt\_client.h](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/tfrt_cpu_pjrt_client.h).
1. Implement a few C API methods that are not part of C++ PJRT client:
  * [PJRT\_Client\_Create](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L344-L365). Below is some sample pseudo code (assuming `GetPluginPjRtClient` returns a C++ PJRT client implemented above):
```
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace my_plugin {
PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  std::unique_ptr<xla::PjRtClient> client = GetPluginPjRtClient();
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}
}  // namespace my_plugin
```
  Note [PJRT\_Client\_Create](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L344-L365) can take options passed from the framework. [Here](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L48-L102) is an example of how a GPU client uses this feature.

  * [Optional] [PJRT\_TopologyDescription\_Create](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L1815-L1830).
  * [Optional] [PJRT\_Plugin\_Initialize](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L173-L180). This is a one-time plugin setup, which will be called by the framework before any other functions are called.
  * [Optional] [PJRT\_Plugin\_Attributes](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L182-L194).

With the [wrapper](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_wrapper_impl.h), you do not need to implement the remaining C APIs.


### Step 2: Implement GetPjRtApi

You need to implement a method `GetPjRtApi` which returns a `PJRT_Api*` containing function pointers to PJRT C API implementations. Below is an example assuming implementing through wrapper (similar to [pjrt\_c\_api\_cpu.cc](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_cpu.cc)):
```
const PJRT_Api* GetPjrtApi() {
  static const PJRT_Api pjrt_api =
      pjrt::CreatePjrtApi(my_plugin::PJRT_Client_Create);
  return &pjrt_api;
}
```

### Step 3: Test C API implementations

You can call [RegisterPjRtCApiTestFactory](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_test.h#L31C6-L31C33) to run a small set of tests for basic PJRT C API behaviors.

## How to use a PJRT plugin from JAX

### Step 1: Set up JAX

You can either use JAX nightly
```
pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

pip install git+https://github.com/google/jax
```
or [build JAX from source](https://jax.readthedocs.io/en/latest/developer.html#building-jaxlib-from-source).

For now, you need to match the jaxlib version with the PJRT C API version. It's usually sufficient to use a jaxlib nightly version from the same day as the TF commit you're building your plugin against, e.g.
```
pip install --pre -U jaxlib==0.4.2.dev20230103 -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```
You can also build a jaxlib from source at exactly the XLA commit you're building against ([instructions](https://jax.readthedocs.io/en/latest/developer.html#building-jaxlib-from-source-with-a-modified-xla-repository)).

We will start supporting ABI compatibility soon.

### Step 2: Use jax\_plugins namespace or set up entry\_point

There are two options for your plugin to be discovered by JAX.

1. Using namespace packages ([ref](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-naming-convention)). Define a globally unique module under the `jax_plugins` namespace package (i.e. just create a `jax_plugins` directory and define your module below it). Here is an example directory structure:
```
jax_plugins/
  my_plugin/
    __init__.py
    my_plugin.so
```
2. Using package metadata ([ref](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)). If building a package via pyproject.toml or setup.py, advertise your plugin module name by including an entry-point under the `jax_plugins` group which points to your full module name. Here is an example via pyproject.toml or setup.py:
```
# use pyproject.toml
[project.entry-points.'jax_plugins']
my_plugin = 'my_plugin'

# use setup.py
entry_points={
  "jax_plugins": [
    "my_plugin = my_plugin",
  ],
}
```
Here are examples of how openxla-pjrt-plugin is implemented using Option 2: https://github.com/openxla/openxla-pjrt-plugin/pull/119, https://github.com/openxla/openxla-pjrt-plugin/pull/120.

### Step 3: Implement an initialize() method

You need to implement an initialize() method in your python module to register the plugin, for example:
```
import os
import jax._src.xla_bridge as xb

def initialize():
  path = os.path.join(os.path.dirname(__file__), 'my_plugin.so')
  xb.register_plugin('my_plugin', priority=500, library_path=path, options=None)
```
Please refer to [here](https://github.com/google/jax/blob/8f283bc9ed50d3828bd468ae57b1ee4df1527624/jax/_src/xla_bridge.py#L420) about how to use `xla_bridge.register_plugin`. It is currently a private method. A public API will be released in the future.

You can run the line below to verify that the plugin is registered and raise an error if it can't be loaded.
```
jax.config.update("jax_platforms", "my_plugin")
```
JAX may have multiple backends/plugins. There are a few options to ensure your plugin is used as the default backend:
*   Option 1: run `jax.config.update("jax_platforms", "my_plugin")` in the beginning of the program.
*   Option 2: set ENV `JAX_PLATFORMS=my_plugin`.
*   Option 3: set a high enough priority when calling xb.register\_plugin (the default value is 400 which is higher than other existing backends). Note the backend with highest priority will be used only when `JAX_PLATFORMS=''`. The default value of `JAX_PLATFORMS` is `''` but sometimes it will get overwritten.

## How to test with JAX

Some basic test cases to try:
```
# JAX 1+1
print(jax.numpy.add(1, 1))
# => 2

# jit
print(jax.jit(lambda x: x * 2)(1.))
# => 2.0

# pmap

arr = jax.numpy.arange(jax.device_count()) print(jax.pmap(lambda x: x +
jax.lax.psum(x, 'i'), axis_name='i')(arr))

# single device: [0]

# 4 devices: [6 7 8 9]

```
(We'll add instructions for running the jax unit tests against your plugin soon!)

## Example: JAX CUDA plugin

1. PJRT C API implementation through wrapper ([pjrt\_c\_api\_gpu.h](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_gpu.h)).
1. Set up the entry point for the package ([setup.py](https://github.com/google/jax/blob/main/jax_plugins/cuda/setup.py)).
1. Implement an initialize() method ([\_\_init\_\_.py](https://github.com/google/jax/blob/a10854786b6d1bc92a65dd314916b151640789af/plugins/cuda/__init__.py#L31-L51)).
1. Can be tested with any jax tests for CUDA.
```
