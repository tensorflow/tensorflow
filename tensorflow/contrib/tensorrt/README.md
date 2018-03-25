# Using TensorRT in TensorFlow


This module provides necessary bindings and introduces TRT_engine_op
operator that wraps a subgraph in TensorRT. This is still a work in progress
but should be useable with most common graphs.

## Compilation


In order to compile the module, you need to have a local TensorRT
installation ( libnvinfer.so and respective include files ). During the
configuration step, TensorRT should be enabled and installation path
should be set. If installed through package managers (deb,rpm),
configure script should find the necessary components from the system
automatically. If installed from tar packages, user has to set path to
location where the library is installed during configuration.

```shell
bazel build --config=cuda --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/
```

After the installation of tensorflow package, TensorRT transformation
will be available. An example use can be found in test/test_tftrt.py script

## Installing TensorRT 3.0.4

In order to make use of TensorRT integration, you will need a local installation of TensorRT 3.0.4 from the [NVIDIA Developer website](https://developer.nvidia.com/tensorrt). Due to compiler compatibility, you will need to download and install the TensorRT 3.0.4 tarball for _Ubuntu 14.04_, i.e., **_TensorRT-3.0.4.Ubuntu-14.04.5.x86_64.cuda-9.0.cudnn7.0-tar.gz_**, even if you are using Ubuntu 16.04 or later.

### Preparing TensorRT installation

Once you have downloaded TensorRT-3.0.4.Ubuntu-14.04.5.x86_64.cuda-9.0.cudnn7.0-tar.gz, you will need to unpack it to an installation directory, which will be referred to as <install_dir>. Please replace <install_dir> with the full path of actual installation directory you choose in commands below.

```shell
cd <install_dir> && tar -zxf /path/to/TensorRT-3.0.4.Ubuntu-14.04.5.x86_64.cuda-9.0.cudnn7.0-tar.gz
```

After unpacking the binaries, you have several options to use them:

#### To run TensorFlow as a user without superuser privileges

For a regular user without any sudo rights, you should add TensorRT to your `$LD_LIBRARY_PATH`:

  ```shell
   export LD_LIBRARY_PATH=<install_dir>/TensorRT-3.0.4/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```

Then you are ready to use TensorFlow-TensorRT integration. `$LD_LIBRARY_PATH` must contain the path to TensorRT installation for TensorFlow-TensorRT integration to work. If you are using a VirtualEnv-like setup, you can add the command above to your `bin/activate` script or to your `.bashrc` script.

#### To run TensorFlow as a superuser

 When running as a superuser, such as in a container or via sudo, the `$LD_LIBRARY_PATH` approach above may not work. The following is preferred when the user has superuser privileges:

  ```shell
  echo "<install_dir>/TensorRT-3.0.4/lib" | sudo tee /etc/ld.so.conf.d/tensorrt304.conf && sudo ldconfig
  ```

  Please ensure that any existing deb package installation of TensorRT is removed before following these instructions to avoid package conflicts.