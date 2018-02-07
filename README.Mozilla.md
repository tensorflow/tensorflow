Quick startup:
==============

All work will be done in ~/DeepSpeech/. Please update accordingly.
This has been tested successfully on:
* Ubuntu 14.04 with TITAN X GPU
* Ubuntu 16.04 with K1000M GPU
* Debian Sid (as of september/october 2016) with GTX 970 GPU
* Arch Linux (as of october 2016) with CPU Only

If you use different versions of CUDA/CUDNN please update accordingly below,
CUDA version 8.0 and CUDNN version 5.1 are assumed.

Compute capabilities should cover most usecase. If not, please refer to NVIDIA
website: https://developer.nvidia.com/cuda-gpus and update the list with your
needs.

CUDA/cudnn path should be normalized (i.e., no "../" component) otherwise build
system will not be happy.

You might need an account on NVIDIA's developper area to download cudnn.

# TaskCluster-built PIP packages

If you prefer, you can directly install the PIP package out of TaskCluster
builds. Those are linux 64 bits, python 2.7 builds, and linked against CUDA 8.0,
CuDNN 5.1.3 for the GPU version.

Please refer to the artifacts available at:
* CPU only build: https://tools.taskcluster.net/index/artifacts/#project.deepspeech.tensorflow.pip.master.cpu/project.deepspeech.tensorflow.pip.master.cpu
* GPU enabled build: https://tools.taskcluster.net/index/artifacts/#project.deepspeech.tensorflow.pip.master.gpu/project.deepspeech.tensorflow.pip.master.gpu

Pick the URL to the "tensorflow-VERSION-py2-none-any.whl" and pass it to PIP:
```
$ pip install --upgrade URL
```

For the current version being built, 1.3.0, this should also be a stable
and direct link:
* Linux/AMD64 CPU: https://index.taskcluster.net/v1/task/project.deepspeech.tensorflow.pip.master.cpu/artifacts/public/tensorflow_warpctc-1.3.0rc0-cp27-cp27mu-linux_x86_64.whl
* Linux/AMD64 GPU: https://index.taskcluster.net/v1/task/project.deepspeech.tensorflow.pip.master.gpu/artifacts/public/tensorflow_gpu_warpctc-1.3.0rc0-cp27-cp27mu-linux_x86_64.whl
* OSX 10.12/AMD64 CPU: https://index.taskcluster.net/v1/task/project.deepspeech.tensorflow.pip.master.osx/artifacts/public/tensorflow_warpctc-1.3.0rc0-cp27-cp27m-macosx_10_12_x86_64.whl
* Linux/ARMv6 CPU: https://index.taskcluster.net/v1/task/project.deepspeech.tensorflow.pip.master.arm/artifacts/public/libtensorflow_cc.so

# System setup

* install bazel in ~/DeepSpeech/bin/:
 * just use the shell script installer at https://github.com/bazelbuild/bazel/releases
 * this will require java8 either install from your distro or from bazel packages
* install CUDA in ~/DeepSpeech/CUDA/ from https://developer.nvidia.com/cuda-toolkit-archive
 * ``PERL5LIB=. sh cuda_VERSION_linux.run --silent --verbose --override --toolkit --toolkitpath=$HOME/DeepSpeech/CUDA/``
* install cudnn in ~/DeepSpeech/CUDA/ from https://developer.nvidia.com/rdp/cudnn-download
 * ``tar xvf cudnn-CUDA_VERSION-linux-x64-CUDNN_VERSION.tgz --strip-components=1 -C $HOME/DeepSpeech/CUDA/``
* ``cd DeepSpeech && virtualenv tf-venv && tf-venv/bin/pip install numpy scipy python_speech_features && cd ..``
* You will also need to install SWIG 3
* ``git clone https://github.com/mozilla/tensorflow --branch warpctc && cd tensorflow``
* For ld, either:
 * ``export LD_LIBRARY_PATH=$HOME/DeepSpeech/CUDA/lib64/:$LD_LIBRARY_PATH`` in your ~/.profile or others
 * ``echo "$HOME/DeepSpeech/CUDA/lib64/" > /etc/ld.so.conf.d/cuda.conf && ldconfig`` (as root)
* On recent systems, GCC 4.9 might be needed, since TensorFlow will not build with GCC 5/6

# Extra changes

If building on some recent systems, you might get strange errors about ``<string>``
during the build. It seems to be related to recent GCC. Please apply the following patch:
```
diff --git a/third_party/gpus/crosstool/CROSSTOOL.tpl b/third_party/gpus/crosstool/CROSSTOOL.tpl
index 3ce6b74..be726ff 100644
--- a/third_party/gpus/crosstool/CROSSTOOL.tpl
+++ b/third_party/gpus/crosstool/CROSSTOOL.tpl
@@ -54,6 +54,7 @@ toolchain {
   # Use "-std=c++11" for nvcc. For consistency, force both the host compiler
   # and the device compiler to use "-std=c++11".
   cxx_flag: "-std=c++11"
+  cxx_flag: "-D_FORCE_INLINES"
   linker_flag: "-lstdc++"
   linker_flag: "-B/usr/bin/"
```

# Configure step

NB: On recent distros, please install older GCC (4.9) and change GCC_HOST_COMPILER_PATH=/usr/bin/gcc to GCC_HOST_COMPILER_PATH=/usr/bin/gcc-4.9

## CPU Only:
* ``echo "" | PATH=$HOME/DeepSpeech/bin/:$PATH PYTHON_BIN_PATH=$HOME/DeepSpeech/tf-venv/bin/python PYTHONPATH=$HOME/DeepSpeech/tf-venv/lib/python2.7/ TF_NEED_GCP=0 TF_NEED_HDFS=0 GCC_HOST_COMPILER_PATH=/usr/bin/gcc TF_NEED_CUDA=0 TF_NEED_OPENCL=0 ./configure``

## GPU Support:
* ``echo "" | PATH=$HOME/DeepSpeech/bin/:$PATH PYTHON_BIN_PATH=$HOME/DeepSpeech/tf-venv/bin/python PYTHONPATH=$HOME/DeepSpeech/tf-venv/lib/python2.7/ TF_NEED_GCP=0 TF_NEED_HDFS=0 GCC_HOST_COMPILER_PATH=/usr/bin/gcc TF_NEED_CUDA=1 TF_NEED_OPENCL=0 TF_CUDA_VERSION=8.0 TF_CUDNN_VERSION=5.1 CUDA_TOOLKIT_PATH=$HOME/DeepSpeech/CUDA CUDNN_INSTALL_PATH=$HOME/DeepSpeech/CUDA TF_CUDA_COMPUTE_CAPABILITIES="3.0,3.5,3.7,5.2,6.0,6.1" ./configure``

The configure step will take some time as it downloads a lot of dependencies.

# Build/package step

Some people reported that building with 8GiB of RAM was not sufficient and at
least 12GiB would be needed, otherwise resulting in OOM.

## CPU Only:
* ``PATH=$HOME/DeepSpeech/bin/:$PATH bazel build -c opt //tensorflow/tools/pip_package:build_pip_package && ./tensorflow/tools/pip_package/build_pip_package.sh /tmp/tensorflow_pkg/``

## GPU Support:
* ``PATH=$HOME/DeepSpeech/bin/:$PATH bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package && ./tensorflow/tools/pip_package/build_pip_package.sh /tmp/tensorflow_pkg/``

# Install step
And then just install the pip package:
* ``cd .. && tf-venv/bin/pip install --upgrade /tmp/tensorflow_pkg/tensorflow-*.whl``

From there, you can jump into the virtual env and just use your tensorflow
package as expected:
* ``source tf-venv/bin/activate``

# Notebook
To make use of the notebook, you should also install Jupyter in the virtualenv,
which, assuming you sourced the venv, is:
* pip install jupyter

And then just:
* jupyter-notebook path/to/DeepSpeech.ipynb
