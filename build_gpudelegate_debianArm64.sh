#!/bin/bash

echo -e "\e[1;32;49m
----------------------------------------------------
this script installs bazel and compiles tensorflow
lite with OpenGLES support on debian arm64
you'll need several GB space and this can take over
an hour to finish!
----------------------------------------------------
\e[0m"

echo -e "\e[1;32;49m
installing dependencies...
\e[0m"
sudo apt update
sudo apt install -y apt-transport-https wget curl gnupg cmake build-essential git unzip
sudo apt install -y mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
sudo apt install -y python python3 python3-pip
pip3 install numpy

echo -e "\e[1;32;49m
downloading tensorflow sources...
\e[0m"
cd ~
export TENSORFLOW_VER=master
export TENSORFLOW_DIR=`pwd`/tensorflow_${TENSORFLOW_VER}

git clone -b ${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR} || echo "git folder exists..."

echo -e "\e[1;32;49m
installing bazel, to remove do 'sudo rm -f /usr/local/bin/bazel'
\e[0m"
cd ${TENSORFLOW_DIR}
mkdir bazel || echo "folder exists..."
cd bazel

if [[ ! -f "bazel-5.2.0-linux-arm64" ]]; then
    wget https://github.com/bazelbuild/bazel/releases/download/5.2.0/bazel-5.2.0-linux-arm64
else
    echo "not downloading bazel, exists already..."
fi

chmod +x ./bazel-5.2.0-linux-arm64
sudo cp ./bazel-5.2.0-linux-arm64 /usr/local/bin/bazel
cd ..

echo -e "\e[1;32;49m
----------------------------------------------------
 (configure) press ENTER-KEY several times.         
----------------------------------------------------
\e[0m"
python3 configure.py

echo -e "\e[1;32;49m
building dependencies...
\e[0m"
mkdir external || echo "folder exists..."
cd external
cmake -j4 ../tensorflow/lite -DCMAKE_FIND_DEBUG_MODE=1

# clean up bazel cache, just in case.
cd ${TENSORFLOW_DIR}
bazel clean

echo -e "\e[1;32;49m
patching build files to fix not linking EGL / GLESv2...
\e[0m"
sed -i "s#conditions:default\": \[\],#conditions:default\": \[\"-lEGL\",\"-lGLESv2\",\],#g" ${TENSORFLOW_DIR}/tensorflow/lite/delegates/gpu/build_defs.bzl

bazel build -s --local_ram_resources=HOST_RAM*0.75 --local_cpu_resources=HOST_CPUS*0.5 -c opt tensorflow/lite:libtensorflowlite.so
bazel build --local_ram_resources=HOST_RAM*0.75 --local_cpu_resources=HOST_CPUS*0.5 --config=native_arch_linux -c opt --copt "-DEGL_NO_X11" --copt="-DMESA_EGL_NO_X11_HEADERS" --copt "-DTFLITE_GPU_BINARY_RELEASE" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

echo -e "\e[1;32;49m
----------------------------------------------------
 build success...         
----------------------------------------------------
\e[0m"

cd ${TENSORFLOW_DIR}
ls -l bazel-bin/tensorflow/lite/
ls -l bazel-bin/tensorflow/lite/delegates/gpu/

echo -e "\e[1;32;49m
----------------------------------------------------
 copying the libraries to ~/lib/tf_${TENSORFLOW_VER}...         
----------------------------------------------------
\e[0m"

user=$(whoami)
sudo mkdir -p ~/lib/tf_${TENSORFLOW_VER}
sudo cp bazel-bin/tensorflow/lite/libtensorflowlite.so ~/lib/tf_${TENSORFLOW_VER}
sudo cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ~/lib/tf_${TENSORFLOW_VER}
sudo chown -R $user: ~/lib/tf_${TENSORFLOW_VER}