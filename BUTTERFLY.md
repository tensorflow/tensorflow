### Butterfly specific changes to tensorflow build.

#### Prerequisites

You'll need to install the following libraries:

```bash
$ brew install autoconf
$ brew install automake
$ brew install libtool
```

#### Background 

We are currently building our own tensorflow binary since by default the binary for mobiles do not include some of the ops required by our models.
Tensorflow for ios specifically is compiled using the script 

```
./tensorflow/contrib/makefile/build_all_ios.sh
```
That scipts calls `tensorflow/contrib/makefile/Makefile` which sets the macro `__ANDROID_TYPES_SLIM__` which is used extensively when defining ops.

An op can be excluded from the final mobile binary files for several reasons:

- The op was not registered in `tensorflow/contrib/makefile/tf_op_files.txt`. 
  In that case all we do is add it. e.g.
  ```
  tensorflow/core/kernels/cwise_op_atan2.cc
  ```
  
- For mobile the op is usually registered only for the first type in the list available at the time of registration. 
  e.g. In the follwing case:
```
REGISTER7(BinaryOp, CPU, "Pow", functor::pow, float, Eigen::half, double, int32,
          int64, complex64, complex128);
```

the only operation that will be available on mobile will be "Pow" with type `float` because `float` is the first type mentioned out of 7 types (float, Eigen::half, double, int32,
          int64, complex64, complex128). In case our model needs `int32` we will need to modify the code and specifically add a line that will registered that op for the `int32` type e.g.

```
REGISTER(BinaryOp, CPU, "Pow", functor::pow, int32);
```

You can see an example in the original tensorflow codebase here:
```
https://github.com/tensorflow/tensorflow/blob/79e65acb81f750ffa88b366c566646d48d16c574/tensorflow/core/kernels/cwise_op_mul_1.cc#L23
```


- The op is registered but not available in the binary. 
In that case we need to modify the Makefile to compile the source code for the op.
e.g. We could add the following line in the Makefile in the place where sources are defined to include source code for image_ops._
```
$(wildcard tensorflow/contrib/image/kernels/ops/*.cc)
```

#### How to build

```
sh ./tensorflow/contrib/makefile/build_all_ios.sh
```

```
sh pack_for_bni.sh
```

Change the version in `TensorflowPod.podspec` and create a release.

### Android (WIP)

The script for building android is colocated with the ios script:
```
./tensorflow/tensorflow/contrib/makefile/build_all_android.sh*
```
This script uses `armeabi-v7a` as the default architecture.

#### NDK

For android, you need to install the NDK. This is the toolchain used for cross compilation. There are a number 
of ways to do this.  The easiest is to install Android Studio and to then fuss with 

`Preferences for New Projects:: Appearance & Behavior > System Settings > Android SDK`

Under the covers this installs a command line tool, the `sdkmanager`, that is used to manage all installed sdk
components.  If you can't use the GUI (say in docker), you can use this directly as follows:

`~/Library/Android/sdk/tools/bin$ ./sdkmanager --list `

The available versions of the $NDK$ with this tool are:
```
ndk;16.1.4479499
ndk;17.2.4988734
ndk;18.1.5063045
ndk;19.2.5345600
ndk;20.0.5594570
ndk;20.1.5948944
ndk;21.0.6113669
```
In instances where you don't want to install the Android SDK, you can install the Android command line tools 
directly using  
```
wget https://dl.google.com/android/repository/commandlinetools-linux-6200805_latest.zip
```

Finally, if you need an older version of the ndk, you can get these 
from [here](https://developer.android.com/ndk/downloads/older_releases).  For instance:
```
wget https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip
```
Now the funky thing about the olders NDKs (<19) is that they are not deployed as standalone toolchains; 
the system roots and structure of includes in all of them are not
configured (e.g. the compiler can't find <stdio.h>). You can use `$NDK_ROOT/build/toolsmake_standalone_toolchain.py`
to configure, but this is a rabbit hole you don't need to go down - our makefiles expect an unconfigured $NDK_ROOT.

#### Butterfly/Tensorflow (~1.13)
 
Because our fork is an older version of 
tensorflow, we need to select a compatible $NDK$ version. For our branch, the candidate versions
are described 
[here](https://github.com/ButterflyNetwork/tensorflow/blob/5f94511e57d55d6fbe840f117b8fec3f77f6aa44/configure.py#L46)
```
_SUPPORTED_ANDROID_NDK_VERSIONS = [10, 11, 12, 13, 14, 15, 16, 17, 18]
```
Now you'd think any of these would work.  But I tried versions 18, 17, 16; these
 all resulted in failures early during compilation of the internal protobuf
 target (`tensorflow/contrib/makefile/compile_android_protobuf.sh -c`) that precedes 
compilation of tensorflow proper.

Seems each version of the NDK is slightly different, and the makefiles are very brittle - they expect to find headers
and libraries in very specific places.

Now version 15 of the `NDK` seems to get much further along, so focused on this for a bit. 
On OSX this seems to barf fairly early, so I tried to run on a docker ubuntu:16.04 container using the following:

       apt update
       apt upgrade
       apt-get install build-essential
       apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev git python
       apt install wget
       wget https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip
       unzip android-ndk-r15c-linux-x86_64.zip
       export NDK_ROOT=/android-ndk-r15c
       git clone https://github.com/ButterflyNetwork/tensorflow.git
       cd tensorflow
       ./tensorflow/contrib/makefile/build_all_android.sh

But after a while, we then fail on this: 
```
./tensorflow/contrib/image/kernels/image_ops.h:93:51: error: 'round' is not a member of 'std'
     return read_with_fill_value(batch, DenseIndex(std::round(y)),
```
This `std::round(y)` issue seems to be a known 
[error](https://github.com/tensorflow/tensorflow/issues/24358#issuecomment-447202118) with
older NDKs and GNU tools in general (it's a bug)

So I found all the code the uses std::round(), and replaced it with round() (thanks emacs macros!).  About 16 source
files were affected. 

Compilation then proceeded for a while.  Finally barfing on 
```
arm-linux-androideabi-g++: internal compiler error: Killed (program cc1plus)
Please submit a full bug report,
with preprocessed source if appropriate.
See <http://source.android.com/source/report-bugs.html> for instructions.
```
This seems to arise from an out of memory exception.

So I spun up a new `dl-android` node  on an m5a.4xlarge instance running ubuntu 16.04 and set up as follows:
```
    1  sudo apt update
    2  sudo apt upgrade
    3  sudo apt-get install build-essential
    5  sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev git python
    7  wget https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip
    8  sudo apt install emacs
   10  unzip android-ndk-r15c-linux-x86_64.zip 
   12  export NDK_ROOT=/home/ubuntu/android-ndk-r15c
   13  git clone git@github.com:ButterflyNetwork/tensorflow.git
   19  emacs -nw `find . -type f | xargs grep -l std::round\( 2>/dev/null`
   26  cd tensorflow/
   29  ./tensorflow/contrib/makefile/build_all_android.sh 
```
Compilation finished!  

```
ubuntu@ip-10-0-6-111:~/tensorflow$ find ./tensorflow/contrib/makefile/gen  -name "*.a" | grep -v protobuf-host
./tensorflow/contrib/makefile/gen/lib/android_armeabi-v7a/libtensorflow-core.a
./tensorflow/contrib/makefile/gen/protobuf_android/armeabi-v7a/lib/libprotobuf-lite.a
./tensorflow/contrib/makefile/gen/protobuf_android/armeabi-v7a/lib/libprotobuf.a
./tensorflow/contrib/makefile/gen/protobuf_android/armeabi-v7a/lib/libprotoc.a
```
Not sure if these are complete - In the iOS world, we used the script pack_for_bni.sh. This seems to rely on 
`libprotobuf-lite.a libprotobuf.a libtensorflow-core.a nsync.a`.  We build the latter too, but it's not under the `gen`
output.

