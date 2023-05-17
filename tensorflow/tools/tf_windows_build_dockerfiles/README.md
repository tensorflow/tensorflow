**To build a Windows container locally, execute these commands. These commands have been tested on Windows PowerShell when run as an administrator:**
```
md C:\tf
md C:\tf\tmp
cd C:\tf
git clone https://github.com/Bobarshad/tensorflow.git
cd C:\tf\tensorflow\tensorflow\tools\tf_windows_build_dockerfiles\
docker build . -f <Dockerfile 3.9 or 3.11> -t win-docker
```

**Starting docker:**
```
docker run --name tf -itd --rm -v C:\tf\tensorflow:C:\workspace -v C:\tf\tmp:C:\tmp -e TEST_TMPDIR=C:\tmp -w C:\workspace win-docker bash
```

**To build TensorFlow pip_package:**
```
#Change to tensorlflow root directory
cd C:\tf\tensorflow
# When configuring TensorFlow, choose all the default options without making any changes.
docker exec tf python ./configure.py
docker exec tf bazel build //tensorflow/tools/pip_package:build_pip_package
```
