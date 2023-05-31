**To build a Windows container locally, execute these commands. These commands have been tested on Windows PowerShell when run as an administrator:**

```
md C:\tf
md C:\tf\tmp
cd C:\tf
git clone https://github.com/tensorflow/tensorflow.git
cd C:\tf\tensorflow\tensorflow\tools\tf_windows_build_dockerfiles\
# The testing of building the Docker image has been conducted using Python versions <?.?.?> = 3.11.3, 3.10.11, and 3.9.13.
docker build  --no-cache --build-arg PYTHON_VERSION=<?.?.?> -t win-docker .

```

**Starting docker:**

```
#Change to tensorflow root directory
cd C:\tf\tensorflow
docker exec -it tf bash
```
Now we are inside the docker:
follow all options and select the defaults after running `python ./configure.py`

```
ContainerAdministrator@1dd3ffa0732a  /c/tensorflow
python ./configure.py 
exit
```
Running Docker image:

```
docker run --name tf -itd --rm -v C:\tf\tensorflow:C:\tensorflow -v C:\tf\tmp:C:\tmp -e TEST_TMPDIR=C:\tmp -w C:\tensorflow win-docker bash
```
**To create the TensorFlow package builder run:**

```
docker exec tf bazel build //tensorflow/tools/pip_package:build_pip_package
```
