To build a Windows container locally, execute these commands. These commands have been tested on Windows PowerShell when run as an administrator:

```
md C:\tf
md C:\tf\tmp
cd C:\tf
git clone https://github.com/tensorflow/tensorflow.git
cd C:\tf\tensorflow\tensorflow\tools\tf_windows_build_dockerfiles\
# The testing of building the Docker image has been conducted using Python versions <?.?.?> = 3.11.3, 3.10.11, and 3.9.13.
docker build  --no-cache --build-arg PYTHON_VERSION=<?.?.?> -t win-docker .

```

Running Docker image:
```
docker run --name tf -itd --rm -v C:\tf\tensorflow:C:\tensorflow -v C:\tf\tmp:C:\tmp -e TEST_TMPDIR=C:\tmp -w C:\tensorflow win-docker bash
docker exec -it tf bash
```

Now we are inside the docker:
```
ContainerAdministrator@1dd3ffa0732a  /c/tensorflow
python ./configure.py
# follow all options and select the defaults after running `python ./configure.py` 
exit
```

Running Test:
```
docker exec tf export  PYTHON_DIRECTORY=Python
docker exec tf export  IS_NIGHTLY=1
docker exec tf tensorflow/tools/ci_build/windows/cpu/pip/run.bat --extra_test_flags "--test_env=TF2_BEHAVIOR=1"
```




