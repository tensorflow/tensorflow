To build a Windows container locally, execute these commands. These commands have been tested on Windows PowerShell when run as an administrator:

```
# Enable long path support on main Windows server
reg.exe add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f
md C:\tmp
cp docker_win.bazelrc C:/tmp/docker_win.bazelrc
cd C:\tmp
git clone https://github.com/tensorflow/tensorflow.git
cd C:\tensorflow\tensorflow\tools\tf_windows_build_dockerfiles\
# The testing of building the Docker image has been conducted using Python versions <?.?.?> = 3.11.3, 3.10.11, and 3.9.13.
docker build  --no-cache --build-arg PYTHON_VERSION=<?.?.?> -t win-docker .

```

Running Docker image:
```
docker run --name tf -itd --rm --env TF_PYTHON_VERSION=3.9 --env TMPDIR=C:/tmp --env TMP=C:/tmp --env TEMP=C:/tmp -v C:\tmp\tensorflow:C:\workspace -v C:\tmp:C:\tmp -e TEST_TMPDIR=C:\tmp -w C:\workspace win-docker bash
```

Now we are inside the docker:
```
ContainerAdministrator@1dd3ffa0732a  /c/tensorflow
python ./configure.py
# follow all options and select the defaults after running `python ./configure.py` 
exit
```

Running pip Test:
```
docker exec tf export  PYTHON_DIRECTORY=Python
docker exec tf export  IS_NIGHTLY=1
docker exec tf tensorflow/tools/ci_build/windows/cpu/pip/run.bat --extra_test_flags "--test_env=TF2_BEHAVIOR=1"
```

Running non-pip Test:
```
docker exec tf bazel --output_user_root=C:/tmp --bazelrc=C:/tmp/docker_win.bazelrc test  --jobs=150 --flaky_test_attempts=5 --config=docker_win --config=docker_win_py --config=docker_win_py39 --dynamic_mode=off --config=xla --config=short_logs --announce_rc --build_tag_filters=-no_windows,-windows_excluded,-no_oss,-oss_excluded --build_tests_only --config=monolithic --keep_going --test_output=errors --test_tag_filters=-no_windows,-windows_excluded,-no_oss,-oss_excluded,-gpu,-tpu --test_size_filters=small,medium --test_timeout="300,450,1200,3600" --verbose_failures --copt=/d2ReducedOptimizeHugeFunctions --host_copt=/d2ReducedOptimizeHugeFunctions -- //tensorflow/... -//tensorflow/java/... -//tensorflow/lite/... -//tensorflow/compiler/xla/python/tpu_driver/... -//tensorflow/compiler/...
```


