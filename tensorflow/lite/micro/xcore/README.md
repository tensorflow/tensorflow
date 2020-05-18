# Quickstart to install tools and run unit tests:

```
$ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" clean clean_downloads && make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test_greedy_memory_planner_test || true && pushd tensorflow/lite/micro/tools/make/downloads/xtimecomposer/xTIMEcomposer/15.0.0/ && source SetEnv && popd  && make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test
```

(add -jN to the final make command to run builds / tests in N parallel threads)

# Background information:

*   To start from a fresh repo (this will also remove non-xcore builds and
    downloads): `$ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" clean clean_downloads`
*   To force xcore.ai tools download from a clean repo: `$ make -f
    tensorflow/lite/micro/tools/make/Makefile TARGET="xcore"
    test_greedy_memory_planner_test` (this will fail to build the test, but if
    it succeeds because you already have tools it will exit quickly)

*   To set up environment variables correctly run the following from the top
    tensorflow directory: `$ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" test $ pushd
    ./tensorflow/lite/micro/tools/make/downloads/xtimecomposer/xTIMEcomposer/15.0.0/
    && source SetEnv && popd $ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" test`

*   Assuming tools are already set up the following are the most commonly used
    commands: `$ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" build $ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" test $ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" < name_of_example i.e. hello_world_test >`
