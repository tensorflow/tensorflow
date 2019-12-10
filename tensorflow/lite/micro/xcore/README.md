IMPORTANT: to set up environment variables correctly run the following from the top tensorflow directory:
    $ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test
    $ pushd ./tensorflow/lite/micro/tools/make/downloads/xtimecomposer/ && source SetEnv && popd
    $ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test 

(add -jN to the make command to run builds / tests in N parallel threads)