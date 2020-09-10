# Microspeech Example

```
$ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" micro_speech
```

This example is for the **XCOREAI Explorer Board**.

*   To set up environment variables correctly run the following from the top
    tensorflow directory: `$ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" test $ pushd
    ./tensorflow/lite/micro/tools/make/downloads/xtimecomposer/xTIMEcomposer/15.0.0/
    && source SetEnv && popd $ make -f tensorflow/lite/micro/tools/make/Makefile
    TARGET="xcore" test`

*   In addition to setting up the environment variables, copy XMOS dependency libraries to tensorflow/lite/micro/tools/make/downloads:
  - [lib_dsp](https://github.com/xmos/lib_dsp)
  - [lib_logging](https://github.com/xmos/lib_logging)
  - [lib_mic_array](https://github.com/xmos/lib_mic_array)
  - [lib_xassert](https://github.com/xmos/lib_xassert).
