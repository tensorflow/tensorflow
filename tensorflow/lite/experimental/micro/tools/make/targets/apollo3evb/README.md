Follow these steps to get the preprocessor test working on Apollo 3:

1. Download the SDK to the be at the same level as tensorflow.git
2. Copy and prepare files by running tensorflow/lite/experimental/micro/tools/make/targets/apollo3evb/prep_apollo3_files.sh
3. Recompile libarm_cortexM4lf_math.a with the softfp option, and call it libarm_cartexM4lf_math_softfp.a. The original version was compiled with the hard option, and this caused conflicts with existing software. We might be able to fix this in the future
4. Install Segger JLink tools from https://www.segger.com/downloads/jlink/ 
5. Compile the preprocessor_test_bin project with the following command: make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=apollo3evb VENDORLIB=cmsis-dsp preprocessor_test_bin
6. Download to the target with JFlashLiteExe with the following settings:
    1. Device = AMA3B1KK-KBR
    2. Interface = SWD at 1000 kHz
    3. Data file = tensorflow/lite/experimental/micro/tools/make/gen/apollo3evb_cortex-m4/bin/preprocessor_test.bin
    4. Prog Addr = 0x0000C000
7. Connect to device via serial port (115200 baud) and press reset button. Should see all tests passed. Seeing a discrepance between Windows and Linux testing --> need to debug
