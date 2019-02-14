Running The Micro Speech Example On Eta Compute's ECM3531EVB

This code will enable you to compile and execute the Tensorflow Lite Micro Speech Example on Eta Computes's low power ECM3531 chip.


GETTING STARTED:

1. Download the Tensorflow code from Github and follow instructions there to download other dependencies.  

2. Download the Eta Compute SDK, version 0.0.17.

3. Install the Arm compiler arm-none-eabi-gcc, version = arm-none-eabi-gcc (GNU Tools for Arm Embedded Processors 7-2018-q2-update) 7.3.1 20180622 (release) [ARM/embedded-7-branch revision 261907]

4. Edit the file   tensorflow/lite/experimental/micro/tools/make/targets/ecm3531_makefile.inc  so that the variable ETA_SDK points to the location where the Eta Compute SDK is installed, and the variable GCC_ARM points to the Arm compiler.

5. Compile the code with the command   "make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=ecm3531 test".  This will create the executable tensorflow/lite/experimental/micro/tools/make/gen/ecm3531_cortex-m3/bin/micro_speech_test.

6. Connect the board to the host computer, start PuTTY (Connection type = Serial, Speed = 11520, Data bits = 8, Stop bits = 1,  Parity = None), and load the executable with ocd.  A sample script for loading the image is provided in tensorflow/lite/experimental/micro/tools/make/targets/ecm3531/load_program.  

The following  will be printed on the Uart:

Testing TestInvoke
Ran successfully

/ tests passed
~~~ALL TESTS PASSED~~~



CONTACT INFORMATION:

Contact info@etacompute.com  for more information on obtaining the Eta Compute SDK and evalution board.
