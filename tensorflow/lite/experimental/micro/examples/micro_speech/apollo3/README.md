# Description of Apollo3 Makefile targets

* **pushbutton_cmsis_speech_test_bin**: 
    * When users press BTN2 on the Apollo3 EVK, 1 second of audio is captured. 
    * Then the audio is sent to the CMSIS version of the preprocessor and into the neural net
    * To print out the neural net's inference scores, run GDB and source pushbutton\_cmsis\_scores.cmd
    * To save the captured audio to a text file (captured\_data.txt), run GDB and source pushbutton\_cmsis\_voice.cmd
    * Setup python
        * sudo apt install python-pip
        * sudo apt install python-tk
        * pip install numpy
        * pip install matplotlib
        * pip install pysoundfile
        * python captured_data_to_wav.py
    * captured\_data.txt can be turned into a \*.wav file using captured\_data\_to\_wav.py by executing "python captured\_data\_to\_wav.py"
* **preprocessor_1k_cmsis_test_bin**: 
    * Sends a 1 kHz sine wave to the CMSIS fixed\_point version of the preprocessor
    * **This test should be compiled with the -O0 option.** Otherwise, the breakpoints will not be reached
        * In tensorflow/lite/experimental/micro/tools/make/targets/apollo3evb_makefile.inc change "-O3" to "-O0" on line 47
        * **DO NOT FORGET TO REVERT CHANGE AFTER EXPERIMENT**
        * In future, enhance scripts to handle automatically, NOT manually!
    * Clean project by running "make -f tensorflow/lite/experimental/micro/tools/make/Makefile clean"
    * Compile BIN by running "make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=apollo3evb preprocessor_1k_cmsis_test_bin"
    * Run with the preprocessor\_1k\_cmsis\_test.cmd GDB command file
    * Produces four text files corresponding to outputs from the CMSIS fixed\_point version of this algorithm:
        * cmsis_windowed_input.txt: the sinusoid after multiplying elementwise with a Hann window
        * cmsis_dft.txt: the DFT of the windowed sinusoid
        * cmsis_power.txt: the magnitude squared of the DFT
        * cmsis_power_avg.txt: the 6-bin average of the magnitude squared of the DFT
    * Run both verisons of the 1KHz pre-processor test and then compare.
        * These files can be plotted with "python compare\_1k.py"
    * Also prints out the number of cycles the code took to execute (using the DWT->CYCCNT register) 
* **preprocessor_1k_micro_test_bin**
    * Sends a 1 kHz sine wave to the Micro-Lite fixed\_point version of the preprocessor
    * **This test should be compiled with the -O0 option.** Otherwise, the breakpoints will not be reached
    * Run with the preprocessor\_1k\_micro\_test.cmd GDB command file
    * Produces four text files corresponding to outputs from the Micro-Lite version of this algorithm:
        * micro_windowed_input.txt: the sinusoid after multiplying elementwise with a Hann window
        * micro_dft.txt: the DFT of the windowed sinusoid
        * micro_power.txt: the magnitude squared of the DFT
        * micro_power_avg.txt: the 6-bin average of the magnitude squared of the DFT
    * Run both verisons of the 1KHz pre-processor test and then compare.
        * These files can be plotted with "python compare\_1k.py"
    * Also prints out the number of cycles the code took to execute (using the DWT->CYCCNT register) 

# Description of files

* **.gitignore**: Git should ignore \*.txt and \*.wav files that result from experiments run in this directory
* **apollo3.h**: Apollo 3 version of the [CMSIS Device Header File (device.h)](https://www.keil.com/pack/doc/CMSIS/Core/html/device_h_pg.html). Available in the [Ambiq Keil Pack](http://s3.ambiqmicro.com/pack/AmbiqMicro.Apollo_DFP.1.1.0.pack).
* **captured\_data\_to\_wav.py**: Python script that parses a text file containing data dumped from GDB (specifically the verilog format) and creates a \*.wav file using [PySoundFile](https://pysoundfile.readthedocs.io/en/0.9.0/).
* **compare\_1k.py**: This script compares the intermediate variables and final outputs of the micro-lite fixed-point preprocessor function and the CMSIS version of this function. The stimulus provided to each preprocessor is the same: a 1 kHz sinusoid.
* **get\_yesno\_data.cmd**: A GDB command file that runs preprocessor_test (where TARGET=apollo3evb) and dumps the calculated data for the "yes" and "no" input wavfeorms to text files
* **\_main.c**: Point of entry for the micro_speech test
* **preprocessor_1k.cc**: A version of preprocessor.cc where a 1 kHz sinusoid is provided as input to the preprocessor
* **preprocessor_1k_cmsis_test.cmd**: GDB command file for the CMSIS preprocessor 1 kHz test
* **preprocessor_1k_micro_test.cmd**: GDB command file for the Micro-Lite preprocessor 1 kHz test
* **preprocessor_test.cmd**: GDB command file for the preprocessor test
* **pushbutton_cmsis_scores.cmd**: GDB command file that runs pushbutton_cmsis_speech_test_bin. It adds a breakpoint immediately after the scores are reported and prints out each score. Then it continues code execution.
* **pushbutton_cmsis_voice.cmd**: GDB command file that runs pushbutton_cmsis_speech_test_bin. Dumps the recorded 1 second of audio to captured_data.txt, which can then be processed by the python file captured_data_to_wav.py.
* **pushbutton_main.c**: Source file containing program point of entry \_main() for the pushbutton\_\* tests. Contains Interrupt Service Routines for PDM data capture and pushbuttons. Calls the main() function of pushbutton_test.cc
* **pushbutton_test.cc**: Source file containing main() function for the pushbutton\_\* tests. main() calls the preprocessor function and the neural net inference function.
* **system_apollo3.c**: Apollo 3 version of the [CMSIS System Configuration File system\_\<device\>.c](https://www.keil.com/pack/doc/CMSIS/Core/html/system_c_pg.html). Available in the [Ambiq Keil Pack](http://s3.ambiqmicro.com/pack/AmbiqMicro.Apollo_DFP.1.1.0.pack).
* **system_apollo3.h**: Apollo 3 version of the [CMSIS System Configuration File system\_\<device\>.h](https://www.keil.com/pack/doc/CMSIS/Core/html/system_c_pg.html). Available in the [Ambiq Keil Pack](http://s3.ambiqmicro.com/pack/AmbiqMicro.Apollo_DFP.1.1.0.pack).


# FFT scaling
See https://github.com/ARM-software/CMSIS_5/issues/220
>And as @xizhizhang pointed, I think there may be an error on the internal downscaling, or at least on the documentation. It looks like during the fft computation, the downscaling factor reach 2**-9 for a 512 rfft operation, being the output in Q10.22, instead the documented 2**-8 and Q9.23.
