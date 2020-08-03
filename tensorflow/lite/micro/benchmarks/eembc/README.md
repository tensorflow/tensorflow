This document describes the EEMBC test-harness extension for TFLite Micro.

# What is a "Test Harness"

A test harness is a thin wrapper of _instrumentation code_ into which is placed _test code_ to be isolated and analyzed. It is a "petri dish" of sorts for algorithms: it provides a way for an external observer to control the stimulus and measure the response in order to characterize the device across many different metrics. Generally, the test harness connects to an external host system which coordinates the testing and measurement, but it may be self-hosted (i.e., not requiring external apparatus).

The thin wrapper defines the _test-harness API_, which has a few common characteristics for each benchmark (such as initialization), but often the API is tied to the behavior of the benchmark itself. In the case of the test harness by EEMBC, you will see a `monitor` code that provides the most basic functionality, and the `profile` code which is specific to the behavior.

## Repeatability

Benchmarks that aren't repeatable are useless. One way a test harness helps with this is by providing strict control over the inputs and state of the device. The inputs must be identical for every _device under test_ (a.k.a., DUT), otherwise one platform may perform more work than another, which may skew results. The test harness must also provide a means to configure the DUT so that each platform starts from the same state.

## Connectivity

The test harness provides a means to send and receive data to and from the DUT, often through an external serial port. Furthermore, if the code under test is sensitive to (in this case) the UART, the test harness provides hooks to enable and disable test functionality before entering, and after exiting, critical code (in this harness: `th_pre()` and `th_post()`. Why? If the benchmark is measuring a low-power state, it would not make sense to have the GPIO clocks and UART running during the measured portion of the test, consuming lots of energy!

## Precision Measurement

Benchmarks aren't just about speed: sometimes something other than milliseconds needs to be measured, such as power or energy. Often the test harness is just a component in a much larger framework, which they capability perform high-speed timing, or triggering logic- or real-time-spectrum analyzers, or energy monitors. Typically, a test harness that requires external components will require at least one GPIO API call that can be used in the benchmark. This test harness is designed to work with the IoTConnect Framework by EEMBC, which has a host UI application for each benchmark. However, at this time (Summer 2020) the host UI is still under development.

## Trust

The test harness also provides a means to verify the results of the DUT. This provides a first-order verification of results without a 3rd party.

Simply measuring the amount of time it takes to run an algorithm might be suitable for individual analysis, but when comparing results from competitors, there needs to be a mechanism to establish trust. Sometimes the mechanism is a simple honor system, other times measurements are performed by a 3rd-part test lab, such as Underwriter's Labs, Consumer Reports, or the EEMBC certification program.

The test harness attempts to provide a way for the host UI application to perform much self-checking and verification as possible to defeat cheating without 3rd party certification.

# The EEMBC Test Harness for TFLite Micro

The test harness provided here by EEMBC utilizes just the most basic components for external communication: UART Tx, Rx, and a GPIO. This allows control over execution and input data. Currently, this is all that the EEMBC IoTConnect framework has required. However, the benchmark is still under development, and may require a more sophisticated API going forward. This has not yet been decided on by the consortium's tinyML working group.

A set of monitor commands allow control over loop execution and input dataset. Each command ends with a '%'.

For example, when connecting to a terminal application and booting the device, one would see:
~~~
m-timestamp-mode-energy
m-init-done
m-ready
~~~

And issueing `help%` would print this, list of commands.

~~~
ULPMark-ML Firmware V0.0.1

help         : Print this information
loop         : first-stab at measurement
buff SUBCMD  : Input buffer functions
  on         : Infer from buffer data
  off        : Infer from default data
  load N     : Allocate N bytes and set load counter
  i16 N [N]+ : Load 16-bit signed decimal integer(s)
  x8 N [N]+  : Load 8-bit hex byte(s)
  print i16  : Print buffer as 16-bit signed decimal
  print x8   : Print buffer as 8-bit hex
m-ready
~~~

## Implementation

The basic harness is found under `tensorflow/lite/micro/benchmarks/eembc`. Files and functions that begin with `ee_` may not be altered under any circumstances. Files and functions that begin with `th_` must be implemented, as they often require calls to hardware-specific SDK, such as configuring a UART to receive bytes.

Two profiles have been implemented using the `benchmarks/Makefile.inc` rules:

### Person Detection (target `person_detection_eembc_th`)

Building this target installs the harness, replacing `main()` with a UART parser, and overrides the functions `GetImage` and `RespondToDetection`. This allows the host framework to download new test data on-the-fly during benchmarking.

### Micro Speech (target `micro_speech_eembc_th`)

Building this target installs the harness, replacing `main()` with a UART parser, and overrides the functions `GetAudioSamples`, `LatestAudioTimestamp`, and `RespondToCommand`. This allows the host framework to download new test data on-the-fly during benchmarking.

## Porting

The porting process for this benchmark on TFLite Micro is not particularly complex. Simply provide the file `tensorflow/lite/micro/<TARGET>/eembc/monitor/th_api/th_lib.cc` and `th_libc.cc`. The bare-bones file is included in `tensorflow/lite/micro/benchmarks/eembc/monitor/th_api`. The three key functions that need to be implemented are:

1. `th_printf` - often this can be implemented with vsprintf (or even #defined as printf) since retargeting STDIO is very common these days. Otherwise you will need to provide your own per-character UART function and initialize it during `th_monitor_initialize`.

2. `th_check_serial` - this responds to a new character(s) in the UART rx buffer and feeds it to `ee_serial_callback` one character at a time.

3. `th_timestamp` - this must generate an open-drain falling edge with 1us hold time

Lastly, `th_serialport_initialize` may need to be modified to make sure transmission is done at 9600 baud 8N1. This is a requirement of the EEMBC IoTConnect framework.

An example port is provided in `tensorflow/lite/micro/ecm3532`.

# Changes from Standard EEMBC Harness and TODOs

1. To avoid C-to-C++ linkage hassles, all harness files end with ".cc"

2. Full paths to include files are used due to the way the TFLite Micro makefiles build targets.

# What is EEMBC?

Founded in 1997, EEMBC is US non-profit which develops  benchmarks for the hardware and software used in autonomous driving, mobile imaging, the Internet of Things, mobile devices, and many other applications. The EEMBC community includes member companies, commercial licensees, and academic licensees at institutions of higher learning around the world. Visit our [website](https://www.eembc.org).