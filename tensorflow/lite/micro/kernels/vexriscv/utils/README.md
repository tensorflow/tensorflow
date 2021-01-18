# VexRISC-V utils

This directory contains scripts for some utilities when debugging with TFLM
applications.

## log_parser.py

This script is used to analyze the function call stack obtained when a
application is running on Renode with GDB session attached or with Renode's
logging set to true (include the following line in your `*.resc` file of enter
this command in Renode every time you launch an simulator.)

```
sysbus.cpu LogFunctionNames true
```

Also, make sure you check out Antmicro's repo
[antmicro/litex-vexriscv-tensorflow-lite-demo](https://github.com/antmicro/litex-vexriscv-tensorflow-lite-demo)
for the guide to run TensorFlow Lite Micro examples on Renode.

In the following guide with GDB, we will be using the example gdb script
described here
[Cobbled-together Profiler](https://xobs.io/cobbled-together-profiler/)

### Launch Renode console

Include (`include` or `i` for short) the renode script (`*.resc`), DO NOT start
the simulation yet.

The symbol `@` is the path where Renode is installed, you can navigate to
anywhere on the disk as long as you follow the linux syntax (`../` for parent
directory, etc.), here I install Renode under home (`/home/$USER` or `~/`)
directory and I put the demo repository litex-vexriscv-tensorflow-lite-demo
under home directory.

```
i @../litex-vexriscv-tensorflow-lite-demo/renode/litex-vexriscv-tflite.resc
```

### Start GDB server on Renode on port 8833

```
machine StartGdbServer 8833
```

### Launch GDB

First you need to find a proper GDB executable for your target architecture,
here, we will follow Antmicro's repo and use the riscv GDB executable from
zephyr

Usage: `[GDB] -x [GDB_SCRIPT] [TFLM_BINARY]`

Example: `/opt/zephyr-sdk/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gdb \ -x
profiling.gdb \
../tensorflow/tensorflow/lite/micro/tools/make/gen/zephyr_vexriscv_x86_64/magic_wand/build/zephyr/zephyr.elf`

### Connect GDB to Renode's gdbserver on the same port

```
(gdb) target remote :8833
(gdb) monitor start
(gdb) continue

# Run the function in the GDB script with required parameter
(gdb) poor_profile 1000
```

### Interrupt the gdb script regularly with a shell command

```
for i in $(seq 1000); do echo $i...; killall -INT riscv64-zephyr-elf-gdb; sleep 5; done
```

### Interpreting the log

#### Parse and visualize the log

```
# The following command is used to parse and visualize the log file
# obtained from GDB and only keep top 7 most frequent functions in the
# image, for detail usage of the script, please refer to the source code

python log_parser.py [INPUT] --regex=gdb_regex.json --visualize --top=7 --source=gdb
```

Since we are redirecting the gdb interrupt messages to the file
`<path-you-run-gdb>/profile.txt` (see the gdb script,) we can now parse the log
and visualize it. (set the image title with argument `--title`)

```
python log_parser.py profile.txt --regex=gdb_regex.json --visualize --top=7 --title=magic_wand
```

![image](https://user-images.githubusercontent.com/21079720/91764987-198fec00-eb8d-11ea-8eb1-90355fe4f28c.png)

#### Get the statistic of the function call hierarchy

To get a more detail view of how the entire function call stack looks like and
how many time the function is called with the exact same call stack, we can add
another option `--full-trace` to the script and it will generate a `*.json` file
for the complete call stack trace. `python log_parser.py profile.txt
--regex=gdb_regex.json --visualize --top=7 --full-trace`

```
# In the `*.json` file
root
|-- fcn0
|    |-- [stack0, stack1, ...] # List of function call stacks, see below
|
|-- fcn1
|    |-- [stack0, stack1, ...]
...
```

```
# Each stack* object contains the following information
stack*
|-- counts: 5 # Number of occurrences with the exact same call stack
|-- [list of functions in the call stack]
```

![image](https://user-images.githubusercontent.com/21079720/91755189-8bf9cf80-eb7f-11ea-884c-2354f3470271.png)

### Customizing `*.json` used in the script

The regular expression used in this script is configured with a standard
`*.json` file with the following content:

*   `base`: Base regular expression to clean up the log, this is set to clean up
    the ANSI color codes in GDB
*   `custom`: A series of other regular expressions (the script will run them in
    order) to extract the information from the log
