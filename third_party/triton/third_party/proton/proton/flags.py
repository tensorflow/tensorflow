"""
This file contains the global flags used in the proton package.
"""

# Whether to enable profiling. Default is False.
profiling_on = False
# Whether the script is run from the command line. Default is False.
command_line = False


def set_profiling_on():
    global profiling_on
    profiling_on = True


def set_profiling_off():
    global profiling_on
    profiling_on = False


def get_profiling_on():
    global profiling_on
    return profiling_on


def set_command_line():
    global command_line
    command_line = True


def is_command_line():
    global command_line
    return command_line
