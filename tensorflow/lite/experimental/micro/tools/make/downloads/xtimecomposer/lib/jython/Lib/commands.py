"""Execute shell commands via os.popen() and return status, output.

Interface summary:

       import commands

       outtext = commands.getoutput(cmd)
       (exitstatus, outtext) = commands.getstatusoutput(cmd)
       outtext = commands.getstatus(file)  # returns output of "ls -ld file"

A trailing newline is removed from the output string.

Encapsulates the basic operation:

      pipe = os.popen('{ ' + cmd + '; } 2>&1', 'r')
      text = pipe.read()
      sts = pipe.close()

 [Note:  it would be nice to add functions to interpret the exit status.]
"""

__all__ = ["getstatusoutput","getoutput","getstatus"]

# Module 'commands'
#
# Various tools for executing commands and looking at their output and status.
#
# NB This only works (and is only relevant) for UNIX.


# Get 'ls -l' status for an object into a string
#
def getstatus(file):
    """Return output of "ls -ld <file>" in a string."""
    return getoutput('ls -ld' + mkarg(file))


# Get the output from a shell command into a string.
# The exit status is ignored; a trailing newline is stripped.
# Assume the command will work with '{ ... ; } 2>&1' around it..
#
def getoutput(cmd):
    """Return output (stdout or stderr) of executing cmd in a shell."""
    return getstatusoutput(cmd)[1]


# Ditto but preserving the exit status.
# Returns a pair (sts, output)
#
def getstatusoutput(cmd):
    """Return (status, output) of executing cmd in a shell."""
    import os
    pipe = os.popen('{ ' + cmd + '; } 2>&1', 'r')
    text = pipe.read()
    sts = pipe.close()
    if sts is None: sts = 0
    if text[-1:] == '\n': text = text[:-1]
    return sts, text


# Make command argument from directory and pathname (prefix space, add quotes).
#
def mk2arg(head, x):
    import os
    return mkarg(os.path.join(head, x))


# Make a shell command argument from a string.
# Return a string beginning with a space followed by a shell-quoted
# version of the argument.
# Two strategies: enclose in single quotes if it contains none;
# otherwise, enclose in double quotes and prefix quotable characters
# with backslash.
#
def mkarg(x):
    if '\'' not in x:
        return ' \'' + x + '\''
    s = ' "'
    for c in x:
        if c in '\\$"`':
            s = s + '\\'
        s = s + c
    s = s + '"'
    return s
