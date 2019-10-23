"""
Implement subshell functionality for Jython.

This is mostly to provide the environ object for the os module,
and subshell execution functionality for os.system and popen* functions.

javashell attempts to determine a suitable command shell for the host
operating system, and uses that shell to determine environment variables
and to provide subshell execution functionality.
"""
from java.lang import System, Runtime
from java.io import File
from java.io import IOException
from java.io import InputStreamReader
from java.io import BufferedReader
from UserDict import UserDict
import jarray
import os
import string
import subprocess
import sys
import types
import warnings
warnings.warn('The javashell module is deprecated. Use the subprocess module.',
              DeprecationWarning, 2)

__all__ = ["shellexecute"]

def __warn( *args ):
    print " ".join( [str( arg ) for arg in args ])

class _ShellEnv:
    """Provide environment derived by spawning a subshell and parsing its
    environment.  Also supports subshell execution functions and provides
    empty environment support for platforms with unknown shell functionality.
    """
    def __init__( self, cmd=None, getEnv=None, keyTransform=None ):
        """Construct _ShellEnv instance.
        cmd: list of exec() arguments required to run a command in
            subshell, or None
        getEnv: shell command to list environment variables, or None.
            deprecated
        keyTransform: normalization function for environment keys,
          such as 'string.upper', or None. deprecated.
        """
        self.cmd = cmd
        self.environment = os.environ

    def execute( self, cmd ):
        """Execute cmd in a shell, and return the java.lang.Process instance.
        Accepts either a string command to be executed in a shell,
        or a sequence of [executable, args...].
        """
        shellCmd = self._formatCmd( cmd )

        env = self._formatEnvironment( self.environment )
        try:
            p = Runtime.getRuntime().exec( shellCmd, env, File(os.getcwd()) )
            return p
        except IOException, ex:
            raise OSError(
                0,
                "Failed to execute command (%s): %s" % ( shellCmd, ex )
                )

    ########## utility methods
    def _formatCmd( self, cmd ):
        """Format a command for execution in a shell."""
        if self.cmd is None:
            msgFmt = "Unable to execute commands in subshell because shell" \
                     " functionality not implemented for OS %s"  \
                     " Failed command=%s"
            raise OSError( 0, msgFmt % ( os._name, cmd ))

        if isinstance(cmd, basestring):
            shellCmd = self.cmd + [cmd]
        else:
            shellCmd = cmd

        return shellCmd

    def _formatEnvironment( self, env ):
        """Format enviroment in lines suitable for Runtime.exec"""
        lines = []
        for keyValue in env.items():
            lines.append( "%s=%s" % keyValue )
        return lines

def _getOsType():
    return os._name

_shellEnv = _ShellEnv(subprocess._shell_command)
shellexecute = _shellEnv.execute
