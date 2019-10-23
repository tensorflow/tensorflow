"""Create new objects of various types.  Deprecated.

This module is no longer required except for backward compatibility.
Objects of most types can now be created by calling the type object.
"""

from types import ClassType as classobj
from types import FunctionType as function
from types import InstanceType as instance
from types import MethodType as instancemethod
from types import ModuleType as module

# XXX: Jython can't really create a code object like CPython does
# (according to test.test_new)
## CodeType is not accessible in restricted execution mode
#try:
#    from types import CodeType as code
#except ImportError:
#    pass

from org.python.core import PyBytecode as code
