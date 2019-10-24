""" Emulate module 'readline' from CPython.
We are using the JavaReadline JNI wrapper for GNU readline.

2004-10-27, mark.asbach@rwth-aachen.de

"""

try:
    from org.gnu.readline import Readline, ReadlineCompleter
except ImportError, msg:
    raise ImportError, '%s. The readline module requires that java-readline from http://java-readline.sourceforge.net/ be on the classpath' % msg

__all__ = ["readline"]

def parse_and_bind (bindings):
    """Parse and execute single line of a readline init file.\
    
    """
    Readline.parseAndBind(bindings)

def get_line_buffer():
    """Return the current contents of the line buffer. 
        
    """
    return Readline.getLineBuffer()

def read_init_file(filename):
    """Parse a readline initialization file. 
    The default filename is the last filename used. 

    """
    Readline.readInitFile(filename)

def read_history_file(filename):
    """Load a readline history file. 
    The default filename is '~/.history'.

    """ 
    Readline.readHistoryFile(filename)

def write_history_file(filename):
    """Save a readline history file. 
    The default filename is '~/.history'.

    """ 
    Readline.writeHistoryFile(filename)

def set_completer(completionfunction = None):
    """Set or remove the completer instance. If an instance of ReadlineCompleter is specified, 
    it will be used as the new completer; if omitted or None, any completer already installed is removed.

    The completer method is called as completerclass.completer(text, state), for state in 0, 1, 2, ..., 
    until it returns a non-string value. It should return the next possible completion starting with text. 

    """
    class DerivedCompleter (ReadlineCompleter):
        def __init__ (self, method):
            self.method = method

        def completer (self, text, state):
            return self.method(text, state)

    Readline.setCompleter(DerivedCompleter(completionfunction))

def get_completer():
    """Get the current completer instance."""
    return Readline.getCompleter()

def set_completer_delims(delimiters):
    """Set the readline word delimiters for tab-completion."""
    Readline.setWordBreakCharacters(delimiters)

def get_completer_delims():
    """Get the readline word delimiters for tab-completion."""
    return Readline.getWordBreakCharacters()

def add_history(line):
    """Append a line to the history buffer, as if it was the last line typed."""
    Readline.addToHistory(line)

def get_current_history_length():
    """Get the number of lines currently available in history."""
    return Readline.getHistorySize()

