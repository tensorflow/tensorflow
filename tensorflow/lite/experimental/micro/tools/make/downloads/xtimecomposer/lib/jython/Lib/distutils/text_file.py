"""text_file

provides the TextFile class, which gives an interface to text files
that (optionally) takes care of stripping comments, ignoring blank
lines, and joining lines with backslashes."""

__revision__ = "$Id: text_file.py 29687 2002-11-14 02:25:42Z akuchling $"

from types import *
import sys, os, string


class TextFile:

    """Provides a file-like object that takes care of all the things you
       commonly want to do when processing a text file that has some
       line-by-line syntax: strip comments (as long as "#" is your
       comment character), skip blank lines, join adjacent lines by
       escaping the newline (ie. backslash at end of line), strip
       leading and/or trailing whitespace.  All of these are optional
       and independently controllable.

       Provides a 'warn()' method so you can generate warning messages that
       report physical line number, even if the logical line in question
       spans multiple physical lines.  Also provides 'unreadline()' for
       implementing line-at-a-time lookahead.

       Constructor is called as:

           TextFile (filename=None, file=None, **options)

       It bombs (RuntimeError) if both 'filename' and 'file' are None;
       'filename' should be a string, and 'file' a file object (or
       something that provides 'readline()' and 'close()' methods).  It is
       recommended that you supply at least 'filename', so that TextFile
       can include it in warning messages.  If 'file' is not supplied,
       TextFile creates its own using the 'open()' builtin.

       The options are all boolean, and affect the value returned by
       'readline()':
         strip_comments [default: true]
           strip from "#" to end-of-line, as well as any whitespace
           leading up to the "#" -- unless it is escaped by a backslash
         lstrip_ws [default: false]
           strip leading whitespace from each line before returning it
         rstrip_ws [default: true]
           strip trailing whitespace (including line terminator!) from
           each line before returning it
         skip_blanks [default: true}
           skip lines that are empty *after* stripping comments and
           whitespace.  (If both lstrip_ws and rstrip_ws are false,
           then some lines may consist of solely whitespace: these will
           *not* be skipped, even if 'skip_blanks' is true.)
         join_lines [default: false]
           if a backslash is the last non-newline character on a line
           after stripping comments and whitespace, join the following line
           to it to form one "logical line"; if N consecutive lines end
           with a backslash, then N+1 physical lines will be joined to
           form one logical line.
         collapse_join [default: false]
           strip leading whitespace from lines that are joined to their
           predecessor; only matters if (join_lines and not lstrip_ws)

       Note that since 'rstrip_ws' can strip the trailing newline, the
       semantics of 'readline()' must differ from those of the builtin file
       object's 'readline()' method!  In particular, 'readline()' returns
       None for end-of-file: an empty string might just be a blank line (or
       an all-whitespace line), if 'rstrip_ws' is true but 'skip_blanks' is
       not."""

    default_options = { 'strip_comments': 1,
                        'skip_blanks':    1,
                        'lstrip_ws':      0,
                        'rstrip_ws':      1,
                        'join_lines':     0,
                        'collapse_join':  0,
                      }

    def __init__ (self, filename=None, file=None, **options):
        """Construct a new TextFile object.  At least one of 'filename'
           (a string) and 'file' (a file-like object) must be supplied.
           They keyword argument options are described above and affect
           the values returned by 'readline()'."""

        if filename is None and file is None:
            raise RuntimeError, \
                  "you must supply either or both of 'filename' and 'file'"

        # set values for all options -- either from client option hash
        # or fallback to default_options
        for opt in self.default_options.keys():
            if options.has_key (opt):
                setattr (self, opt, options[opt])

            else:
                setattr (self, opt, self.default_options[opt])

        # sanity check client option hash
        for opt in options.keys():
            if not self.default_options.has_key (opt):
                raise KeyError, "invalid TextFile option '%s'" % opt

        if file is None:
            self.open (filename)
        else:
            self.filename = filename
            self.file = file
            self.current_line = 0       # assuming that file is at BOF!

        # 'linebuf' is a stack of lines that will be emptied before we
        # actually read from the file; it's only populated by an
        # 'unreadline()' operation
        self.linebuf = []


    def open (self, filename):
        """Open a new file named 'filename'.  This overrides both the
           'filename' and 'file' arguments to the constructor."""

        self.filename = filename
        self.file = open (self.filename, 'r')
        self.current_line = 0


    def close (self):
        """Close the current file and forget everything we know about it
           (filename, current line number)."""

        self.file.close ()
        self.file = None
        self.filename = None
        self.current_line = None


    def gen_error (self, msg, line=None):
        outmsg = []
        if line is None:
            line = self.current_line
        outmsg.append(self.filename + ", ")
        if type (line) in (ListType, TupleType):
            outmsg.append("lines %d-%d: " % tuple (line))
        else:
            outmsg.append("line %d: " % line)
        outmsg.append(str(msg))
        return string.join(outmsg, "")


    def error (self, msg, line=None):
        raise ValueError, "error: " + self.gen_error(msg, line)

    def warn (self, msg, line=None):
        """Print (to stderr) a warning message tied to the current logical
           line in the current file.  If the current logical line in the
           file spans multiple physical lines, the warning refers to the
           whole range, eg. "lines 3-5".  If 'line' supplied, it overrides
           the current line number; it may be a list or tuple to indicate a
           range of physical lines, or an integer for a single physical
           line."""
        sys.stderr.write("warning: " + self.gen_error(msg, line) + "\n")


    def readline (self):
        """Read and return a single logical line from the current file (or
           from an internal buffer if lines have previously been "unread"
           with 'unreadline()').  If the 'join_lines' option is true, this
           may involve reading multiple physical lines concatenated into a
           single string.  Updates the current line number, so calling
           'warn()' after 'readline()' emits a warning about the physical
           line(s) just read.  Returns None on end-of-file, since the empty
           string can occur if 'rstrip_ws' is true but 'strip_blanks' is
           not."""

        # If any "unread" lines waiting in 'linebuf', return the top
        # one.  (We don't actually buffer read-ahead data -- lines only
        # get put in 'linebuf' if the client explicitly does an
        # 'unreadline()'.
        if self.linebuf:
            line = self.linebuf[-1]
            del self.linebuf[-1]
            return line

        buildup_line = ''

        while 1:
            # read the line, make it None if EOF
            line = self.file.readline()
            if line == '': line = None

            if self.strip_comments and line:

                # Look for the first "#" in the line.  If none, never
                # mind.  If we find one and it's the first character, or
                # is not preceded by "\", then it starts a comment --
                # strip the comment, strip whitespace before it, and
                # carry on.  Otherwise, it's just an escaped "#", so
                # unescape it (and any other escaped "#"'s that might be
                # lurking in there) and otherwise leave the line alone.

                pos = string.find (line, "#")
                if pos == -1:           # no "#" -- no comments
                    pass

                # It's definitely a comment -- either "#" is the first
                # character, or it's elsewhere and unescaped.
                elif pos == 0 or line[pos-1] != "\\":
                    # Have to preserve the trailing newline, because it's
                    # the job of a later step (rstrip_ws) to remove it --
                    # and if rstrip_ws is false, we'd better preserve it!
                    # (NB. this means that if the final line is all comment
                    # and has no trailing newline, we will think that it's
                    # EOF; I think that's OK.)
                    eol = (line[-1] == '\n') and '\n' or ''
                    line = line[0:pos] + eol

                    # If all that's left is whitespace, then skip line
                    # *now*, before we try to join it to 'buildup_line' --
                    # that way constructs like
                    #   hello \\
                    #   # comment that should be ignored
                    #   there
                    # result in "hello there".
                    if string.strip(line) == "":
                        continue

                else:                   # it's an escaped "#"
                    line = string.replace (line, "\\#", "#")


            # did previous line end with a backslash? then accumulate
            if self.join_lines and buildup_line:
                # oops: end of file
                if line is None:
                    self.warn ("continuation line immediately precedes "
                               "end-of-file")
                    return buildup_line

                if self.collapse_join:
                    line = string.lstrip (line)
                line = buildup_line + line

                # careful: pay attention to line number when incrementing it
                if type (self.current_line) is ListType:
                    self.current_line[1] = self.current_line[1] + 1
                else:
                    self.current_line = [self.current_line,
                                         self.current_line+1]
            # just an ordinary line, read it as usual
            else:
                if line is None:        # eof
                    return None

                # still have to be careful about incrementing the line number!
                if type (self.current_line) is ListType:
                    self.current_line = self.current_line[1] + 1
                else:
                    self.current_line = self.current_line + 1


            # strip whitespace however the client wants (leading and
            # trailing, or one or the other, or neither)
            if self.lstrip_ws and self.rstrip_ws:
                line = string.strip (line)
            elif self.lstrip_ws:
                line = string.lstrip (line)
            elif self.rstrip_ws:
                line = string.rstrip (line)

            # blank line (whether we rstrip'ed or not)? skip to next line
            # if appropriate
            if (line == '' or line == '\n') and self.skip_blanks:
                continue

            if self.join_lines:
                if line[-1] == '\\':
                    buildup_line = line[:-1]
                    continue

                if line[-2:] == '\\\n':
                    buildup_line = line[0:-2] + '\n'
                    continue

            # well, I guess there's some actual content there: return it
            return line

    # readline ()


    def readlines (self):
        """Read and return the list of all logical lines remaining in the
           current file."""

        lines = []
        while 1:
            line = self.readline()
            if line is None:
                return lines
            lines.append (line)


    def unreadline (self, line):
        """Push 'line' (a string) onto an internal buffer that will be
           checked by future 'readline()' calls.  Handy for implementing
           a parser with line-at-a-time lookahead."""

        self.linebuf.append (line)


if __name__ == "__main__":
    test_data = """# test file

line 3 \\
# intervening comment
  continues on next line
"""
    # result 1: no fancy options
    result1 = map (lambda x: x + "\n", string.split (test_data, "\n")[0:-1])

    # result 2: just strip comments
    result2 = ["\n",
               "line 3 \\\n",
               "  continues on next line\n"]

    # result 3: just strip blank lines
    result3 = ["# test file\n",
               "line 3 \\\n",
               "# intervening comment\n",
               "  continues on next line\n"]

    # result 4: default, strip comments, blank lines, and trailing whitespace
    result4 = ["line 3 \\",
               "  continues on next line"]

    # result 5: strip comments and blanks, plus join lines (but don't
    # "collapse" joined lines
    result5 = ["line 3   continues on next line"]

    # result 6: strip comments and blanks, plus join lines (and
    # "collapse" joined lines
    result6 = ["line 3 continues on next line"]

    def test_input (count, description, file, expected_result):
        result = file.readlines ()
        # result = string.join (result, '')
        if result == expected_result:
            print "ok %d (%s)" % (count, description)
        else:
            print "not ok %d (%s):" % (count, description)
            print "** expected:"
            print expected_result
            print "** received:"
            print result


    filename = "test.txt"
    out_file = open (filename, "w")
    out_file.write (test_data)
    out_file.close ()

    in_file = TextFile (filename, strip_comments=0, skip_blanks=0,
                        lstrip_ws=0, rstrip_ws=0)
    test_input (1, "no processing", in_file, result1)

    in_file = TextFile (filename, strip_comments=1, skip_blanks=0,
                        lstrip_ws=0, rstrip_ws=0)
    test_input (2, "strip comments", in_file, result2)

    in_file = TextFile (filename, strip_comments=0, skip_blanks=1,
                        lstrip_ws=0, rstrip_ws=0)
    test_input (3, "strip blanks", in_file, result3)

    in_file = TextFile (filename)
    test_input (4, "default processing", in_file, result4)

    in_file = TextFile (filename, strip_comments=1, skip_blanks=1,
                        join_lines=1, rstrip_ws=1)
    test_input (5, "join lines without collapsing", in_file, result5)

    in_file = TextFile (filename, strip_comments=1, skip_blanks=1,
                        join_lines=1, rstrip_ws=1, collapse_join=1)
    test_input (6, "join lines with collapsing", in_file, result6)

    os.remove (filename)
