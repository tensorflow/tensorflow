"""An NNTP client class based on RFC 977: Network News Transfer Protocol.

Example:

>>> from nntplib import NNTP
>>> s = NNTP('news')
>>> resp, count, first, last, name = s.group('comp.lang.python')
>>> print 'Group', name, 'has', count, 'articles, range', first, 'to', last
Group comp.lang.python has 51 articles, range 5770 to 5821
>>> resp, subs = s.xhdr('subject', first + '-' + last)
>>> resp = s.quit()
>>>

Here 'resp' is the server response line.
Error responses are turned into exceptions.

To post an article from a file:
>>> f = open(filename, 'r') # file containing article, including header
>>> resp = s.post(f)
>>>

For descriptions of all methods, read the comments in the code below.
Note that all arguments and return values representing article numbers
are strings, not numbers, since they are rarely used for calculations.
"""

# RFC 977 by Brian Kantor and Phil Lapsley.
# xover, xgtitle, xpath, date methods by Kevan Heydon


# Imports
import re
import socket

__all__ = ["NNTP","NNTPReplyError","NNTPTemporaryError",
           "NNTPPermanentError","NNTPProtocolError","NNTPDataError",
           "error_reply","error_temp","error_perm","error_proto",
           "error_data",]

# Exceptions raised when an error or invalid response is received
class NNTPError(Exception):
    """Base class for all nntplib exceptions"""
    def __init__(self, *args):
        Exception.__init__(self, *args)
        try:
            self.response = args[0]
        except IndexError:
            self.response = 'No response given'

class NNTPReplyError(NNTPError):
    """Unexpected [123]xx reply"""
    pass

class NNTPTemporaryError(NNTPError):
    """4xx errors"""
    pass

class NNTPPermanentError(NNTPError):
    """5xx errors"""
    pass

class NNTPProtocolError(NNTPError):
    """Response does not begin with [1-5]"""
    pass

class NNTPDataError(NNTPError):
    """Error in response data"""
    pass

# for backwards compatibility
error_reply = NNTPReplyError
error_temp = NNTPTemporaryError
error_perm = NNTPPermanentError
error_proto = NNTPProtocolError
error_data = NNTPDataError



# Standard port used by NNTP servers
NNTP_PORT = 119


# Response numbers that are followed by additional text (e.g. article)
LONGRESP = ['100', '215', '220', '221', '222', '224', '230', '231', '282']


# Line terminators (we always output CRLF, but accept any of CRLF, CR, LF)
CRLF = '\r\n'



# The class itself
class NNTP:
    def __init__(self, host, port=NNTP_PORT, user=None, password=None,
                 readermode=None, usenetrc=True):
        """Initialize an instance.  Arguments:
        - host: hostname to connect to
        - port: port to connect to (default the standard NNTP port)
        - user: username to authenticate with
        - password: password to use with username
        - readermode: if true, send 'mode reader' command after
                      connecting.

        readermode is sometimes necessary if you are connecting to an
        NNTP server on the local machine and intend to call
        reader-specific comamnds, such as `group'.  If you get
        unexpected NNTPPermanentErrors, you might need to set
        readermode.
        """
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.file = self.sock.makefile('rb')
        self.debugging = 0
        self.welcome = self.getresp()

        # 'mode reader' is sometimes necessary to enable 'reader' mode.
        # However, the order in which 'mode reader' and 'authinfo' need to
        # arrive differs between some NNTP servers. Try to send
        # 'mode reader', and if it fails with an authorization failed
        # error, try again after sending authinfo.
        readermode_afterauth = 0
        if readermode:
            try:
                self.welcome = self.shortcmd('mode reader')
            except NNTPPermanentError:
                # error 500, probably 'not implemented'
                pass
            except NNTPTemporaryError, e:
                if user and e.response[:3] == '480':
                    # Need authorization before 'mode reader'
                    readermode_afterauth = 1
                else:
                    raise
        # If no login/password was specified, try to get them from ~/.netrc
        # Presume that if .netc has an entry, NNRP authentication is required.
        try:
            if usenetrc and not user:
                import netrc
                credentials = netrc.netrc()
                auth = credentials.authenticators(host)
                if auth:
                    user = auth[0]
                    password = auth[2]
        except IOError:
            pass
        # Perform NNRP authentication if needed.
        if user:
            resp = self.shortcmd('authinfo user '+user)
            if resp[:3] == '381':
                if not password:
                    raise NNTPReplyError(resp)
                else:
                    resp = self.shortcmd(
                            'authinfo pass '+password)
                    if resp[:3] != '281':
                        raise NNTPPermanentError(resp)
            if readermode_afterauth:
                try:
                    self.welcome = self.shortcmd('mode reader')
                except NNTPPermanentError:
                    # error 500, probably 'not implemented'
                    pass


    # Get the welcome message from the server
    # (this is read and squirreled away by __init__()).
    # If the response code is 200, posting is allowed;
    # if it 201, posting is not allowed

    def getwelcome(self):
        """Get the welcome message from the server
        (this is read and squirreled away by __init__()).
        If the response code is 200, posting is allowed;
        if it 201, posting is not allowed."""

        if self.debugging: print '*welcome*', repr(self.welcome)
        return self.welcome

    def set_debuglevel(self, level):
        """Set the debugging level.  Argument 'level' means:
        0: no debugging output (default)
        1: print commands and responses but not body text etc.
        2: also print raw lines read and sent before stripping CR/LF"""

        self.debugging = level
    debug = set_debuglevel

    def putline(self, line):
        """Internal: send one line to the server, appending CRLF."""
        line = line + CRLF
        if self.debugging > 1: print '*put*', repr(line)
        self.sock.sendall(line)

    def putcmd(self, line):
        """Internal: send one command to the server (through putline())."""
        if self.debugging: print '*cmd*', repr(line)
        self.putline(line)

    def getline(self):
        """Internal: return one line from the server, stripping CRLF.
        Raise EOFError if the connection is closed."""
        line = self.file.readline()
        if self.debugging > 1:
            print '*get*', repr(line)
        if not line: raise EOFError
        if line[-2:] == CRLF: line = line[:-2]
        elif line[-1:] in CRLF: line = line[:-1]
        return line

    def getresp(self):
        """Internal: get a response from the server.
        Raise various errors if the response indicates an error."""
        resp = self.getline()
        if self.debugging: print '*resp*', repr(resp)
        c = resp[:1]
        if c == '4':
            raise NNTPTemporaryError(resp)
        if c == '5':
            raise NNTPPermanentError(resp)
        if c not in '123':
            raise NNTPProtocolError(resp)
        return resp

    def getlongresp(self, file=None):
        """Internal: get a response plus following text from the server.
        Raise various errors if the response indicates an error."""

        openedFile = None
        try:
            # If a string was passed then open a file with that name
            if isinstance(file, str):
                openedFile = file = open(file, "w")

            resp = self.getresp()
            if resp[:3] not in LONGRESP:
                raise NNTPReplyError(resp)
            list = []
            while 1:
                line = self.getline()
                if line == '.':
                    break
                if line[:2] == '..':
                    line = line[1:]
                if file:
                    file.write(line + "\n")
                else:
                    list.append(line)
        finally:
            # If this method created the file, then it must close it
            if openedFile:
                openedFile.close()

        return resp, list

    def shortcmd(self, line):
        """Internal: send a command and get the response."""
        self.putcmd(line)
        return self.getresp()

    def longcmd(self, line, file=None):
        """Internal: send a command and get the response plus following text."""
        self.putcmd(line)
        return self.getlongresp(file)

    def newgroups(self, date, time, file=None):
        """Process a NEWGROUPS command.  Arguments:
        - date: string 'yymmdd' indicating the date
        - time: string 'hhmmss' indicating the time
        Return:
        - resp: server response if successful
        - list: list of newsgroup names"""

        return self.longcmd('NEWGROUPS ' + date + ' ' + time, file)

    def newnews(self, group, date, time, file=None):
        """Process a NEWNEWS command.  Arguments:
        - group: group name or '*'
        - date: string 'yymmdd' indicating the date
        - time: string 'hhmmss' indicating the time
        Return:
        - resp: server response if successful
        - list: list of message ids"""

        cmd = 'NEWNEWS ' + group + ' ' + date + ' ' + time
        return self.longcmd(cmd, file)

    def list(self, file=None):
        """Process a LIST command.  Return:
        - resp: server response if successful
        - list: list of (group, last, first, flag) (strings)"""

        resp, list = self.longcmd('LIST', file)
        for i in range(len(list)):
            # Parse lines into "group last first flag"
            list[i] = tuple(list[i].split())
        return resp, list

    def description(self, group):

        """Get a description for a single group.  If more than one
        group matches ('group' is a pattern), return the first.  If no
        group matches, return an empty string.

        This elides the response code from the server, since it can
        only be '215' or '285' (for xgtitle) anyway.  If the response
        code is needed, use the 'descriptions' method.

        NOTE: This neither checks for a wildcard in 'group' nor does
        it check whether the group actually exists."""

        resp, lines = self.descriptions(group)
        if len(lines) == 0:
            return ""
        else:
            return lines[0][1]

    def descriptions(self, group_pattern):
        """Get descriptions for a range of groups."""
        line_pat = re.compile("^(?P<group>[^ \t]+)[ \t]+(.*)$")
        # Try the more std (acc. to RFC2980) LIST NEWSGROUPS first
        resp, raw_lines = self.longcmd('LIST NEWSGROUPS ' + group_pattern)
        if resp[:3] != "215":
            # Now the deprecated XGTITLE.  This either raises an error
            # or succeeds with the same output structure as LIST
            # NEWSGROUPS.
            resp, raw_lines = self.longcmd('XGTITLE ' + group_pattern)
        lines = []
        for raw_line in raw_lines:
            match = line_pat.search(raw_line.strip())
            if match:
                lines.append(match.group(1, 2))
        return resp, lines

    def group(self, name):
        """Process a GROUP command.  Argument:
        - group: the group name
        Returns:
        - resp: server response if successful
        - count: number of articles (string)
        - first: first article number (string)
        - last: last article number (string)
        - name: the group name"""

        resp = self.shortcmd('GROUP ' + name)
        if resp[:3] != '211':
            raise NNTPReplyError(resp)
        words = resp.split()
        count = first = last = 0
        n = len(words)
        if n > 1:
            count = words[1]
            if n > 2:
                first = words[2]
                if n > 3:
                    last = words[3]
                    if n > 4:
                        name = words[4].lower()
        return resp, count, first, last, name

    def help(self, file=None):
        """Process a HELP command.  Returns:
        - resp: server response if successful
        - list: list of strings"""

        return self.longcmd('HELP',file)

    def statparse(self, resp):
        """Internal: parse the response of a STAT, NEXT or LAST command."""
        if resp[:2] != '22':
            raise NNTPReplyError(resp)
        words = resp.split()
        nr = 0
        id = ''
        n = len(words)
        if n > 1:
            nr = words[1]
            if n > 2:
                id = words[2]
        return resp, nr, id

    def statcmd(self, line):
        """Internal: process a STAT, NEXT or LAST command."""
        resp = self.shortcmd(line)
        return self.statparse(resp)

    def stat(self, id):
        """Process a STAT command.  Argument:
        - id: article number or message id
        Returns:
        - resp: server response if successful
        - nr:   the article number
        - id:   the message id"""

        return self.statcmd('STAT ' + id)

    def next(self):
        """Process a NEXT command.  No arguments.  Return as for STAT."""
        return self.statcmd('NEXT')

    def last(self):
        """Process a LAST command.  No arguments.  Return as for STAT."""
        return self.statcmd('LAST')

    def artcmd(self, line, file=None):
        """Internal: process a HEAD, BODY or ARTICLE command."""
        resp, list = self.longcmd(line, file)
        resp, nr, id = self.statparse(resp)
        return resp, nr, id, list

    def head(self, id):
        """Process a HEAD command.  Argument:
        - id: article number or message id
        Returns:
        - resp: server response if successful
        - nr: article number
        - id: message id
        - list: the lines of the article's header"""

        return self.artcmd('HEAD ' + id)

    def body(self, id, file=None):
        """Process a BODY command.  Argument:
        - id: article number or message id
        - file: Filename string or file object to store the article in
        Returns:
        - resp: server response if successful
        - nr: article number
        - id: message id
        - list: the lines of the article's body or an empty list
                if file was used"""

        return self.artcmd('BODY ' + id, file)

    def article(self, id):
        """Process an ARTICLE command.  Argument:
        - id: article number or message id
        Returns:
        - resp: server response if successful
        - nr: article number
        - id: message id
        - list: the lines of the article"""

        return self.artcmd('ARTICLE ' + id)

    def slave(self):
        """Process a SLAVE command.  Returns:
        - resp: server response if successful"""

        return self.shortcmd('SLAVE')

    def xhdr(self, hdr, str, file=None):
        """Process an XHDR command (optional server extension).  Arguments:
        - hdr: the header type (e.g. 'subject')
        - str: an article nr, a message id, or a range nr1-nr2
        Returns:
        - resp: server response if successful
        - list: list of (nr, value) strings"""

        pat = re.compile('^([0-9]+) ?(.*)\n?')
        resp, lines = self.longcmd('XHDR ' + hdr + ' ' + str, file)
        for i in range(len(lines)):
            line = lines[i]
            m = pat.match(line)
            if m:
                lines[i] = m.group(1, 2)
        return resp, lines

    def xover(self, start, end, file=None):
        """Process an XOVER command (optional server extension) Arguments:
        - start: start of range
        - end: end of range
        Returns:
        - resp: server response if successful
        - list: list of (art-nr, subject, poster, date,
                         id, references, size, lines)"""

        resp, lines = self.longcmd('XOVER ' + start + '-' + end, file)
        xover_lines = []
        for line in lines:
            elem = line.split("\t")
            try:
                xover_lines.append((elem[0],
                                    elem[1],
                                    elem[2],
                                    elem[3],
                                    elem[4],
                                    elem[5].split(),
                                    elem[6],
                                    elem[7]))
            except IndexError:
                raise NNTPDataError(line)
        return resp,xover_lines

    def xgtitle(self, group, file=None):
        """Process an XGTITLE command (optional server extension) Arguments:
        - group: group name wildcard (i.e. news.*)
        Returns:
        - resp: server response if successful
        - list: list of (name,title) strings"""

        line_pat = re.compile("^([^ \t]+)[ \t]+(.*)$")
        resp, raw_lines = self.longcmd('XGTITLE ' + group, file)
        lines = []
        for raw_line in raw_lines:
            match = line_pat.search(raw_line.strip())
            if match:
                lines.append(match.group(1, 2))
        return resp, lines

    def xpath(self,id):
        """Process an XPATH command (optional server extension) Arguments:
        - id: Message id of article
        Returns:
        resp: server response if successful
        path: directory path to article"""

        resp = self.shortcmd("XPATH " + id)
        if resp[:3] != '223':
            raise NNTPReplyError(resp)
        try:
            [resp_num, path] = resp.split()
        except ValueError:
            raise NNTPReplyError(resp)
        else:
            return resp, path

    def date (self):
        """Process the DATE command. Arguments:
        None
        Returns:
        resp: server response if successful
        date: Date suitable for newnews/newgroups commands etc.
        time: Time suitable for newnews/newgroups commands etc."""

        resp = self.shortcmd("DATE")
        if resp[:3] != '111':
            raise NNTPReplyError(resp)
        elem = resp.split()
        if len(elem) != 2:
            raise NNTPDataError(resp)
        date = elem[1][2:8]
        time = elem[1][-6:]
        if len(date) != 6 or len(time) != 6:
            raise NNTPDataError(resp)
        return resp, date, time


    def post(self, f):
        """Process a POST command.  Arguments:
        - f: file containing the article
        Returns:
        - resp: server response if successful"""

        resp = self.shortcmd('POST')
        # Raises error_??? if posting is not allowed
        if resp[0] != '3':
            raise NNTPReplyError(resp)
        while 1:
            line = f.readline()
            if not line:
                break
            if line[-1] == '\n':
                line = line[:-1]
            if line[:1] == '.':
                line = '.' + line
            self.putline(line)
        self.putline('.')
        return self.getresp()

    def ihave(self, id, f):
        """Process an IHAVE command.  Arguments:
        - id: message-id of the article
        - f:  file containing the article
        Returns:
        - resp: server response if successful
        Note that if the server refuses the article an exception is raised."""

        resp = self.shortcmd('IHAVE ' + id)
        # Raises error_??? if the server already has it
        if resp[0] != '3':
            raise NNTPReplyError(resp)
        while 1:
            line = f.readline()
            if not line:
                break
            if line[-1] == '\n':
                line = line[:-1]
            if line[:1] == '.':
                line = '.' + line
            self.putline(line)
        self.putline('.')
        return self.getresp()

    def quit(self):
        """Process a QUIT command and close the socket.  Returns:
        - resp: server response if successful"""

        resp = self.shortcmd('QUIT')
        self.file.close()
        self.sock.close()
        del self.file, self.sock
        return resp


# Test retrieval when run as a script.
# Assumption: if there's a local news server, it's called 'news'.
# Assumption: if user queries a remote news server, it's named
# in the environment variable NNTPSERVER (used by slrn and kin)
# and we want readermode off.
if __name__ == '__main__':
    import os
    newshost = 'news' and os.environ["NNTPSERVER"]
    if newshost.find('.') == -1:
        mode = 'readermode'
    else:
        mode = None
    s = NNTP(newshost, readermode=mode)
    resp, count, first, last, name = s.group('comp.lang.python')
    print resp
    print 'Group', name, 'has', count, 'articles, range', first, 'to', last
    resp, subs = s.xhdr('subject', first + '-' + last)
    print resp
    for item in subs:
        print "%7s %s" % item
    resp = s.quit()
    print resp
