"""MH interface -- purely object-oriented (well, almost)

Executive summary:

import mhlib

mh = mhlib.MH()         # use default mailbox directory and profile
mh = mhlib.MH(mailbox)  # override mailbox location (default from profile)
mh = mhlib.MH(mailbox, profile) # override mailbox and profile

mh.error(format, ...)   # print error message -- can be overridden
s = mh.getprofile(key)  # profile entry (None if not set)
path = mh.getpath()     # mailbox pathname
name = mh.getcontext()  # name of current folder
mh.setcontext(name)     # set name of current folder

list = mh.listfolders() # names of top-level folders
list = mh.listallfolders() # names of all folders, including subfolders
list = mh.listsubfolders(name) # direct subfolders of given folder
list = mh.listallsubfolders(name) # all subfolders of given folder

mh.makefolder(name)     # create new folder
mh.deletefolder(name)   # delete folder -- must have no subfolders

f = mh.openfolder(name) # new open folder object

f.error(format, ...)    # same as mh.error(format, ...)
path = f.getfullname()  # folder's full pathname
path = f.getsequencesfilename() # full pathname of folder's sequences file
path = f.getmessagefilename(n)  # full pathname of message n in folder

list = f.listmessages() # list of messages in folder (as numbers)
n = f.getcurrent()      # get current message
f.setcurrent(n)         # set current message
list = f.parsesequence(seq)     # parse msgs syntax into list of messages
n = f.getlast()         # get last message (0 if no messagse)
f.setlast(n)            # set last message (internal use only)

dict = f.getsequences() # dictionary of sequences in folder {name: list}
f.putsequences(dict)    # write sequences back to folder

f.createmessage(n, fp)  # add message from file f as number n
f.removemessages(list)  # remove messages in list from folder
f.refilemessages(list, tofolder) # move messages in list to other folder
f.movemessage(n, tofolder, ton)  # move one message to a given destination
f.copymessage(n, tofolder, ton)  # copy one message to a given destination

m = f.openmessage(n)    # new open message object (costs a file descriptor)
m is a derived class of mimetools.Message(rfc822.Message), with:
s = m.getheadertext()   # text of message's headers
s = m.getheadertext(pred) # text of message's headers, filtered by pred
s = m.getbodytext()     # text of message's body, decoded
s = m.getbodytext(0)    # text of message's body, not decoded
"""

# XXX To do, functionality:
# - annotate messages
# - send messages
#
# XXX To do, organization:
# - move IntSet to separate file
# - move most Message functionality to module mimetools


# Customizable defaults

MH_PROFILE = '~/.mh_profile'
PATH = '~/Mail'
MH_SEQUENCES = '.mh_sequences'
FOLDER_PROTECT = 0700


# Imported modules

import os
import sys
import re
import mimetools
import multifile
import shutil
from bisect import bisect

__all__ = ["MH","Error","Folder","Message"]

# Exported constants

class Error(Exception):
    pass


class MH:
    """Class representing a particular collection of folders.
    Optional constructor arguments are the pathname for the directory
    containing the collection, and the MH profile to use.
    If either is omitted or empty a default is used; the default
    directory is taken from the MH profile if it is specified there."""

    def __init__(self, path = None, profile = None):
        """Constructor."""
        if profile is None: profile = MH_PROFILE
        self.profile = os.path.expanduser(profile)
        if path is None: path = self.getprofile('Path')
        if not path: path = PATH
        if not os.path.isabs(path) and path[0] != '~':
            path = os.path.join('~', path)
        path = os.path.expanduser(path)
        if not os.path.isdir(path): raise Error, 'MH() path not found'
        self.path = path

    def __repr__(self):
        """String representation."""
        return 'MH(%r, %r)' % (self.path, self.profile)

    def error(self, msg, *args):
        """Routine to print an error.  May be overridden by a derived class."""
        sys.stderr.write('MH error: %s\n' % (msg % args))

    def getprofile(self, key):
        """Return a profile entry, None if not found."""
        return pickline(self.profile, key)

    def getpath(self):
        """Return the path (the name of the collection's directory)."""
        return self.path

    def getcontext(self):
        """Return the name of the current folder."""
        context = pickline(os.path.join(self.getpath(), 'context'),
                  'Current-Folder')
        if not context: context = 'inbox'
        return context

    def setcontext(self, context):
        """Set the name of the current folder."""
        fn = os.path.join(self.getpath(), 'context')
        f = open(fn, "w")
        f.write("Current-Folder: %s\n" % context)
        f.close()

    def listfolders(self):
        """Return the names of the top-level folders."""
        folders = []
        path = self.getpath()
        for name in os.listdir(path):
            fullname = os.path.join(path, name)
            if os.path.isdir(fullname):
                folders.append(name)
        folders.sort()
        return folders

    def listsubfolders(self, name):
        """Return the names of the subfolders in a given folder
        (prefixed with the given folder name)."""
        fullname = os.path.join(self.path, name)
        # Get the link count so we can avoid listing folders
        # that have no subfolders.
        nlinks = os.stat(fullname).st_nlink
        if nlinks <= 2:
            return []
        subfolders = []
        subnames = os.listdir(fullname)
        for subname in subnames:
            fullsubname = os.path.join(fullname, subname)
            if os.path.isdir(fullsubname):
                name_subname = os.path.join(name, subname)
                subfolders.append(name_subname)
                # Stop looking for subfolders when
                # we've seen them all
                nlinks = nlinks - 1
                if nlinks <= 2:
                    break
        subfolders.sort()
        return subfolders

    def listallfolders(self):
        """Return the names of all folders and subfolders, recursively."""
        return self.listallsubfolders('')

    def listallsubfolders(self, name):
        """Return the names of subfolders in a given folder, recursively."""
        fullname = os.path.join(self.path, name)
        # Get the link count so we can avoid listing folders
        # that have no subfolders.
        nlinks = os.stat(fullname).st_nlink
        if nlinks <= 2:
            return []
        subfolders = []
        subnames = os.listdir(fullname)
        for subname in subnames:
            if subname[0] == ',' or isnumeric(subname): continue
            fullsubname = os.path.join(fullname, subname)
            if os.path.isdir(fullsubname):
                name_subname = os.path.join(name, subname)
                subfolders.append(name_subname)
                if not os.path.islink(fullsubname):
                    subsubfolders = self.listallsubfolders(
                              name_subname)
                    subfolders = subfolders + subsubfolders
                # Stop looking for subfolders when
                # we've seen them all
                nlinks = nlinks - 1
                if nlinks <= 2:
                    break
        subfolders.sort()
        return subfolders

    def openfolder(self, name):
        """Return a new Folder object for the named folder."""
        return Folder(self, name)

    def makefolder(self, name):
        """Create a new folder (or raise os.error if it cannot be created)."""
        protect = pickline(self.profile, 'Folder-Protect')
        if protect and isnumeric(protect):
            mode = int(protect, 8)
        else:
            mode = FOLDER_PROTECT
        os.mkdir(os.path.join(self.getpath(), name), mode)

    def deletefolder(self, name):
        """Delete a folder.  This removes files in the folder but not
        subdirectories.  Raise os.error if deleting the folder itself fails."""
        fullname = os.path.join(self.getpath(), name)
        for subname in os.listdir(fullname):
            fullsubname = os.path.join(fullname, subname)
            try:
                os.unlink(fullsubname)
            except os.error:
                self.error('%s not deleted, continuing...' %
                          fullsubname)
        os.rmdir(fullname)


numericprog = re.compile('^[1-9][0-9]*$')
def isnumeric(str):
    return numericprog.match(str) is not None

class Folder:
    """Class representing a particular folder."""

    def __init__(self, mh, name):
        """Constructor."""
        self.mh = mh
        self.name = name
        if not os.path.isdir(self.getfullname()):
            raise Error, 'no folder %s' % name

    def __repr__(self):
        """String representation."""
        return 'Folder(%r, %r)' % (self.mh, self.name)

    def error(self, *args):
        """Error message handler."""
        self.mh.error(*args)

    def getfullname(self):
        """Return the full pathname of the folder."""
        return os.path.join(self.mh.path, self.name)

    def getsequencesfilename(self):
        """Return the full pathname of the folder's sequences file."""
        return os.path.join(self.getfullname(), MH_SEQUENCES)

    def getmessagefilename(self, n):
        """Return the full pathname of a message in the folder."""
        return os.path.join(self.getfullname(), str(n))

    def listsubfolders(self):
        """Return list of direct subfolders."""
        return self.mh.listsubfolders(self.name)

    def listallsubfolders(self):
        """Return list of all subfolders."""
        return self.mh.listallsubfolders(self.name)

    def listmessages(self):
        """Return the list of messages currently present in the folder.
        As a side effect, set self.last to the last message (or 0)."""
        messages = []
        match = numericprog.match
        append = messages.append
        for name in os.listdir(self.getfullname()):
            if match(name):
                append(name)
        messages = map(int, messages)
        messages.sort()
        if messages:
            self.last = messages[-1]
        else:
            self.last = 0
        return messages

    def getsequences(self):
        """Return the set of sequences for the folder."""
        sequences = {}
        fullname = self.getsequencesfilename()
        try:
            f = open(fullname, 'r')
        except IOError:
            return sequences
        while 1:
            line = f.readline()
            if not line: break
            fields = line.split(':')
            if len(fields) != 2:
                self.error('bad sequence in %s: %s' %
                          (fullname, line.strip()))
            key = fields[0].strip()
            value = IntSet(fields[1].strip(), ' ').tolist()
            sequences[key] = value
        return sequences

    def putsequences(self, sequences):
        """Write the set of sequences back to the folder."""
        fullname = self.getsequencesfilename()
        f = None
        for key, seq in sequences.iteritems():
            s = IntSet('', ' ')
            s.fromlist(seq)
            if not f: f = open(fullname, 'w')
            f.write('%s: %s\n' % (key, s.tostring()))
        if not f:
            try:
                os.unlink(fullname)
            except os.error:
                pass
        else:
            f.close()

    def getcurrent(self):
        """Return the current message.  Raise Error when there is none."""
        seqs = self.getsequences()
        try:
            return max(seqs['cur'])
        except (ValueError, KeyError):
            raise Error, "no cur message"

    def setcurrent(self, n):
        """Set the current message."""
        updateline(self.getsequencesfilename(), 'cur', str(n), 0)

    def parsesequence(self, seq):
        """Parse an MH sequence specification into a message list.
        Attempt to mimic mh-sequence(5) as close as possible.
        Also attempt to mimic observed behavior regarding which
        conditions cause which error messages."""
        # XXX Still not complete (see mh-format(5)).
        # Missing are:
        # - 'prev', 'next' as count
        # - Sequence-Negation option
        all = self.listmessages()
        # Observed behavior: test for empty folder is done first
        if not all:
            raise Error, "no messages in %s" % self.name
        # Common case first: all is frequently the default
        if seq == 'all':
            return all
        # Test for X:Y before X-Y because 'seq:-n' matches both
        i = seq.find(':')
        if i >= 0:
            head, dir, tail = seq[:i], '', seq[i+1:]
            if tail[:1] in '-+':
                dir, tail = tail[:1], tail[1:]
            if not isnumeric(tail):
                raise Error, "bad message list %s" % seq
            try:
                count = int(tail)
            except (ValueError, OverflowError):
                # Can't use sys.maxint because of i+count below
                count = len(all)
            try:
                anchor = self._parseindex(head, all)
            except Error, msg:
                seqs = self.getsequences()
                if not head in seqs:
                    if not msg:
                        msg = "bad message list %s" % seq
                    raise Error, msg, sys.exc_info()[2]
                msgs = seqs[head]
                if not msgs:
                    raise Error, "sequence %s empty" % head
                if dir == '-':
                    return msgs[-count:]
                else:
                    return msgs[:count]
            else:
                if not dir:
                    if head in ('prev', 'last'):
                        dir = '-'
                if dir == '-':
                    i = bisect(all, anchor)
                    return all[max(0, i-count):i]
                else:
                    i = bisect(all, anchor-1)
                    return all[i:i+count]
        # Test for X-Y next
        i = seq.find('-')
        if i >= 0:
            begin = self._parseindex(seq[:i], all)
            end = self._parseindex(seq[i+1:], all)
            i = bisect(all, begin-1)
            j = bisect(all, end)
            r = all[i:j]
            if not r:
                raise Error, "bad message list %s" % seq
            return r
        # Neither X:Y nor X-Y; must be a number or a (pseudo-)sequence
        try:
            n = self._parseindex(seq, all)
        except Error, msg:
            seqs = self.getsequences()
            if not seq in seqs:
                if not msg:
                    msg = "bad message list %s" % seq
                raise Error, msg
            return seqs[seq]
        else:
            if n not in all:
                if isnumeric(seq):
                    raise Error, "message %d doesn't exist" % n
                else:
                    raise Error, "no %s message" % seq
            else:
                return [n]

    def _parseindex(self, seq, all):
        """Internal: parse a message number (or cur, first, etc.)."""
        if isnumeric(seq):
            try:
                return int(seq)
            except (OverflowError, ValueError):
                return sys.maxint
        if seq in ('cur', '.'):
            return self.getcurrent()
        if seq == 'first':
            return all[0]
        if seq == 'last':
            return all[-1]
        if seq == 'next':
            n = self.getcurrent()
            i = bisect(all, n)
            try:
                return all[i]
            except IndexError:
                raise Error, "no next message"
        if seq == 'prev':
            n = self.getcurrent()
            i = bisect(all, n-1)
            if i == 0:
                raise Error, "no prev message"
            try:
                return all[i-1]
            except IndexError:
                raise Error, "no prev message"
        raise Error, None

    def openmessage(self, n):
        """Open a message -- returns a Message object."""
        return Message(self, n)

    def removemessages(self, list):
        """Remove one or more messages -- may raise os.error."""
        errors = []
        deleted = []
        for n in list:
            path = self.getmessagefilename(n)
            commapath = self.getmessagefilename(',' + str(n))
            try:
                os.unlink(commapath)
            except os.error:
                pass
            try:
                os.rename(path, commapath)
            except os.error, msg:
                errors.append(msg)
            else:
                deleted.append(n)
        if deleted:
            self.removefromallsequences(deleted)
        if errors:
            if len(errors) == 1:
                raise os.error, errors[0]
            else:
                raise os.error, ('multiple errors:', errors)

    def refilemessages(self, list, tofolder, keepsequences=0):
        """Refile one or more messages -- may raise os.error.
        'tofolder' is an open folder object."""
        errors = []
        refiled = {}
        for n in list:
            ton = tofolder.getlast() + 1
            path = self.getmessagefilename(n)
            topath = tofolder.getmessagefilename(ton)
            try:
                os.rename(path, topath)
            except os.error:
                # Try copying
                try:
                    shutil.copy2(path, topath)
                    os.unlink(path)
                except (IOError, os.error), msg:
                    errors.append(msg)
                    try:
                        os.unlink(topath)
                    except os.error:
                        pass
                    continue
            tofolder.setlast(ton)
            refiled[n] = ton
        if refiled:
            if keepsequences:
                tofolder._copysequences(self, refiled.items())
            self.removefromallsequences(refiled.keys())
        if errors:
            if len(errors) == 1:
                raise os.error, errors[0]
            else:
                raise os.error, ('multiple errors:', errors)

    def _copysequences(self, fromfolder, refileditems):
        """Helper for refilemessages() to copy sequences."""
        fromsequences = fromfolder.getsequences()
        tosequences = self.getsequences()
        changed = 0
        for name, seq in fromsequences.items():
            try:
                toseq = tosequences[name]
                new = 0
            except KeyError:
                toseq = []
                new = 1
            for fromn, ton in refileditems:
                if fromn in seq:
                    toseq.append(ton)
                    changed = 1
            if new and toseq:
                tosequences[name] = toseq
        if changed:
            self.putsequences(tosequences)

    def movemessage(self, n, tofolder, ton):
        """Move one message over a specific destination message,
        which may or may not already exist."""
        path = self.getmessagefilename(n)
        # Open it to check that it exists
        f = open(path)
        f.close()
        del f
        topath = tofolder.getmessagefilename(ton)
        backuptopath = tofolder.getmessagefilename(',%d' % ton)
        try:
            os.rename(topath, backuptopath)
        except os.error:
            pass
        try:
            os.rename(path, topath)
        except os.error:
            # Try copying
            ok = 0
            try:
                tofolder.setlast(None)
                shutil.copy2(path, topath)
                ok = 1
            finally:
                if not ok:
                    try:
                        os.unlink(topath)
                    except os.error:
                        pass
            os.unlink(path)
        self.removefromallsequences([n])

    def copymessage(self, n, tofolder, ton):
        """Copy one message over a specific destination message,
        which may or may not already exist."""
        path = self.getmessagefilename(n)
        # Open it to check that it exists
        f = open(path)
        f.close()
        del f
        topath = tofolder.getmessagefilename(ton)
        backuptopath = tofolder.getmessagefilename(',%d' % ton)
        try:
            os.rename(topath, backuptopath)
        except os.error:
            pass
        ok = 0
        try:
            tofolder.setlast(None)
            shutil.copy2(path, topath)
            ok = 1
        finally:
            if not ok:
                try:
                    os.unlink(topath)
                except os.error:
                    pass

    def createmessage(self, n, txt):
        """Create a message, with text from the open file txt."""
        path = self.getmessagefilename(n)
        backuppath = self.getmessagefilename(',%d' % n)
        try:
            os.rename(path, backuppath)
        except os.error:
            pass
        ok = 0
        BUFSIZE = 16*1024
        try:
            f = open(path, "w")
            while 1:
                buf = txt.read(BUFSIZE)
                if not buf:
                    break
                f.write(buf)
            f.close()
            ok = 1
        finally:
            if not ok:
                try:
                    os.unlink(path)
                except os.error:
                    pass

    def removefromallsequences(self, list):
        """Remove one or more messages from all sequences (including last)
        -- but not from 'cur'!!!"""
        if hasattr(self, 'last') and self.last in list:
            del self.last
        sequences = self.getsequences()
        changed = 0
        for name, seq in sequences.items():
            if name == 'cur':
                continue
            for n in list:
                if n in seq:
                    seq.remove(n)
                    changed = 1
                    if not seq:
                        del sequences[name]
        if changed:
            self.putsequences(sequences)

    def getlast(self):
        """Return the last message number."""
        if not hasattr(self, 'last'):
            self.listmessages() # Set self.last
        return self.last

    def setlast(self, last):
        """Set the last message number."""
        if last is None:
            if hasattr(self, 'last'):
                del self.last
        else:
            self.last = last

class Message(mimetools.Message):

    def __init__(self, f, n, fp = None):
        """Constructor."""
        self.folder = f
        self.number = n
        if fp is None:
            path = f.getmessagefilename(n)
            fp = open(path, 'r')
        mimetools.Message.__init__(self, fp)

    def __repr__(self):
        """String representation."""
        return 'Message(%s, %s)' % (repr(self.folder), self.number)

    def getheadertext(self, pred = None):
        """Return the message's header text as a string.  If an
        argument is specified, it is used as a filter predicate to
        decide which headers to return (its argument is the header
        name converted to lower case)."""
        if pred is None:
            return ''.join(self.headers)
        headers = []
        hit = 0
        for line in self.headers:
            if not line[0].isspace():
                i = line.find(':')
                if i > 0:
                    hit = pred(line[:i].lower())
            if hit: headers.append(line)
        return ''.join(headers)

    def getbodytext(self, decode = 1):
        """Return the message's body text as string.  This undoes a
        Content-Transfer-Encoding, but does not interpret other MIME
        features (e.g. multipart messages).  To suppress decoding,
        pass 0 as an argument."""
        self.fp.seek(self.startofbody)
        encoding = self.getencoding()
        if not decode or encoding in ('', '7bit', '8bit', 'binary'):
            return self.fp.read()
        try:
            from cStringIO import StringIO
        except ImportError:
            from StringIO import StringIO
        output = StringIO()
        mimetools.decode(self.fp, output, encoding)
        return output.getvalue()

    def getbodyparts(self):
        """Only for multipart messages: return the message's body as a
        list of SubMessage objects.  Each submessage object behaves
        (almost) as a Message object."""
        if self.getmaintype() != 'multipart':
            raise Error, 'Content-Type is not multipart/*'
        bdry = self.getparam('boundary')
        if not bdry:
            raise Error, 'multipart/* without boundary param'
        self.fp.seek(self.startofbody)
        mf = multifile.MultiFile(self.fp)
        mf.push(bdry)
        parts = []
        while mf.next():
            n = "%s.%r" % (self.number, 1 + len(parts))
            part = SubMessage(self.folder, n, mf)
            parts.append(part)
        mf.pop()
        return parts

    def getbody(self):
        """Return body, either a string or a list of messages."""
        if self.getmaintype() == 'multipart':
            return self.getbodyparts()
        else:
            return self.getbodytext()


class SubMessage(Message):

    def __init__(self, f, n, fp):
        """Constructor."""
        Message.__init__(self, f, n, fp)
        if self.getmaintype() == 'multipart':
            self.body = Message.getbodyparts(self)
        else:
            self.body = Message.getbodytext(self)
        self.bodyencoded = Message.getbodytext(self, decode=0)
            # XXX If this is big, should remember file pointers

    def __repr__(self):
        """String representation."""
        f, n, fp = self.folder, self.number, self.fp
        return 'SubMessage(%s, %s, %s)' % (f, n, fp)

    def getbodytext(self, decode = 1):
        if not decode:
            return self.bodyencoded
        if type(self.body) == type(''):
            return self.body

    def getbodyparts(self):
        if type(self.body) == type([]):
            return self.body

    def getbody(self):
        return self.body


class IntSet:
    """Class implementing sets of integers.

    This is an efficient representation for sets consisting of several
    continuous ranges, e.g. 1-100,200-400,402-1000 is represented
    internally as a list of three pairs: [(1,100), (200,400),
    (402,1000)].  The internal representation is always kept normalized.

    The constructor has up to three arguments:
    - the string used to initialize the set (default ''),
    - the separator between ranges (default ',')
    - the separator between begin and end of a range (default '-')
    The separators must be strings (not regexprs) and should be different.

    The tostring() function yields a string that can be passed to another
    IntSet constructor; __repr__() is a valid IntSet constructor itself.
    """

    # XXX The default begin/end separator means that negative numbers are
    #     not supported very well.
    #
    # XXX There are currently no operations to remove set elements.

    def __init__(self, data = None, sep = ',', rng = '-'):
        self.pairs = []
        self.sep = sep
        self.rng = rng
        if data: self.fromstring(data)

    def reset(self):
        self.pairs = []

    def __cmp__(self, other):
        return cmp(self.pairs, other.pairs)

    def __hash__(self):
        return hash(self.pairs)

    def __repr__(self):
        return 'IntSet(%r, %r, %r)' % (self.tostring(), self.sep, self.rng)

    def normalize(self):
        self.pairs.sort()
        i = 1
        while i < len(self.pairs):
            alo, ahi = self.pairs[i-1]
            blo, bhi = self.pairs[i]
            if ahi >= blo-1:
                self.pairs[i-1:i+1] = [(alo, max(ahi, bhi))]
            else:
                i = i+1

    def tostring(self):
        s = ''
        for lo, hi in self.pairs:
            if lo == hi: t = repr(lo)
            else: t = repr(lo) + self.rng + repr(hi)
            if s: s = s + (self.sep + t)
            else: s = t
        return s

    def tolist(self):
        l = []
        for lo, hi in self.pairs:
            m = range(lo, hi+1)
            l = l + m
        return l

    def fromlist(self, list):
        for i in list:
            self.append(i)

    def clone(self):
        new = IntSet()
        new.pairs = self.pairs[:]
        return new

    def min(self):
        return self.pairs[0][0]

    def max(self):
        return self.pairs[-1][-1]

    def contains(self, x):
        for lo, hi in self.pairs:
            if lo <= x <= hi: return True
        return False

    def append(self, x):
        for i in range(len(self.pairs)):
            lo, hi = self.pairs[i]
            if x < lo: # Need to insert before
                if x+1 == lo:
                    self.pairs[i] = (x, hi)
                else:
                    self.pairs.insert(i, (x, x))
                if i > 0 and x-1 == self.pairs[i-1][1]:
                    # Merge with previous
                    self.pairs[i-1:i+1] = [
                            (self.pairs[i-1][0],
                             self.pairs[i][1])
                          ]
                return
            if x <= hi: # Already in set
                return
        i = len(self.pairs) - 1
        if i >= 0:
            lo, hi = self.pairs[i]
            if x-1 == hi:
                self.pairs[i] = lo, x
                return
        self.pairs.append((x, x))

    def addpair(self, xlo, xhi):
        if xlo > xhi: return
        self.pairs.append((xlo, xhi))
        self.normalize()

    def fromstring(self, data):
        new = []
        for part in data.split(self.sep):
            list = []
            for subp in part.split(self.rng):
                s = subp.strip()
                list.append(int(s))
            if len(list) == 1:
                new.append((list[0], list[0]))
            elif len(list) == 2 and list[0] <= list[1]:
                new.append((list[0], list[1]))
            else:
                raise ValueError, 'bad data passed to IntSet'
        self.pairs = self.pairs + new
        self.normalize()


# Subroutines to read/write entries in .mh_profile and .mh_sequences

def pickline(file, key, casefold = 1):
    try:
        f = open(file, 'r')
    except IOError:
        return None
    pat = re.escape(key) + ':'
    prog = re.compile(pat, casefold and re.IGNORECASE)
    while 1:
        line = f.readline()
        if not line: break
        if prog.match(line):
            text = line[len(key)+1:]
            while 1:
                line = f.readline()
                if not line or not line[0].isspace():
                    break
                text = text + line
            return text.strip()
    return None

def updateline(file, key, value, casefold = 1):
    try:
        f = open(file, 'r')
        lines = f.readlines()
        f.close()
    except IOError:
        lines = []
    pat = re.escape(key) + ':(.*)\n'
    prog = re.compile(pat, casefold and re.IGNORECASE)
    if value is None:
        newline = None
    else:
        newline = '%s: %s\n' % (key, value)
    for i in range(len(lines)):
        line = lines[i]
        if prog.match(line):
            if newline is None:
                del lines[i]
            else:
                lines[i] = newline
            break
    else:
        if newline is not None:
            lines.append(newline)
    tempfile = file + "~"
    f = open(tempfile, 'w')
    for line in lines:
        f.write(line)
    f.close()
    os.rename(tempfile, file)


# Test program

def test():
    global mh, f
    os.system('rm -rf $HOME/Mail/@test')
    mh = MH()
    def do(s): print s; print eval(s)
    do('mh.listfolders()')
    do('mh.listallfolders()')
    testfolders = ['@test', '@test/test1', '@test/test2',
                   '@test/test1/test11', '@test/test1/test12',
                   '@test/test1/test11/test111']
    for t in testfolders: do('mh.makefolder(%r)' % (t,))
    do('mh.listsubfolders(\'@test\')')
    do('mh.listallsubfolders(\'@test\')')
    f = mh.openfolder('@test')
    do('f.listsubfolders()')
    do('f.listallsubfolders()')
    do('f.getsequences()')
    seqs = f.getsequences()
    seqs['foo'] = IntSet('1-10 12-20', ' ').tolist()
    print seqs
    f.putsequences(seqs)
    do('f.getsequences()')
    for t in reversed(testfolders): do('mh.deletefolder(%r)' % (t,))
    do('mh.getcontext()')
    context = mh.getcontext()
    f = mh.openfolder(context)
    do('f.getcurrent()')
    for seq in ('first', 'last', 'cur', '.', 'prev', 'next',
                'first:3', 'last:3', 'cur:3', 'cur:-3',
                'prev:3', 'next:3',
                '1:3', '1:-3', '100:3', '100:-3', '10000:3', '10000:-3',
                'all'):
        try:
            do('f.parsesequence(%r)' % (seq,))
        except Error, msg:
            print "Error:", msg
        stuff = os.popen("pick %r 2>/dev/null" % (seq,)).read()
        list = map(int, stuff.split())
        print list, "<-- pick"
    do('f.listmessages()')


if __name__ == '__main__':
    test()
