"""A dumb and slow but simple dbm clone.

For database spam, spam.dir contains the index (a text file),
spam.bak *may* contain a backup of the index (also a text file),
while spam.dat contains the data (a binary file).

XXX TO DO:

- seems to contain a bug when updating...

- reclaim free space (currently, space once occupied by deleted or expanded
items is never reused)

- support concurrent access (currently, if two processes take turns making
updates, they can mess up the index)

- support efficient access to large databases (currently, the whole index
is read when the database is opened, and some updates rewrite the whole index)

- support opening for read-only (flag = 'm')

"""

import os as _os
import __builtin__
import UserDict

_open = __builtin__.open

_BLOCKSIZE = 512

error = IOError                         # For anydbm

class _Database(UserDict.DictMixin):

    # The on-disk directory and data files can remain in mutually
    # inconsistent states for an arbitrarily long time (see comments
    # at the end of __setitem__).  This is only repaired when _commit()
    # gets called.  One place _commit() gets called is from __del__(),
    # and if that occurs at program shutdown time, module globals may
    # already have gotten rebound to None.  Since it's crucial that
    # _commit() finish successfully, we can't ignore shutdown races
    # here, and _commit() must not reference any globals.
    _os = _os       # for _commit()
    _open = _open   # for _commit()

    def __init__(self, filebasename, mode):
        self._mode = mode

        # The directory file is a text file.  Each line looks like
        #    "%r, (%d, %d)\n" % (key, pos, siz)
        # where key is the string key, pos is the offset into the dat
        # file of the associated value's first byte, and siz is the number
        # of bytes in the associated value.
        self._dirfile = filebasename + _os.extsep + 'dir'

        # The data file is a binary file pointed into by the directory
        # file, and holds the values associated with keys.  Each value
        # begins at a _BLOCKSIZE-aligned byte offset, and is a raw
        # binary 8-bit string value.
        self._datfile = filebasename + _os.extsep + 'dat'
        self._bakfile = filebasename + _os.extsep + 'bak'

        # The index is an in-memory dict, mirroring the directory file.
        self._index = None  # maps keys to (pos, siz) pairs

        # Mod by Jack: create data file if needed
        try:
            f = _open(self._datfile, 'r')
        except IOError:
            f = _open(self._datfile, 'w', self._mode)
        f.close()
        self._update()

    # Read directory file into the in-memory index dict.
    def _update(self):
        self._index = {}
        try:
            f = _open(self._dirfile)
        except IOError:
            pass
        else:
            for line in f:
                line = line.rstrip()
                key, pos_and_siz_pair = eval(line)
                self._index[key] = pos_and_siz_pair
            f.close()

    # Write the index dict to the directory file.  The original directory
    # file (if any) is renamed with a .bak extension first.  If a .bak
    # file currently exists, it's deleted.
    def _commit(self):
        # CAUTION:  It's vital that _commit() succeed, and _commit() can
        # be called from __del__().  Therefore we must never reference a
        # global in this routine.
        if self._index is None:
            return  # nothing to do

        try:
            self._os.unlink(self._bakfile)
        except self._os.error:
            pass

        try:
            self._os.rename(self._dirfile, self._bakfile)
        except self._os.error:
            pass

        f = self._open(self._dirfile, 'w', self._mode)
        for key, pos_and_siz_pair in self._index.iteritems():
            f.write("%r, %r\n" % (key, pos_and_siz_pair))
        f.close()

    sync = _commit

    def __getitem__(self, key):
        pos, siz = self._index[key]     # may raise KeyError
        f = _open(self._datfile, 'rb')
        f.seek(pos)
        dat = f.read(siz)
        f.close()
        return dat

    # Append val to the data file, starting at a _BLOCKSIZE-aligned
    # offset.  The data file is first padded with NUL bytes (if needed)
    # to get to an aligned offset.  Return pair
    #     (starting offset of val, len(val))
    def _addval(self, val):
        f = _open(self._datfile, 'rb+')
        f.seek(0, 2)
        pos = int(f.tell())
        npos = ((pos + _BLOCKSIZE - 1) // _BLOCKSIZE) * _BLOCKSIZE
        f.write('\0'*(npos-pos))
        pos = npos
        f.write(val)
        f.close()
        return (pos, len(val))

    # Write val to the data file, starting at offset pos.  The caller
    # is responsible for ensuring that there's enough room starting at
    # pos to hold val, without overwriting some other value.  Return
    # pair (pos, len(val)).
    def _setval(self, pos, val):
        f = _open(self._datfile, 'rb+')
        f.seek(pos)
        f.write(val)
        f.close()
        return (pos, len(val))

    # key is a new key whose associated value starts in the data file
    # at offset pos and with length siz.  Add an index record to
    # the in-memory index dict, and append one to the directory file.
    def _addkey(self, key, pos_and_siz_pair):
        self._index[key] = pos_and_siz_pair
        f = _open(self._dirfile, 'a', self._mode)
        f.write("%r, %r\n" % (key, pos_and_siz_pair))
        f.close()

    def __setitem__(self, key, val):
        if not type(key) == type('') == type(val):
            raise TypeError, "keys and values must be strings"
        if key not in self._index:
            self._addkey(key, self._addval(val))
        else:
            # See whether the new value is small enough to fit in the
            # (padded) space currently occupied by the old value.
            pos, siz = self._index[key]
            oldblocks = (siz + _BLOCKSIZE - 1) // _BLOCKSIZE
            newblocks = (len(val) + _BLOCKSIZE - 1) // _BLOCKSIZE
            if newblocks <= oldblocks:
                self._index[key] = self._setval(pos, val)
            else:
                # The new value doesn't fit in the (padded) space used
                # by the old value.  The blocks used by the old value are
                # forever lost.
                self._index[key] = self._addval(val)

            # Note that _index may be out of synch with the directory
            # file now:  _setval() and _addval() don't update the directory
            # file.  This also means that the on-disk directory and data
            # files are in a mutually inconsistent state, and they'll
            # remain that way until _commit() is called.  Note that this
            # is a disaster (for the database) if the program crashes
            # (so that _commit() never gets called).

    def __delitem__(self, key):
        # The blocks used by the associated value are lost.
        del self._index[key]
        # XXX It's unclear why we do a _commit() here (the code always
        # XXX has, so I'm not changing it).  _setitem__ doesn't try to
        # XXX keep the directory file in synch.  Why should we?  Or
        # XXX why shouldn't __setitem__?
        self._commit()

    def keys(self):
        return self._index.keys()

    def has_key(self, key):
        return key in self._index

    def __contains__(self, key):
        return key in self._index

    def iterkeys(self):
        return self._index.iterkeys()
    __iter__ = iterkeys

    def __len__(self):
        return len(self._index)

    def close(self):
        self._commit()
        self._index = self._datfile = self._dirfile = self._bakfile = None

    __del__ = close



def open(file, flag=None, mode=0666):
    """Open the database file, filename, and return corresponding object.

    The flag argument, used to control how the database is opened in the
    other DBM implementations, is ignored in the dumbdbm module; the
    database is always opened for update, and will be created if it does
    not exist.

    The optional mode argument is the UNIX mode of the file, used only when
    the database has to be created.  It defaults to octal code 0666 (and
    will be modified by the prevailing umask).

    """
    # flag argument is currently ignored
    return _Database(file, mode)
