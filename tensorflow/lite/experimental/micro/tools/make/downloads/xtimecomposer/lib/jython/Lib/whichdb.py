# !/usr/bin/env python
"""Guess which db package to use to open a db file."""

import os
import struct
import sys

try:
    import dbm
    _dbmerror = dbm.error
except ImportError:
    dbm = None
    # just some sort of valid exception which might be raised in the
    # dbm test
    _dbmerror = IOError

def whichdb(filename):
    """Guess which db package to use to open a db file.

    Return values:

    - None if the database file can't be read;
    - empty string if the file can be read but can't be recognized
    - the module name (e.g. "dbm" or "gdbm") if recognized.

    Importing the given module may still fail, and opening the
    database using that module may still fail.
    """

    # Check for dbm first -- this has a .pag and a .dir file
    try:
        f = open(filename + os.extsep + "pag", "rb")
        f.close()
        # dbm linked with gdbm on OS/2 doesn't have .dir file
        if not (dbm.library == "GNU gdbm" and sys.platform == "os2emx"):
            f = open(filename + os.extsep + "dir", "rb")
            f.close()
        return "dbm"
    except IOError:
        # some dbm emulations based on Berkeley DB generate a .db file
        # some do not, but they should be caught by the dbhash checks
        try:
            f = open(filename + os.extsep + "db", "rb")
            f.close()
            # guarantee we can actually open the file using dbm
            # kind of overkill, but since we are dealing with emulations
            # it seems like a prudent step
            if dbm is not None:
                d = dbm.open(filename)
                d.close()
                return "dbm"
        except (IOError, _dbmerror):
            pass

    # Check for dumbdbm next -- this has a .dir and a .dat file
    try:
        # First check for presence of files
        os.stat(filename + os.extsep + "dat")
        size = os.stat(filename + os.extsep + "dir").st_size
        # dumbdbm files with no keys are empty
        if size == 0:
            return "dumbdbm"
        f = open(filename + os.extsep + "dir", "rb")
        try:
            if f.read(1) in ("'", '"'):
                return "dumbdbm"
        finally:
            f.close()
    except (OSError, IOError):
        pass

    # See if the file exists, return None if not
    try:
        f = open(filename, "rb")
    except IOError:
        return None

    # Read the start of the file -- the magic number
    s16 = f.read(16)
    f.close()
    s = s16[0:4]

    # Return "" if not at least 4 bytes
    if len(s) != 4:
        return ""

    # Convert to 4-byte int in native byte order -- return "" if impossible
    try:
        (magic,) = struct.unpack("=l", s)
    except struct.error:
        return ""

    # Check for GNU dbm
    if magic == 0x13579ace:
        return "gdbm"

    # Check for old Berkeley db hash file format v2
    if magic in (0x00061561, 0x61150600):
        return "bsddb185"

    # Later versions of Berkeley db hash file have a 12-byte pad in
    # front of the file type
    try:
        (magic,) = struct.unpack("=l", s16[-4:])
    except struct.error:
        return ""

    # Check for BSD hash
    if magic in (0x00061561, 0x61150600):
        return "dbhash"

    # Unknown
    return ""

if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print whichdb(filename) or "UNKNOWN", filename
