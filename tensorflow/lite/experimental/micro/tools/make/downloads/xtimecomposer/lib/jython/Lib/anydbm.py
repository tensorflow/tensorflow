"""Generic interface to all dbm clones.

Instead of

        import dbm
        d = dbm.open(file, 'w', 0666)

use

        import anydbm
        d = anydbm.open(file, 'w')

The returned object is a dbhash, gdbm, dbm or dumbdbm object,
dependent on the type of database being opened (determined by whichdb
module) in the case of an existing dbm. If the dbm does not exist and
the create or new flag ('c' or 'n') was specified, the dbm type will
be determined by the availability of the modules (tested in the above
order).

It has the following interface (key and data are strings):

        d[key] = data   # store data at key (may override data at
                        # existing key)
        data = d[key]   # retrieve data at key (raise KeyError if no
                        # such key)
        del d[key]      # delete data stored at key (raises KeyError
                        # if no such key)
        flag = key in d   # true if the key exists
        list = d.keys() # return a list of all existing keys (slow!)

Future versions may change the order in which implementations are
tested for existence, add interfaces to other dbm-like
implementations.

The open function has an optional second argument.  This can be 'r',
for read-only access, 'w', for read-write access of an existing
database, 'c' for read-write access to a new or existing database, and
'n' for read-write access to a new database.  The default is 'r'.

Note: 'r' and 'w' fail if the database doesn't exist; 'c' creates it
only if it doesn't exist; and 'n' always creates a new database.

"""

class error(Exception):
    pass

_names = ['dbhash', 'gdbm', 'dbm', 'dumbdbm']
_errors = [error]
_defaultmod = None

for _name in _names:
    try:
        _mod = __import__(_name)
    except ImportError:
        continue
    if not _defaultmod:
        _defaultmod = _mod
    _errors.append(_mod.error)

if not _defaultmod:
    raise ImportError, "no dbm clone found; tried %s" % _names

error = tuple(_errors)

def open(file, flag = 'r', mode = 0666):
    # guess the type of an existing database
    from whichdb import whichdb
    result=whichdb(file)
    if result is None:
        # db doesn't exist
        if 'c' in flag or 'n' in flag:
            # file doesn't exist and the new
            # flag was used so use default type
            mod = _defaultmod
        else:
            raise error, "need 'c' or 'n' flag to open new db"
    elif result == "":
        # db type cannot be determined
        raise error, "db type could not be determined"
    else:
        mod = __import__(result)
    return mod.open(file, flag, mode)
