"""
Access to the Unix group database.

Group entries are reported as 4-tuples containing the following fields
from the group database, in order:

  name   - name of the group
  passwd - group password (encrypted); often empty
  gid    - numeric ID of the group
  mem    - list of members

The gid is an integer, name and password are strings.  (Note that most
users are not explicitly listed as members of the groups they are in
according to the password database.  Check both databases to get
complete membership information.)
"""

__all__ = ['getgrgid', 'getgrnam', 'getgrall']

from os import _name, _posix

if _name == 'nt':
    raise ImportError, 'grp module not supported on Windows'

class struct_group(tuple):
    """
    grp.struct_group: Results from getgr*() routines.

    This object may be accessed either as a tuple of
      (gr_name,gr_passwd,gr_gid,gr_mem)
    or via the object attributes as named in the above tuple.
    """

    attrs = ['gr_name', 'gr_passwd', 'gr_gid', 'gr_mem']

    def __new__(cls, grp):
        return tuple.__new__(cls, (grp.gr_name, grp.gr_passwd, grp.gr_gid,
                                   list(grp.getMembers())))

    def __getattr__(self, attr):
        try:
            return self[self.attrs.index(attr)]
        except ValueError:
            raise AttributeError


def getgrgid(uid):
    """
    getgrgid(id) -> tuple
    Return the group database entry for the given numeric group ID.  If
    id is not valid, raise KeyError.
    """
    entry = _posix.getgrgid(uid)
    if not entry:
        raise KeyError(uid)
    return struct_group(entry)


def getgrnam(name):
    """
    getgrnam(name) -> tuple
    Return the group database entry for the given group name.  If
    name is not valid, raise KeyError.
    """
    entry = _posix.getgrnam(name)
    if not entry:
        raise KeyError(name)
    return struct_group(entry)


def getgrall():
    """
    getgrall() -> list of tuples
    Return a list of all available group database entries,
    in arbitrary order.
    """
    groups = []
    while True:
        group = _posix.getgrent()
        if not group:
            break
        groups.append(struct_group(group))
    return groups
