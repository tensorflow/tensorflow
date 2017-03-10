from __future__ import division

from datetime import datetime


def total_seconds(td):
    """Python 2.6 compatability"""
    if hasattr(td, 'total_seconds'):
        return td.total_seconds()

    ms = td.microseconds
    secs = (td.seconds + td.days * 24 * 3600)
    return (ms + secs * 10**6) / 10**6


class RedisCache(object):

    def __init__(self, conn):
        self.conn = conn

    def get(self, key):
        return self.conn.get(key)

    def set(self, key, value, expires=None):
        if not expires:
            self.conn.set(key, value)
        else:
            expires = expires - datetime.now()
            self.conn.setex(key, total_seconds(expires), value)

    def delete(self, key):
        self.conn.delete(key)

    def clear(self):
        """Helper for clearing all the keys in a database. Use with
        caution!"""
        for key in self.conn.keys():
            self.conn.delete(key)

    def close(self):
        self.conn.disconnect()
