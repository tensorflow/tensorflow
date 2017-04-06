from __future__ import absolute_import, division, unicode_literals

from collections import Mapping


class Trie(Mapping):
    """Abstract base class for tries"""

    def keys(self, prefix=None):
        # pylint:disable=arguments-differ
        keys = super(Trie, self).keys()

        if prefix is None:
            return set(keys)

        # Python 2.6: no set comprehensions
        return set([x for x in keys if x.startswith(prefix)])

    def has_keys_with_prefix(self, prefix):
        for key in self.keys():
            if key.startswith(prefix):
                return True

        return False

    def longest_prefix(self, prefix):
        if prefix in self:
            return prefix

        for i in range(1, len(prefix) + 1):
            if prefix[:-i] in self:
                return prefix[:-i]

        raise KeyError(prefix)

    def longest_prefix_item(self, prefix):
        lprefix = self.longest_prefix(prefix)
        return (lprefix, self[lprefix])
