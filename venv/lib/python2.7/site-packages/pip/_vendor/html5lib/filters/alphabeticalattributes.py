from __future__ import absolute_import, division, unicode_literals

from . import base

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class Filter(base.Filter):
    def __iter__(self):
        for token in base.Filter.__iter__(self):
            if token["type"] in ("StartTag", "EmptyTag"):
                attrs = OrderedDict()
                for name, value in sorted(token["data"].items(),
                                          key=lambda x: x[0]):
                    attrs[name] = value
                token["data"] = attrs
            yield token
