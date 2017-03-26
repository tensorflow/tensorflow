# Copyright (c) 2012 Giorgos Verigakis <verigak@gmail.com>
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from __future__ import division

from collections import deque
from datetime import timedelta
from math import ceil
from sys import stderr
from time import time


__version__ = '1.2'


class Infinite(object):
    file = stderr
    sma_window = 10

    def __init__(self, *args, **kwargs):
        self.index = 0
        self.start_ts = time()
        self._ts = self.start_ts
        self._dt = deque(maxlen=self.sma_window)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, key):
        if key.startswith('_'):
            return None
        return getattr(self, key, None)

    @property
    def avg(self):
        return sum(self._dt) / len(self._dt) if self._dt else 0

    @property
    def elapsed(self):
        return int(time() - self.start_ts)

    @property
    def elapsed_td(self):
        return timedelta(seconds=self.elapsed)

    def update(self):
        pass

    def start(self):
        pass

    def finish(self):
        pass

    def next(self, n=1):
        if n > 0:
            now = time()
            dt = (now - self._ts) / n
            self._dt.append(dt)
            self._ts = now

        self.index = self.index + n
        self.update()

    def iter(self, it):
        for x in it:
            yield x
            self.next()
        self.finish()


class Progress(Infinite):
    def __init__(self, *args, **kwargs):
        super(Progress, self).__init__(*args, **kwargs)
        self.max = kwargs.get('max', 100)

    @property
    def eta(self):
        return int(ceil(self.avg * self.remaining))

    @property
    def eta_td(self):
        return timedelta(seconds=self.eta)

    @property
    def percent(self):
        return self.progress * 100

    @property
    def progress(self):
        return min(1, self.index / self.max)

    @property
    def remaining(self):
        return max(self.max - self.index, 0)

    def start(self):
        self.update()

    def goto(self, index):
        incr = index - self.index
        self.next(incr)

    def iter(self, it):
        try:
            self.max = len(it)
        except TypeError:
            pass

        for x in it:
            yield x
            self.next()
        self.finish()
