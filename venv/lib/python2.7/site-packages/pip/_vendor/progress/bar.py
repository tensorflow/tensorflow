# -*- coding: utf-8 -*-

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

from . import Progress
from .helpers import WritelnMixin


class Bar(WritelnMixin, Progress):
    width = 32
    message = ''
    suffix = '%(index)d/%(max)d'
    bar_prefix = ' |'
    bar_suffix = '| '
    empty_fill = ' '
    fill = '#'
    hide_cursor = True

    def update(self):
        filled_length = int(self.width * self.progress)
        empty_length = self.width - filled_length

        message = self.message % self
        bar = self.fill * filled_length
        empty = self.empty_fill * empty_length
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, empty, self.bar_suffix,
                        suffix])
        self.writeln(line)


class ChargingBar(Bar):
    suffix = '%(percent)d%%'
    bar_prefix = ' '
    bar_suffix = ' '
    empty_fill = u'∙'
    fill = u'█'


class FillingSquaresBar(ChargingBar):
    empty_fill = u'▢'
    fill = u'▣'


class FillingCirclesBar(ChargingBar):
    empty_fill = u'◯'
    fill = u'◉'


class IncrementalBar(Bar):
    phases = (u' ', u'▏', u'▎', u'▍', u'▌', u'▋', u'▊', u'▉', u'█')

    def update(self):
        nphases = len(self.phases)
        expanded_length = int(nphases * self.width * self.progress)
        filled_length = int(self.width * self.progress)
        empty_length = self.width - filled_length
        phase = expanded_length - (filled_length * nphases)

        message = self.message % self
        bar = self.phases[-1] * filled_length
        current = self.phases[phase] if phase > 0 else ''
        empty = self.empty_fill * max(0, empty_length - len(current))
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, current, empty,
                        self.bar_suffix, suffix])
        self.writeln(line)


class ShadyBar(IncrementalBar):
    phases = (u' ', u'░', u'▒', u'▓', u'█')
