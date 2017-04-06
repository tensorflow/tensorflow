######################## BEGIN LICENSE BLOCK ########################
# The Original Code is mozilla.org code.
#
# The Initial Developer of the Original Code is
# Netscape Communications Corporation.
# Portions created by the Initial Developer are Copyright (C) 1998
# the Initial Developer. All Rights Reserved.
#
# Contributor(s):
#   Mark Pilgrim - port to Python
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
# 02110-1301  USA
######################### END LICENSE BLOCK #########################

from .constants import eStart
from .compat import wrap_ord


class CodingStateMachine:
    def __init__(self, sm):
        self._mModel = sm
        self._mCurrentBytePos = 0
        self._mCurrentCharLen = 0
        self.reset()

    def reset(self):
        self._mCurrentState = eStart

    def next_state(self, c):
        # for each byte we get its class
        # if it is first byte, we also get byte length
        # PY3K: aBuf is a byte stream, so c is an int, not a byte
        byteCls = self._mModel['classTable'][wrap_ord(c)]
        if self._mCurrentState == eStart:
            self._mCurrentBytePos = 0
            self._mCurrentCharLen = self._mModel['charLenTable'][byteCls]
        # from byte's class and stateTable, we get its next state
        curr_state = (self._mCurrentState * self._mModel['classFactor']
                      + byteCls)
        self._mCurrentState = self._mModel['stateTable'][curr_state]
        self._mCurrentBytePos += 1
        return self._mCurrentState

    def get_current_charlen(self):
        return self._mCurrentCharLen

    def get_coding_state_machine(self):
        return self._mModel['name']
