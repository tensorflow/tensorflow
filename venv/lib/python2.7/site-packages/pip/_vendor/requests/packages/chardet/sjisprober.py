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

import sys
from .mbcharsetprober import MultiByteCharSetProber
from .codingstatemachine import CodingStateMachine
from .chardistribution import SJISDistributionAnalysis
from .jpcntx import SJISContextAnalysis
from .mbcssm import SJISSMModel
from . import constants


class SJISProber(MultiByteCharSetProber):
    def __init__(self):
        MultiByteCharSetProber.__init__(self)
        self._mCodingSM = CodingStateMachine(SJISSMModel)
        self._mDistributionAnalyzer = SJISDistributionAnalysis()
        self._mContextAnalyzer = SJISContextAnalysis()
        self.reset()

    def reset(self):
        MultiByteCharSetProber.reset(self)
        self._mContextAnalyzer.reset()

    def get_charset_name(self):
        return self._mContextAnalyzer.get_charset_name()

    def feed(self, aBuf):
        aLen = len(aBuf)
        for i in range(0, aLen):
            codingState = self._mCodingSM.next_state(aBuf[i])
            if codingState == constants.eError:
                if constants._debug:
                    sys.stderr.write(self.get_charset_name()
                                     + ' prober hit error at byte ' + str(i)
                                     + '\n')
                self._mState = constants.eNotMe
                break
            elif codingState == constants.eItsMe:
                self._mState = constants.eFoundIt
                break
            elif codingState == constants.eStart:
                charLen = self._mCodingSM.get_current_charlen()
                if i == 0:
                    self._mLastChar[1] = aBuf[0]
                    self._mContextAnalyzer.feed(self._mLastChar[2 - charLen:],
                                                charLen)
                    self._mDistributionAnalyzer.feed(self._mLastChar, charLen)
                else:
                    self._mContextAnalyzer.feed(aBuf[i + 1 - charLen:i + 3
                                                     - charLen], charLen)
                    self._mDistributionAnalyzer.feed(aBuf[i - 1:i + 1],
                                                     charLen)

        self._mLastChar[0] = aBuf[aLen - 1]

        if self.get_state() == constants.eDetecting:
            if (self._mContextAnalyzer.got_enough_data() and
               (self.get_confidence() > constants.SHORTCUT_THRESHOLD)):
                self._mState = constants.eFoundIt

        return self.get_state()

    def get_confidence(self):
        contxtCf = self._mContextAnalyzer.get_confidence()
        distribCf = self._mDistributionAnalyzer.get_confidence()
        return max(contxtCf, distribCf)
