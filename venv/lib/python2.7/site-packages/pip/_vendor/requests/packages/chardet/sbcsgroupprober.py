######################## BEGIN LICENSE BLOCK ########################
# The Original Code is Mozilla Universal charset detector code.
#
# The Initial Developer of the Original Code is
# Netscape Communications Corporation.
# Portions created by the Initial Developer are Copyright (C) 2001
# the Initial Developer. All Rights Reserved.
#
# Contributor(s):
#   Mark Pilgrim - port to Python
#   Shy Shalom - original C code
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

from .charsetgroupprober import CharSetGroupProber
from .sbcharsetprober import SingleByteCharSetProber
from .langcyrillicmodel import (Win1251CyrillicModel, Koi8rModel,
                                Latin5CyrillicModel, MacCyrillicModel,
                                Ibm866Model, Ibm855Model)
from .langgreekmodel import Latin7GreekModel, Win1253GreekModel
from .langbulgarianmodel import Latin5BulgarianModel, Win1251BulgarianModel
from .langhungarianmodel import Latin2HungarianModel, Win1250HungarianModel
from .langthaimodel import TIS620ThaiModel
from .langhebrewmodel import Win1255HebrewModel
from .hebrewprober import HebrewProber


class SBCSGroupProber(CharSetGroupProber):
    def __init__(self):
        CharSetGroupProber.__init__(self)
        self._mProbers = [
            SingleByteCharSetProber(Win1251CyrillicModel),
            SingleByteCharSetProber(Koi8rModel),
            SingleByteCharSetProber(Latin5CyrillicModel),
            SingleByteCharSetProber(MacCyrillicModel),
            SingleByteCharSetProber(Ibm866Model),
            SingleByteCharSetProber(Ibm855Model),
            SingleByteCharSetProber(Latin7GreekModel),
            SingleByteCharSetProber(Win1253GreekModel),
            SingleByteCharSetProber(Latin5BulgarianModel),
            SingleByteCharSetProber(Win1251BulgarianModel),
            SingleByteCharSetProber(Latin2HungarianModel),
            SingleByteCharSetProber(Win1250HungarianModel),
            SingleByteCharSetProber(TIS620ThaiModel),
        ]
        hebrewProber = HebrewProber()
        logicalHebrewProber = SingleByteCharSetProber(Win1255HebrewModel,
                                                      False, hebrewProber)
        visualHebrewProber = SingleByteCharSetProber(Win1255HebrewModel, True,
                                                     hebrewProber)
        hebrewProber.set_model_probers(logicalHebrewProber, visualHebrewProber)
        self._mProbers.extend([hebrewProber, logicalHebrewProber,
                               visualHebrewProber])

        self.reset()
