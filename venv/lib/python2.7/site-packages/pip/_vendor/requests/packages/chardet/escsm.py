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

from .constants import eStart, eError, eItsMe

HZ_cls = (
1,0,0,0,0,0,0,0,  # 00 - 07
0,0,0,0,0,0,0,0,  # 08 - 0f
0,0,0,0,0,0,0,0,  # 10 - 17
0,0,0,1,0,0,0,0,  # 18 - 1f
0,0,0,0,0,0,0,0,  # 20 - 27
0,0,0,0,0,0,0,0,  # 28 - 2f
0,0,0,0,0,0,0,0,  # 30 - 37
0,0,0,0,0,0,0,0,  # 38 - 3f
0,0,0,0,0,0,0,0,  # 40 - 47
0,0,0,0,0,0,0,0,  # 48 - 4f
0,0,0,0,0,0,0,0,  # 50 - 57
0,0,0,0,0,0,0,0,  # 58 - 5f
0,0,0,0,0,0,0,0,  # 60 - 67
0,0,0,0,0,0,0,0,  # 68 - 6f
0,0,0,0,0,0,0,0,  # 70 - 77
0,0,0,4,0,5,2,0,  # 78 - 7f
1,1,1,1,1,1,1,1,  # 80 - 87
1,1,1,1,1,1,1,1,  # 88 - 8f
1,1,1,1,1,1,1,1,  # 90 - 97
1,1,1,1,1,1,1,1,  # 98 - 9f
1,1,1,1,1,1,1,1,  # a0 - a7
1,1,1,1,1,1,1,1,  # a8 - af
1,1,1,1,1,1,1,1,  # b0 - b7
1,1,1,1,1,1,1,1,  # b8 - bf
1,1,1,1,1,1,1,1,  # c0 - c7
1,1,1,1,1,1,1,1,  # c8 - cf
1,1,1,1,1,1,1,1,  # d0 - d7
1,1,1,1,1,1,1,1,  # d8 - df
1,1,1,1,1,1,1,1,  # e0 - e7
1,1,1,1,1,1,1,1,  # e8 - ef
1,1,1,1,1,1,1,1,  # f0 - f7
1,1,1,1,1,1,1,1,  # f8 - ff
)

HZ_st = (
eStart,eError,     3,eStart,eStart,eStart,eError,eError,# 00-07
eError,eError,eError,eError,eItsMe,eItsMe,eItsMe,eItsMe,# 08-0f
eItsMe,eItsMe,eError,eError,eStart,eStart,     4,eError,# 10-17
     5,eError,     6,eError,     5,     5,     4,eError,# 18-1f
     4,eError,     4,     4,     4,eError,     4,eError,# 20-27
     4,eItsMe,eStart,eStart,eStart,eStart,eStart,eStart,# 28-2f
)

HZCharLenTable = (0, 0, 0, 0, 0, 0)

HZSMModel = {'classTable': HZ_cls,
             'classFactor': 6,
             'stateTable': HZ_st,
             'charLenTable': HZCharLenTable,
             'name': "HZ-GB-2312"}

ISO2022CN_cls = (
2,0,0,0,0,0,0,0,  # 00 - 07
0,0,0,0,0,0,0,0,  # 08 - 0f
0,0,0,0,0,0,0,0,  # 10 - 17
0,0,0,1,0,0,0,0,  # 18 - 1f
0,0,0,0,0,0,0,0,  # 20 - 27
0,3,0,0,0,0,0,0,  # 28 - 2f
0,0,0,0,0,0,0,0,  # 30 - 37
0,0,0,0,0,0,0,0,  # 38 - 3f
0,0,0,4,0,0,0,0,  # 40 - 47
0,0,0,0,0,0,0,0,  # 48 - 4f
0,0,0,0,0,0,0,0,  # 50 - 57
0,0,0,0,0,0,0,0,  # 58 - 5f
0,0,0,0,0,0,0,0,  # 60 - 67
0,0,0,0,0,0,0,0,  # 68 - 6f
0,0,0,0,0,0,0,0,  # 70 - 77
0,0,0,0,0,0,0,0,  # 78 - 7f
2,2,2,2,2,2,2,2,  # 80 - 87
2,2,2,2,2,2,2,2,  # 88 - 8f
2,2,2,2,2,2,2,2,  # 90 - 97
2,2,2,2,2,2,2,2,  # 98 - 9f
2,2,2,2,2,2,2,2,  # a0 - a7
2,2,2,2,2,2,2,2,  # a8 - af
2,2,2,2,2,2,2,2,  # b0 - b7
2,2,2,2,2,2,2,2,  # b8 - bf
2,2,2,2,2,2,2,2,  # c0 - c7
2,2,2,2,2,2,2,2,  # c8 - cf
2,2,2,2,2,2,2,2,  # d0 - d7
2,2,2,2,2,2,2,2,  # d8 - df
2,2,2,2,2,2,2,2,  # e0 - e7
2,2,2,2,2,2,2,2,  # e8 - ef
2,2,2,2,2,2,2,2,  # f0 - f7
2,2,2,2,2,2,2,2,  # f8 - ff
)

ISO2022CN_st = (
eStart,     3,eError,eStart,eStart,eStart,eStart,eStart,# 00-07
eStart,eError,eError,eError,eError,eError,eError,eError,# 08-0f
eError,eError,eItsMe,eItsMe,eItsMe,eItsMe,eItsMe,eItsMe,# 10-17
eItsMe,eItsMe,eItsMe,eError,eError,eError,     4,eError,# 18-1f
eError,eError,eError,eItsMe,eError,eError,eError,eError,# 20-27
     5,     6,eError,eError,eError,eError,eError,eError,# 28-2f
eError,eError,eError,eItsMe,eError,eError,eError,eError,# 30-37
eError,eError,eError,eError,eError,eItsMe,eError,eStart,# 38-3f
)

ISO2022CNCharLenTable = (0, 0, 0, 0, 0, 0, 0, 0, 0)

ISO2022CNSMModel = {'classTable': ISO2022CN_cls,
                    'classFactor': 9,
                    'stateTable': ISO2022CN_st,
                    'charLenTable': ISO2022CNCharLenTable,
                    'name': "ISO-2022-CN"}

ISO2022JP_cls = (
2,0,0,0,0,0,0,0,  # 00 - 07
0,0,0,0,0,0,2,2,  # 08 - 0f
0,0,0,0,0,0,0,0,  # 10 - 17
0,0,0,1,0,0,0,0,  # 18 - 1f
0,0,0,0,7,0,0,0,  # 20 - 27
3,0,0,0,0,0,0,0,  # 28 - 2f
0,0,0,0,0,0,0,0,  # 30 - 37
0,0,0,0,0,0,0,0,  # 38 - 3f
6,0,4,0,8,0,0,0,  # 40 - 47
0,9,5,0,0,0,0,0,  # 48 - 4f
0,0,0,0,0,0,0,0,  # 50 - 57
0,0,0,0,0,0,0,0,  # 58 - 5f
0,0,0,0,0,0,0,0,  # 60 - 67
0,0,0,0,0,0,0,0,  # 68 - 6f
0,0,0,0,0,0,0,0,  # 70 - 77
0,0,0,0,0,0,0,0,  # 78 - 7f
2,2,2,2,2,2,2,2,  # 80 - 87
2,2,2,2,2,2,2,2,  # 88 - 8f
2,2,2,2,2,2,2,2,  # 90 - 97
2,2,2,2,2,2,2,2,  # 98 - 9f
2,2,2,2,2,2,2,2,  # a0 - a7
2,2,2,2,2,2,2,2,  # a8 - af
2,2,2,2,2,2,2,2,  # b0 - b7
2,2,2,2,2,2,2,2,  # b8 - bf
2,2,2,2,2,2,2,2,  # c0 - c7
2,2,2,2,2,2,2,2,  # c8 - cf
2,2,2,2,2,2,2,2,  # d0 - d7
2,2,2,2,2,2,2,2,  # d8 - df
2,2,2,2,2,2,2,2,  # e0 - e7
2,2,2,2,2,2,2,2,  # e8 - ef
2,2,2,2,2,2,2,2,  # f0 - f7
2,2,2,2,2,2,2,2,  # f8 - ff
)

ISO2022JP_st = (
eStart,     3,eError,eStart,eStart,eStart,eStart,eStart,# 00-07
eStart,eStart,eError,eError,eError,eError,eError,eError,# 08-0f
eError,eError,eError,eError,eItsMe,eItsMe,eItsMe,eItsMe,# 10-17
eItsMe,eItsMe,eItsMe,eItsMe,eItsMe,eItsMe,eError,eError,# 18-1f
eError,     5,eError,eError,eError,     4,eError,eError,# 20-27
eError,eError,eError,     6,eItsMe,eError,eItsMe,eError,# 28-2f
eError,eError,eError,eError,eError,eError,eItsMe,eItsMe,# 30-37
eError,eError,eError,eItsMe,eError,eError,eError,eError,# 38-3f
eError,eError,eError,eError,eItsMe,eError,eStart,eStart,# 40-47
)

ISO2022JPCharLenTable = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

ISO2022JPSMModel = {'classTable': ISO2022JP_cls,
                    'classFactor': 10,
                    'stateTable': ISO2022JP_st,
                    'charLenTable': ISO2022JPCharLenTable,
                    'name': "ISO-2022-JP"}

ISO2022KR_cls = (
2,0,0,0,0,0,0,0,  # 00 - 07
0,0,0,0,0,0,0,0,  # 08 - 0f
0,0,0,0,0,0,0,0,  # 10 - 17
0,0,0,1,0,0,0,0,  # 18 - 1f
0,0,0,0,3,0,0,0,  # 20 - 27
0,4,0,0,0,0,0,0,  # 28 - 2f
0,0,0,0,0,0,0,0,  # 30 - 37
0,0,0,0,0,0,0,0,  # 38 - 3f
0,0,0,5,0,0,0,0,  # 40 - 47
0,0,0,0,0,0,0,0,  # 48 - 4f
0,0,0,0,0,0,0,0,  # 50 - 57
0,0,0,0,0,0,0,0,  # 58 - 5f
0,0,0,0,0,0,0,0,  # 60 - 67
0,0,0,0,0,0,0,0,  # 68 - 6f
0,0,0,0,0,0,0,0,  # 70 - 77
0,0,0,0,0,0,0,0,  # 78 - 7f
2,2,2,2,2,2,2,2,  # 80 - 87
2,2,2,2,2,2,2,2,  # 88 - 8f
2,2,2,2,2,2,2,2,  # 90 - 97
2,2,2,2,2,2,2,2,  # 98 - 9f
2,2,2,2,2,2,2,2,  # a0 - a7
2,2,2,2,2,2,2,2,  # a8 - af
2,2,2,2,2,2,2,2,  # b0 - b7
2,2,2,2,2,2,2,2,  # b8 - bf
2,2,2,2,2,2,2,2,  # c0 - c7
2,2,2,2,2,2,2,2,  # c8 - cf
2,2,2,2,2,2,2,2,  # d0 - d7
2,2,2,2,2,2,2,2,  # d8 - df
2,2,2,2,2,2,2,2,  # e0 - e7
2,2,2,2,2,2,2,2,  # e8 - ef
2,2,2,2,2,2,2,2,  # f0 - f7
2,2,2,2,2,2,2,2,  # f8 - ff
)

ISO2022KR_st = (
eStart,     3,eError,eStart,eStart,eStart,eError,eError,# 00-07
eError,eError,eError,eError,eItsMe,eItsMe,eItsMe,eItsMe,# 08-0f
eItsMe,eItsMe,eError,eError,eError,     4,eError,eError,# 10-17
eError,eError,eError,eError,     5,eError,eError,eError,# 18-1f
eError,eError,eError,eItsMe,eStart,eStart,eStart,eStart,# 20-27
)

ISO2022KRCharLenTable = (0, 0, 0, 0, 0, 0)

ISO2022KRSMModel = {'classTable': ISO2022KR_cls,
                    'classFactor': 6,
                    'stateTable': ISO2022KR_st,
                    'charLenTable': ISO2022KRCharLenTable,
                    'name': "ISO-2022-KR"}

# flake8: noqa
