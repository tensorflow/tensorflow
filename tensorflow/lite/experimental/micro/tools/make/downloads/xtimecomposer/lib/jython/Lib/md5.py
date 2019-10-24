# $Id: md5.py 39316 2005-08-21 18:45:59Z greg $
#
#  Copyright (C) 2005   Gregory P. Smith (greg@electricrain.com)
#  Licensed to PSF under a Contributor Agreement.

from hashlib import md5
new = md5

blocksize = 1        # legacy value (wrong in any useful sense)
digest_size = 16
