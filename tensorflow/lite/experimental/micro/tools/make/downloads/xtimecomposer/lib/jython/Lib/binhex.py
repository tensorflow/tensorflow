"""Macintosh binhex compression/decompression.

easy interface:
binhex(inputfilename, outputfilename)
hexbin(inputfilename, outputfilename)
"""

#
# Jack Jansen, CWI, August 1995.
#
# The module is supposed to be as compatible as possible. Especially the
# easy interface should work "as expected" on any platform.
# XXXX Note: currently, textfiles appear in mac-form on all platforms.
# We seem to lack a simple character-translate in python.
# (we should probably use ISO-Latin-1 on all but the mac platform).
# XXXX The simple routines are too simple: they expect to hold the complete
# files in-core. Should be fixed.
# XXXX It would be nice to handle AppleDouble format on unix
# (for servers serving macs).
# XXXX I don't understand what happens when you get 0x90 times the same byte on
# input. The resulting code (xx 90 90) would appear to be interpreted as an
# escaped *value* of 0x90. All coders I've seen appear to ignore this nicety...
#
import sys
import os
import struct
import binascii

__all__ = ["binhex","hexbin","Error"]

class Error(Exception):
    pass

# States (what have we written)
[_DID_HEADER, _DID_DATA, _DID_RSRC] = range(3)

# Various constants
REASONABLY_LARGE=32768  # Minimal amount we pass the rle-coder
LINELEN=64
RUNCHAR=chr(0x90)   # run-length introducer

#
# This code is no longer byte-order dependent

#
# Workarounds for non-mac machines.
try:
    from Carbon.File import FSSpec, FInfo
    from MacOS import openrf

    def getfileinfo(name):
        finfo = FSSpec(name).FSpGetFInfo()
        dir, file = os.path.split(name)
        # XXX Get resource/data sizes
        fp = open(name, 'rb')
        fp.seek(0, 2)
        dlen = fp.tell()
        fp = openrf(name, '*rb')
        fp.seek(0, 2)
        rlen = fp.tell()
        return file, finfo, dlen, rlen

    def openrsrc(name, *mode):
        if not mode:
            mode = '*rb'
        else:
            mode = '*' + mode[0]
        return openrf(name, mode)

except ImportError:
    #
    # Glue code for non-macintosh usage
    #

    class FInfo:
        def __init__(self):
            self.Type = '????'
            self.Creator = '????'
            self.Flags = 0

    def getfileinfo(name):
        finfo = FInfo()
        # Quick check for textfile
        fp = open(name)
        data = open(name).read(256)
        for c in data:
            if not c.isspace() and (c<' ' or ord(c) > 0x7f):
                break
        else:
            finfo.Type = 'TEXT'
        fp.seek(0, 2)
        dsize = fp.tell()
        fp.close()
        dir, file = os.path.split(name)
        file = file.replace(':', '-', 1)
        return file, finfo, dsize, 0

    class openrsrc:
        def __init__(self, *args):
            pass

        def read(self, *args):
            return ''

        def write(self, *args):
            pass

        def close(self):
            pass

class _Hqxcoderengine:
    """Write data to the coder in 3-byte chunks"""

    def __init__(self, ofp):
        self.ofp = ofp
        self.data = ''
        self.hqxdata = ''
        self.linelen = LINELEN-1

    def write(self, data):
        self.data = self.data + data
        datalen = len(self.data)
        todo = (datalen//3)*3
        data = self.data[:todo]
        self.data = self.data[todo:]
        if not data:
            return
        self.hqxdata = self.hqxdata + binascii.b2a_hqx(data)
        self._flush(0)

    def _flush(self, force):
        first = 0
        while first <= len(self.hqxdata)-self.linelen:
            last = first + self.linelen
            self.ofp.write(self.hqxdata[first:last]+'\n')
            self.linelen = LINELEN
            first = last
        self.hqxdata = self.hqxdata[first:]
        if force:
            self.ofp.write(self.hqxdata + ':\n')

    def close(self):
        if self.data:
            self.hqxdata = \
                 self.hqxdata + binascii.b2a_hqx(self.data)
        self._flush(1)
        self.ofp.close()
        del self.ofp

class _Rlecoderengine:
    """Write data to the RLE-coder in suitably large chunks"""

    def __init__(self, ofp):
        self.ofp = ofp
        self.data = ''

    def write(self, data):
        self.data = self.data + data
        if len(self.data) < REASONABLY_LARGE:
            return
        rledata = binascii.rlecode_hqx(self.data)
        self.ofp.write(rledata)
        self.data = ''

    def close(self):
        if self.data:
            rledata = binascii.rlecode_hqx(self.data)
            self.ofp.write(rledata)
        self.ofp.close()
        del self.ofp

class BinHex:
    def __init__(self, (name, finfo, dlen, rlen), ofp):
        if type(ofp) == type(''):
            ofname = ofp
            ofp = open(ofname, 'w')
            if os.name == 'mac':
                fss = FSSpec(ofname)
                fss.SetCreatorType('BnHq', 'TEXT')
        ofp.write('(This file must be converted with BinHex 4.0)\n\n:')
        hqxer = _Hqxcoderengine(ofp)
        self.ofp = _Rlecoderengine(hqxer)
        self.crc = 0
        if finfo is None:
            finfo = FInfo()
        self.dlen = dlen
        self.rlen = rlen
        self._writeinfo(name, finfo)
        self.state = _DID_HEADER

    def _writeinfo(self, name, finfo):
        nl = len(name)
        if nl > 63:
            raise Error, 'Filename too long'
        d = chr(nl) + name + '\0'
        d2 = finfo.Type + finfo.Creator

        # Force all structs to be packed with big-endian
        d3 = struct.pack('>h', finfo.Flags)
        d4 = struct.pack('>ii', self.dlen, self.rlen)
        info = d + d2 + d3 + d4
        self._write(info)
        self._writecrc()

    def _write(self, data):
        self.crc = binascii.crc_hqx(data, self.crc)
        self.ofp.write(data)

    def _writecrc(self):
        # XXXX Should this be here??
        # self.crc = binascii.crc_hqx('\0\0', self.crc)
        if self.crc < 0:
            fmt = '>h'
        else:
            fmt = '>H'
        self.ofp.write(struct.pack(fmt, self.crc))
        self.crc = 0

    def write(self, data):
        if self.state != _DID_HEADER:
            raise Error, 'Writing data at the wrong time'
        self.dlen = self.dlen - len(data)
        self._write(data)

    def close_data(self):
        if self.dlen != 0:
            raise Error, 'Incorrect data size, diff=%r' % (self.rlen,)
        self._writecrc()
        self.state = _DID_DATA

    def write_rsrc(self, data):
        if self.state < _DID_DATA:
            self.close_data()
        if self.state != _DID_DATA:
            raise Error, 'Writing resource data at the wrong time'
        self.rlen = self.rlen - len(data)
        self._write(data)

    def close(self):
        if self.state < _DID_DATA:
            self.close_data()
        if self.state != _DID_DATA:
            raise Error, 'Close at the wrong time'
        if self.rlen != 0:
            raise Error, \
                  "Incorrect resource-datasize, diff=%r" % (self.rlen,)
        self._writecrc()
        self.ofp.close()
        self.state = None
        del self.ofp

def binhex(inp, out):
    """(infilename, outfilename) - Create binhex-encoded copy of a file"""
    finfo = getfileinfo(inp)
    ofp = BinHex(finfo, out)

    ifp = open(inp, 'rb')
    # XXXX Do textfile translation on non-mac systems
    while 1:
        d = ifp.read(128000)
        if not d: break
        ofp.write(d)
    ofp.close_data()
    ifp.close()

    ifp = openrsrc(inp, 'rb')
    while 1:
        d = ifp.read(128000)
        if not d: break
        ofp.write_rsrc(d)
    ofp.close()
    ifp.close()

class _Hqxdecoderengine:
    """Read data via the decoder in 4-byte chunks"""

    def __init__(self, ifp):
        self.ifp = ifp
        self.eof = 0

    def read(self, totalwtd):
        """Read at least wtd bytes (or until EOF)"""
        decdata = ''
        wtd = totalwtd
        #
        # The loop here is convoluted, since we don't really now how
        # much to decode: there may be newlines in the incoming data.
        while wtd > 0:
            if self.eof: return decdata
            wtd = ((wtd+2)//3)*4
            data = self.ifp.read(wtd)
            #
            # Next problem: there may not be a complete number of
            # bytes in what we pass to a2b. Solve by yet another
            # loop.
            #
            while 1:
                try:
                    decdatacur, self.eof = \
                            binascii.a2b_hqx(data)
                    break
                except binascii.Incomplete:
                    pass
                newdata = self.ifp.read(1)
                if not newdata:
                    raise Error, \
                          'Premature EOF on binhex file'
                data = data + newdata
            decdata = decdata + decdatacur
            wtd = totalwtd - len(decdata)
            if not decdata and not self.eof:
                raise Error, 'Premature EOF on binhex file'
        return decdata

    def close(self):
        self.ifp.close()

class _Rledecoderengine:
    """Read data via the RLE-coder"""

    def __init__(self, ifp):
        self.ifp = ifp
        self.pre_buffer = ''
        self.post_buffer = ''
        self.eof = 0

    def read(self, wtd):
        if wtd > len(self.post_buffer):
            self._fill(wtd-len(self.post_buffer))
        rv = self.post_buffer[:wtd]
        self.post_buffer = self.post_buffer[wtd:]
        return rv

    def _fill(self, wtd):
        self.pre_buffer = self.pre_buffer + self.ifp.read(wtd+4)
        if self.ifp.eof:
            self.post_buffer = self.post_buffer + \
                binascii.rledecode_hqx(self.pre_buffer)
            self.pre_buffer = ''
            return

        #
        # Obfuscated code ahead. We have to take care that we don't
        # end up with an orphaned RUNCHAR later on. So, we keep a couple
        # of bytes in the buffer, depending on what the end of
        # the buffer looks like:
        # '\220\0\220' - Keep 3 bytes: repeated \220 (escaped as \220\0)
        # '?\220' - Keep 2 bytes: repeated something-else
        # '\220\0' - Escaped \220: Keep 2 bytes.
        # '?\220?' - Complete repeat sequence: decode all
        # otherwise: keep 1 byte.
        #
        mark = len(self.pre_buffer)
        if self.pre_buffer[-3:] == RUNCHAR + '\0' + RUNCHAR:
            mark = mark - 3
        elif self.pre_buffer[-1] == RUNCHAR:
            mark = mark - 2
        elif self.pre_buffer[-2:] == RUNCHAR + '\0':
            mark = mark - 2
        elif self.pre_buffer[-2] == RUNCHAR:
            pass # Decode all
        else:
            mark = mark - 1

        self.post_buffer = self.post_buffer + \
            binascii.rledecode_hqx(self.pre_buffer[:mark])
        self.pre_buffer = self.pre_buffer[mark:]

    def close(self):
        self.ifp.close()

class HexBin:
    def __init__(self, ifp):
        if type(ifp) == type(''):
            ifp = open(ifp)
        #
        # Find initial colon.
        #
        while 1:
            ch = ifp.read(1)
            if not ch:
                raise Error, "No binhex data found"
            # Cater for \r\n terminated lines (which show up as \n\r, hence
            # all lines start with \r)
            if ch == '\r':
                continue
            if ch == ':':
                break
            if ch != '\n':
                dummy = ifp.readline()

        hqxifp = _Hqxdecoderengine(ifp)
        self.ifp = _Rledecoderengine(hqxifp)
        self.crc = 0
        self._readheader()

    def _read(self, len):
        data = self.ifp.read(len)
        self.crc = binascii.crc_hqx(data, self.crc)
        return data

    def _checkcrc(self):
        filecrc = struct.unpack('>h', self.ifp.read(2))[0] & 0xffff
        #self.crc = binascii.crc_hqx('\0\0', self.crc)
        # XXXX Is this needed??
        self.crc = self.crc & 0xffff
        if filecrc != self.crc:
            raise Error, 'CRC error, computed %x, read %x' \
                  %(self.crc, filecrc)
        self.crc = 0

    def _readheader(self):
        len = self._read(1)
        fname = self._read(ord(len))
        rest = self._read(1+4+4+2+4+4)
        self._checkcrc()

        type = rest[1:5]
        creator = rest[5:9]
        flags = struct.unpack('>h', rest[9:11])[0]
        self.dlen = struct.unpack('>l', rest[11:15])[0]
        self.rlen = struct.unpack('>l', rest[15:19])[0]

        self.FName = fname
        self.FInfo = FInfo()
        self.FInfo.Creator = creator
        self.FInfo.Type = type
        self.FInfo.Flags = flags

        self.state = _DID_HEADER

    def read(self, *n):
        if self.state != _DID_HEADER:
            raise Error, 'Read data at wrong time'
        if n:
            n = n[0]
            n = min(n, self.dlen)
        else:
            n = self.dlen
        rv = ''
        while len(rv) < n:
            rv = rv + self._read(n-len(rv))
        self.dlen = self.dlen - n
        return rv

    def close_data(self):
        if self.state != _DID_HEADER:
            raise Error, 'close_data at wrong time'
        if self.dlen:
            dummy = self._read(self.dlen)
        self._checkcrc()
        self.state = _DID_DATA

    def read_rsrc(self, *n):
        if self.state == _DID_HEADER:
            self.close_data()
        if self.state != _DID_DATA:
            raise Error, 'Read resource data at wrong time'
        if n:
            n = n[0]
            n = min(n, self.rlen)
        else:
            n = self.rlen
        self.rlen = self.rlen - n
        return self._read(n)

    def close(self):
        if self.rlen:
            dummy = self.read_rsrc(self.rlen)
        self._checkcrc()
        self.state = _DID_RSRC
        self.ifp.close()

def hexbin(inp, out):
    """(infilename, outfilename) - Decode binhexed file"""
    ifp = HexBin(inp)
    finfo = ifp.FInfo
    if not out:
        out = ifp.FName
    if os.name == 'mac':
        ofss = FSSpec(out)
        out = ofss.as_pathname()

    ofp = open(out, 'wb')
    # XXXX Do translation on non-mac systems
    while 1:
        d = ifp.read(128000)
        if not d: break
        ofp.write(d)
    ofp.close()
    ifp.close_data()

    d = ifp.read_rsrc(128000)
    if d:
        ofp = openrsrc(out, 'wb')
        ofp.write(d)
        while 1:
            d = ifp.read_rsrc(128000)
            if not d: break
            ofp.write(d)
        ofp.close()

    if os.name == 'mac':
        nfinfo = ofss.GetFInfo()
        nfinfo.Creator = finfo.Creator
        nfinfo.Type = finfo.Type
        nfinfo.Flags = finfo.Flags
        ofss.SetFInfo(nfinfo)

    ifp.close()

def _test():
    if os.name == 'mac':
        import macfs
        fss, ok = macfs.PromptGetFile('File to convert:')
        if not ok:
            sys.exit(0)
        fname = fss.as_pathname()
    else:
        fname = sys.argv[1]
    binhex(fname, fname+'.hqx')
    hexbin(fname+'.hqx', fname+'.viahqx')
    #hexbin(fname, fname+'.unpacked')
    sys.exit(1)

if __name__ == '__main__':
    _test()
