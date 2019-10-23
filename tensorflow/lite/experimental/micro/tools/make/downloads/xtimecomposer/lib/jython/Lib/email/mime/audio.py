# Copyright (C) 2001-2006 Python Software Foundation
# Author: Anthony Baxter
# Contact: email-sig@python.org

"""Class representing audio/* type MIME documents."""

__all__ = ['MIMEAudio']

import sndhdr

from cStringIO import StringIO
from email import encoders
from email.mime.nonmultipart import MIMENonMultipart



_sndhdr_MIMEmap = {'au'  : 'basic',
                   'wav' :'x-wav',
                   'aiff':'x-aiff',
                   'aifc':'x-aiff',
                   }

# There are others in sndhdr that don't have MIME types. :(
# Additional ones to be added to sndhdr? midi, mp3, realaudio, wma??
def _whatsnd(data):
    """Try to identify a sound file type.

    sndhdr.what() has a pretty cruddy interface, unfortunately.  This is why
    we re-do it here.  It would be easier to reverse engineer the Unix 'file'
    command and use the standard 'magic' file, as shipped with a modern Unix.
    """
    hdr = data[:512]
    fakefile = StringIO(hdr)
    for testfn in sndhdr.tests:
        res = testfn(hdr, fakefile)
        if res is not None:
            return _sndhdr_MIMEmap.get(res[0])
    return None



class MIMEAudio(MIMENonMultipart):
    """Class for generating audio/* MIME documents."""

    def __init__(self, _audiodata, _subtype=None,
                 _encoder=encoders.encode_base64, **_params):
        """Create an audio/* type MIME document.

        _audiodata is a string containing the raw audio data.  If this data
        can be decoded by the standard Python `sndhdr' module, then the
        subtype will be automatically included in the Content-Type header.
        Otherwise, you can specify  the specific audio subtype via the
        _subtype parameter.  If _subtype is not given, and no subtype can be
        guessed, a TypeError is raised.

        _encoder is a function which will perform the actual encoding for
        transport of the image data.  It takes one argument, which is this
        Image instance.  It should use get_payload() and set_payload() to
        change the payload to the encoded form.  It should also add any
        Content-Transfer-Encoding or other headers to the message as
        necessary.  The default encoding is Base64.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        """
        if _subtype is None:
            _subtype = _whatsnd(_audiodata)
        if _subtype is None:
            raise TypeError('Could not find audio MIME subtype')
        MIMENonMultipart.__init__(self, 'audio', _subtype, **_params)
        self.set_payload(_audiodata)
        _encoder(self)
