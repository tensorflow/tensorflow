"""Generic MIME writer.

This module defines the class MimeWriter.  The MimeWriter class implements
a basic formatter for creating MIME multi-part files.  It doesn't seek around
the output file nor does it use large amounts of buffer space. You must write
the parts out in the order that they should occur in the final file.
MimeWriter does buffer the headers you add, allowing you to rearrange their
order.

"""


import mimetools

__all__ = ["MimeWriter"]

class MimeWriter:

    """Generic MIME writer.

    Methods:

    __init__()
    addheader()
    flushheaders()
    startbody()
    startmultipartbody()
    nextpart()
    lastpart()

    A MIME writer is much more primitive than a MIME parser.  It
    doesn't seek around on the output file, and it doesn't use large
    amounts of buffer space, so you have to write the parts in the
    order they should occur on the output file.  It does buffer the
    headers you add, allowing you to rearrange their order.

    General usage is:

    f = <open the output file>
    w = MimeWriter(f)
    ...call w.addheader(key, value) 0 or more times...

    followed by either:

    f = w.startbody(content_type)
    ...call f.write(data) for body data...

    or:

    w.startmultipartbody(subtype)
    for each part:
        subwriter = w.nextpart()
        ...use the subwriter's methods to create the subpart...
    w.lastpart()

    The subwriter is another MimeWriter instance, and should be
    treated in the same way as the toplevel MimeWriter.  This way,
    writing recursive body parts is easy.

    Warning: don't forget to call lastpart()!

    XXX There should be more state so calls made in the wrong order
    are detected.

    Some special cases:

    - startbody() just returns the file passed to the constructor;
      but don't use this knowledge, as it may be changed.

    - startmultipartbody() actually returns a file as well;
      this can be used to write the initial 'if you can read this your
      mailer is not MIME-aware' message.

    - If you call flushheaders(), the headers accumulated so far are
      written out (and forgotten); this is useful if you don't need a
      body part at all, e.g. for a subpart of type message/rfc822
      that's (mis)used to store some header-like information.

    - Passing a keyword argument 'prefix=<flag>' to addheader(),
      start*body() affects where the header is inserted; 0 means
      append at the end, 1 means insert at the start; default is
      append for addheader(), but insert for start*body(), which use
      it to determine where the Content-Type header goes.

    """

    def __init__(self, fp):
        self._fp = fp
        self._headers = []

    def addheader(self, key, value, prefix=0):
        """Add a header line to the MIME message.

        The key is the name of the header, where the value obviously provides
        the value of the header. The optional argument prefix determines
        where the header is inserted; 0 means append at the end, 1 means
        insert at the start. The default is to append.

        """
        lines = value.split("\n")
        while lines and not lines[-1]: del lines[-1]
        while lines and not lines[0]: del lines[0]
        for i in range(1, len(lines)):
            lines[i] = "    " + lines[i].strip()
        value = "\n".join(lines) + "\n"
        line = key + ": " + value
        if prefix:
            self._headers.insert(0, line)
        else:
            self._headers.append(line)

    def flushheaders(self):
        """Writes out and forgets all headers accumulated so far.

        This is useful if you don't need a body part at all; for example,
        for a subpart of type message/rfc822 that's (mis)used to store some
        header-like information.

        """
        self._fp.writelines(self._headers)
        self._headers = []

    def startbody(self, ctype, plist=[], prefix=1):
        """Returns a file-like object for writing the body of the message.

        The content-type is set to the provided ctype, and the optional
        parameter, plist, provides additional parameters for the
        content-type declaration.  The optional argument prefix determines
        where the header is inserted; 0 means append at the end, 1 means
        insert at the start. The default is to insert at the start.

        """
        for name, value in plist:
            ctype = ctype + ';\n %s=\"%s\"' % (name, value)
        self.addheader("Content-Type", ctype, prefix=prefix)
        self.flushheaders()
        self._fp.write("\n")
        return self._fp

    def startmultipartbody(self, subtype, boundary=None, plist=[], prefix=1):
        """Returns a file-like object for writing the body of the message.

        Additionally, this method initializes the multi-part code, where the
        subtype parameter provides the multipart subtype, the boundary
        parameter may provide a user-defined boundary specification, and the
        plist parameter provides optional parameters for the subtype.  The
        optional argument, prefix, determines where the header is inserted;
        0 means append at the end, 1 means insert at the start. The default
        is to insert at the start.  Subparts should be created using the
        nextpart() method.

        """
        self._boundary = boundary or mimetools.choose_boundary()
        return self.startbody("multipart/" + subtype,
                              [("boundary", self._boundary)] + plist,
                              prefix=prefix)

    def nextpart(self):
        """Returns a new instance of MimeWriter which represents an
        individual part in a multipart message.

        This may be used to write the part as well as used for creating
        recursively complex multipart messages. The message must first be
        initialized with the startmultipartbody() method before using the
        nextpart() method.

        """
        self._fp.write("\n--" + self._boundary + "\n")
        return self.__class__(self._fp)

    def lastpart(self):
        """This is used to designate the last part of a multipart message.

        It should always be used when writing multipart messages.

        """
        self._fp.write("\n--" + self._boundary + "--\n")


if __name__ == '__main__':
    import test.test_MimeWriter
