This is a stripped down installation directory of the C avro library.

I had to make two custom modifications to hack around tensorflow,
unfortunately. I needed to include the libavro.so file directory
within the pip package, but because this is a symlink, I had to
overwrite it with the contents of libavro.so.22.0.0 (I could not
directly use libavro.so.22.0.0 because tensorflow's MANIFEST.in
includes only shared object files matching *.so).

Secondly, I had to modify the SONAME of the newly copied libavro.so to
be libavro.so, not libavro.so.22.0.0. If you don't know what this
means, don't worry about it.

Since this library has minimal dependencies (zlib and snappy), I will
experimentally try to just provide these binaries with tensorflow and
see if that works.