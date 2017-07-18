#!/usr/bin/perl
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

use strict;

my $copyright =
'/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
';

my $count;

my $option = '-t', my $template;

sub usage {
    print "Usage: tftypes [-ctdT] <type desc file> <tmpl file>\n\n"
         ."This script generates parts of various .java files that depend on which"
         ."TensorFlow types are supported by the Java API and how much. For each"
         ."such .java file, there is a .tmpl file in the same source directory in"
         ."which the strings \@TYPEINFO\@ and \@IMPORTS\@ are replaced with"
         ."appropriate Java code. Output code is sent to standard output.\n\n";

    print "Modulo putting in the correct directory names, it can be invoked as follows:\n";
    print "tftypes -c tftypes.csv Tensors.java.tmpl > Tensors.java\n";
    print "tftypes -t tftypes.csv <dir>                                   [outputs files to dir]\n";
}

if ($ARGV[0] =~ m/^-/) {
    $option = shift;
}
my $typedesc = shift;
my $tmpl = shift;

my $dirname;

if ($option eq '-t') {
    $dirname = $tmpl;
}

open (TMPL, "<$tmpl") || die "Cannot open $tmpl for reading\n";

my $text = do { local $/; <TMPL> };

my %jtypecount;

my $typeinfo, my $imports;

open (TYPEDESC, $typedesc);

my @info = ([]);

while (<TYPEDESC>) {
    chomp;
    my $line = $_;
    if ($line =~ m/^TF type/) { next }
    $line =~ s/\r$//;
    (my $name, my $jtype, my $creat, my $default, my $desc) =
        split /,/, $line, 5;
    $desc =~ s/^ *//g;
    $desc =~ s/ *$//g;
    $jtypecount{$jtype}++;
    if ($jtypecount{$jtype} > 1) {
# currently allowing Java types to stand for more than one TF type, but
# may want to revisit this.
#       print STDERR "Ambiguous Java type for $name : $jtype\n";
#       exit 1
    }

    push @info, [$name, $jtype, $creat, $default, $desc];
}

for (my $i = 1; $i <= $#info; $i++) {
    (my $name, my $jtype, my $creat, my $default, my $desc) =
        @{$info[$i]};
    my $tfname = "TF".$name;
    my $ucname = uc $name;

    if ($option eq '-t') {
        if ($jtype eq '') { next }
        # Generate class declarations
        # print STDERR "Creating $dirname/$tfname.java\n";
        open (CLASSFILE, ">$dirname/$tfname.java") || die "Can't open $tfname.java";
        print CLASSFILE $copyright;
        print CLASSFILE "// GENERATED FILE. To update, edit tftypes.pl instead.\n\n";

        my $fulldesc = $desc;
        if (substr($desc, 0, 1) =~ m/^[aeoiu8]$/i) {
            $fulldesc = "an $desc"
        } else {
            $fulldesc = "a $desc"
        }
        print CLASSFILE  "package org.tensorflow.types;\n\n"
                        ."import org.tensorflow.DataType;\n\n";
        print CLASSFILE  "/** Represents $fulldesc. */\n"
                        ."public class $tfname implements TFType {\n"
                        ."  private $tfname() {}\n"
                        ."  static {\n"
                        ."    Types.typeCodes.put($tfname.class, DataType.$ucname);\n"
                        ."  }\n";
        if ($default ne '') {
            print CLASSFILE
                         "  static {\n"
                        ."    Types.scalars.put($tfname.class, $default);\n"
                        ."  }\n";
        }
        print CLASSFILE  "}\n";
        close(CLASSFILE);
    } elsif ($option eq '-c') {
      # Generate creator declarations for Tensors.java
      if ($jtype ne '' && $creat eq 'y') {
        for (my $brackets = ''; length $brackets <= 12; $brackets .= '[]') {
            $typeinfo .=
                "  public static Tensor<$tfname> create($jtype$brackets data) {\n"
               ."    return Tensor.create(data, $tfname.class);\n"
               ."  }\n";
        }
      }
      if ($text =~ m/\b$tfname\b/ || $creat eq 'y') {
            $imports .= "import org.tensorflow.types.$tfname;\n";
      }
    }
}

if ($option ne '-t') {
  print "// GENERATED FILE. Edits to this file will be lost -- edit $tmpl instead.\n";

  $text =~ s/\@TYPEINFO\@/$typeinfo/;
  $text =~ s/\@IMPORTS\@/$imports/;

  print $text;
}
