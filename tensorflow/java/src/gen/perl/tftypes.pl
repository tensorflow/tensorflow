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

sub trim {
    (my $ret) = @_;
    $ret =~ s/^\s*//g;
    $ret =~ s/\s*$//g;
    return $ret;
}

while (<TYPEDESC>) {
    chomp;
    my $line = $_;
    if ($line =~ m/^TF type/) { next }
    $line =~ s/\r$//;
    my @items = split /,/, $line, 6;
    for (my $i = 0; $i <= $#items; $i++) {
        $items[$i] = trim $items[$i];
    }
    my $jtype = $items[2];
    $jtypecount{$jtype}++;
    if ($jtypecount{$jtype} > 1) {
# currently allowing Java types to stand for more than one TF type, but
# may want to revisit this.
#       print STDERR "Ambiguous Java type for $name : $jtype\n";
#       exit 1
    }

    push @info, \@items;
}

sub article {
    (my $s) = @_;
    if (substr($s, 0, 1) =~ m/^[aeoiu8]$/i) {
        return "an $s"
    } else {
        return "a $s"
    }
}

for (my $i = 1; $i <= $#info; $i++) {
    (my $name, my $builtin, my $jtype, my $creat, my $default, my $desc) =
        @{$info[$i]};
    my $tfname = $name;
    my $ucname = uc $name;

    print STDERR "$name $desc\n";

    if ($option eq '-t') {
        if ($jtype eq '') { next }
        if ($builtin eq 'y') { next }
        # Generate class declarations
        # print STDERR "Creating $dirname/$tfname.java\n";
        open (CLASSFILE, ">$dirname/$tfname.java") || die "Can't open $tfname.java";
        print CLASSFILE $copyright, "\n";
        # print CLASSFILE "// GENERATED FILE. To update, edit tftypes.pl instead.\n\n";

        my $fulldesc = article($desc);
        print CLASSFILE  "package org.tensorflow.types;\n\n";
        print CLASSFILE  "/** Represents $fulldesc. */\n"
                        ."public class $tfname {\n"
                        ."  private $tfname() {\n"
                        ."  }\n"
                        ."}\n";
        close(CLASSFILE);
    } elsif ($option eq '-c') {
      # Generate creator declarations for Tensors.java
      if ($jtype ne '' && $creat eq 'y') {
        for (my $brackets = '', my $rank = 0; length $brackets <= 12; $brackets .= '[]', $rank++) {
            my $datainfo = "   *  \@param data An array containing the values to put into the new tensor.\n"
                          ."   *  The dimensions of the new tensor will match those of the array.\n";
            if ($rank == 0) {
                $datainfo = "   *  \@param data The value to put into the new scalar tensor.\n"
            }

            my $trank = $rank;
            if ($tfname eq 'String') {
                $trank = $rank-1;
                next if $trank < 0;

                $datainfo = "   *  \@param data An array containing the data to put into the new tensor.\n"
                           ."   *  String elements are sequences of bytes from the last array dimension.\n";
            }

    
            my $intro = ($trank > 0)
                ?  "Creates a rank-$trank tensor of {\@code $jtype} elements."
                :  "Creates a scalar tensor containing a single {\@code $jtype} element.";
            $typeinfo .=
             "  /**\n"
            ."   * $intro\n"
            ."   * \n"
            .$datainfo
            ."   */\n"
            ."  public static Tensor<$tfname> create($jtype$brackets data) {\n"
            ."    return Tensor.create(data, $tfname.class);\n"
            ."  }\n\n";
        }
      }
      if ($text =~ m/\b$tfname\b/ && $builtin eq 'n' && $creat eq 'y') {
            $imports .= "import org.tensorflow.types.$tfname;\n";
      }
    }
}

if ($option ne '-t') {
# print "// GENERATED FILE. Edits to this file will be lost -- edit $tmpl instead.\n";

  $text =~ s/\@TYPEINFO\@/$typeinfo/;
  $text =~ s/\@IMPORTS\@/$imports/;

  print $text;
}
