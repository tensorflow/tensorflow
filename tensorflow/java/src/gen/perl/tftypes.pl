#!/usr/bin/perl

use strict;
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
    print "tftypes -d tftypes.csv DataType.java.tmpl > DataType.java\n";
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
    (my $name, my $index, my $jtype, my $jbox, my $creat, my $default, my $desc) =
        split /,/, $line, 7;
    $desc =~ s/^ *//g;
    $desc =~ s/ *$//g;
    $jtypecount{$jtype}++;
    if ($jtypecount{$jtype} > 1) {
# currently allowing Java types to stand for more than one TF type, but
# may want to revisit this.
#       print STDERR "Ambiguous Java type for $name : $jtype\n";
#       exit 1
    }

    push @info, [$name, $index, $jtype, $jbox, $creat, $default, $desc];
}

my $first = 1;

for (my $i = 1; $i <= $#info; $first = 0, $i++) {
    (my $name, my $index, my $jtype, my $jbox, my $creat, my $default, my $desc) =
        @{$info[$i]};
    my $tfname = "TF".$name;
    my $ucname = uc $name;

    if ($option eq '-t') {
        if ($jtype eq '') { next }
        # Generate class declarations
        # print STDERR "Creating $dirname/$tfname.java\n";
        open (CLASSFILE, ">$dirname/$tfname.java") || die "Can't open $tfname.java";
        print CLASSFILE "// GENERATED FILE. Edit tftypes.pl instead.\n";
        print CLASSFILE "package org.tensorflow.types;\n\n";
        print CLASSFILE  "/** The class $tfname represents $desc. */\n"
                        ."public class $tfname implements Types.TFType {\n"
                        ."  /** Represents the type $tfname at run time. */\n"
                        ."  public static final Class<$tfname> T = $tfname.class;\n"
                        ."  static {\n"
                        ."    Types.typeCodes.put($tfname.T, $index);\n"
                        ."  }\n";
        if ($default ne '') {
            print CLASSFILE
                         "  static {\n"
                        ."    Types.scalars.put($tfname.T, $default);\n"
                        ."  }\n";
        }
        print CLASSFILE  "}\n";
        close(CLASSFILE);
    } elsif ($option eq '-d') {
      # Generate datatype enums for DataType.java
      # TODO: implement
      if ($jtype ne '') {
        if (!$first) {
            $typeinfo .= ",\n\n";
        }
        if ($desc ne '') {
            $typeinfo .= "  /** $desc. */\n";
        }
        $typeinfo .=   "  $ucname($index)";
      }
    } elsif ($option eq '-c') { # creators
      # Generate creator declarations for Tensors.java
      if ($jtype ne '' && $creat eq 'y') {
        for (my $brackets = ''; length $brackets <= 12; $brackets .= '[]') {
            $typeinfo .=
                "  public static Tensor<$tfname> create($jtype$brackets data) {\n"
               ."    return Tensor.create(data, $tfname.T);\n"
               ."  }\n";
        }
      }
      if ($text =~ m/\b$tfname\b/ || $creat eq 'y') {
            $imports .= "import org.tensorflow.types.$tfname;\n";
      }
      #if ($text =~ m/\b$ucname\b/ || $creat eq 'y') {
      #  $imports .= "import static org.tensorflow.Types.$ucname;\n";
      #}
    } elsif ($option eq '-T') { # Tensor.java
      if ($text =~ m/\b$tfname\b/) {
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
