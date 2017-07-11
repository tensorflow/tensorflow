#!/usr/bin/perl

my $script = $0;
my $dir = `dirname $script`;
chomp $dir;
my $gen = "$dir/..";
my $tfjavasrc = "$gen/..";
my $rsrc = "$gen/resources";
my $root = "$tfjavasrc/main/java";
my $pkg = "$root/org/tensorflow";

sub locchk {
    (my $f) = @_;
    if (! -r $f) {
        print STDERR "Script tftypes-runall seems to be located in the wrong place (could not find $f)\n";
        exit 1;
    }
}
&locchk("$gen");
&locchk("$tfjavasrc/gen");
&locchk("$dir/tftypes.pl");
&locchk("$rsrc/tftypes.csv");

system("perl $dir/tftypes.pl -t $rsrc/tftypes.csv $pkg/types");
system("perl $dir/tftypes.pl -c $rsrc/tftypes.csv $rsrc/Tensors.java.tmpl > $pkg/op/Tensors.java");
# system("perl $dir/tftypes.pl -d $rsrc/tftypes.csv $rsrc/DataType.java.tmpl > $pkg/DataType.java");
