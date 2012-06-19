#! /usr/bin/perl 
#
# usage:
#
# $ lookup.pl filelist errorlist
#
# comment:
#
# This script is to generate a simple script that moves error files.
# use as follows.
#
# $ kocr IMGDIR/FILELIST.lst
# $ kocr IMGDIR/FILELIST.db
# $ cd IMGDIR/../error
# $ ls > errorlist
# $ lookup.pl IMGDIR/FILELIST.lst errorlist > move_files.sh
# $ cd IMGDIR
# $ ./move_files

%Hash = ();

open(E, $ARGV[0]);
$n = 0;
while (<E>) {
    $word = $_;
    chop($word);
#    print "> " . $word;

    $Hash{$n} = $word;
    $n++;
}
close(E);

open(D, $ARGV[1]);
while (<D>) {
    chop($_);
    /err-\d+-[0-9a-z]-(\d+)-[0-9a-z]-(\d+).png/;
#    /err-\d-[0-9a-z]-(\d+)-[0-9a-z]-(\d+).png/;
#    $qword = $_;
#    print "> " . $qword;
#    print "> " . $_;
    print "# " . $1 ."\n";
    print "mv " .  $Hash{$1} . " trash2\n";
    print "# " . $2 ."\n";
    print "mv " .  $Hash{$2} . " trash2\n";
}
close(D);
