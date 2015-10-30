#!/usr/bin/perl

use utf8;

my $count = 0;
my $success = 0;
my $KOCR_PATH = "/home/faxocr/src/kocr/src/kocr";
my $DB_PATH = "/home/faxocr/src/kocr/list-num.xml";

foreach (@ARGV) {
    $filepath = $_;
    $filepath =~ /([^\/]+)$/;

    $filename = $1;
    $filename =~ /^(\d)-/;

    $num_label = $1;
    $count++;

    $cmd = $KOCR_PATH . " " . $DB_PATH . " " . $filepath;
    $output = qx/$cmd/;

    $output =~ /Result:\s+(\d+)/;
    $result = $1;
    
    print $filename . "\t" . $num_label . "\t" . $result . "\n";

    $success = ($result == $num_label) ? $success + 1 : $success;
}


$score = $success / $count;

print "Total: " . $success . " / " . $count . " = " . $score . "\n";
