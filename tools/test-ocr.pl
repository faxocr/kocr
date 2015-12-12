#!/usr/bin/perl

use utf8;

my $count = 0;
my $success = 0;
my $KOCR_PATH = "/home/devel/faxocr/src/kocr/src/kocr";
my $DB_PATH = "/home/devel/faxocr/src/kocr/list-num.xml";
# my $DB_PATH = "/home/devel/faxocr/src/kocr/databases/list-num.xml";

foreach (@ARGV) {
    $filepath = $_;

    if ($filepath =~ /([^\/]+)$/) {
	$filename = $1;
	if ($filename =~ /^(\d)-/) {
	    $num_label = $1;
	} else {
	    next;
	}
	$count++;
    } else {
	next;
    }

    $cmd = $KOCR_PATH . " " . $DB_PATH . " " . $filepath;
    $output = qx/$cmd/;

    if ($output =~ /Result:\s+(\d+)/) {
	$result = $1;
    } else {
	$result = "N/A";
    }
    
    print $filename . "\t" . $num_label . "\t" . $result . "\n";

    $success = ($result == $num_label) ? $success + 1 : $success;
}


$score = $success / $count;

print "Total: " . $success . " / " . $count . " = " . $score . "\n";
