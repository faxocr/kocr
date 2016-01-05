#!/usr/bin/perl

use utf8;

foreach (@ARGV) {
    $filename = $_;

    if ($filename =~ /(\S+)\.(\S+)$/) {
	$lname = $1 . "-left." . $2;
	$rname = $1 . "-right." . $2;
	$gname = $1 . "-large." . $2;
	# print $newfname . "\n";
	# system("echo ". $filename . " " . $newfname);

	system("convert -geometry 171% ". $filename . " " . $gname);
    }
}
