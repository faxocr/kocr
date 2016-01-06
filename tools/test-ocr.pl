#!/usr/bin/perl

use utf8;

my $count = 0;
my $success = 0;
my $KOCR_PATH = "./src/kocr";
my $DB_PATH = "./src/list-num.xml";

my %error_hash;
my %n_sample;
my %n_success;

if (!-e $DB_PATH) {
	print "no db file found: " . $DB_PATH . "\n";
	die;
}

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

    if ($result != $num_label) {
	if (defined $error_hash{$num_label}{$result})	{
	    $error_hash{$num_label}{$result} += 1;
	} else {
	    $error_hash{$num_label}{$result} = 1;
	}
    }
    $n_sample{$num_label}++;
    $n_success{$num_label} = $n_success{$num_label} + 
	(($result == $num_label) ? 1 : 0);
    $success = ($result == $num_label) ? $success + 1 : $success;
}


$score = $success / $count;

print "Total: " . $success . " / " . $count . " = " . $score . "\n";

foreach my $list (sort keys %error_hash) {
    print "[" . $list ."] -> [" . $list . "]\t";
    print $n_success{$list} . " / " . $n_sample{$list};
    printf "\t(%2.2f)", $n_success{$list} / $n_sample{$list};
    print "\n";
    foreach ( keys %{$error_hash {$list}}) {
	print "    -> ";
	print "[" . $_ ."]\t";
	print $error_hash{$list}->{$_} . " / " . $n_sample{$list};
	printf "\t(%2.2f)", $error_hash{$list}->{$_} / $n_sample{$list};
	print "\n";
    }
}
