/*
 * main.cpp
 * 
 * Copyright (c) 2012, Takashi Okumura. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define _WITH_GETLINE

#include <stdio.h>
#include <stdlib.h>
#include <search.h> // for qsort
#include <string.h>
#include <math.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "kocr.h"
#include "subr.h"
#include "cropnums.h"
#include "Labeling.h"

#define ERR_DIR "../images/error"

#define MAXSTRLEN 1024

/* ============================================================
 * ファイル名変換
 * ============================================================ */
char *
conv_fname(char *fname, const char *ext)
{
    int		    slen = strlen(fname) + strlen(ext);
    char           *newname = (char *)malloc(slen);
    char           *p;

    strcpy(newname, fname);
    p = strrchr(newname, '.');

    if (p) {
	strcpy(p, ext);
    } else {
	p = strrchr(newname, '\0');
	strcpy(p, ext);
    }

    return newname;
}


/* ============================================================
 * 利用法説明
 * ============================================================ */
static void 
usage(char *cmdname)
{
    printf("\n");
    printf("usage:\n\n");
    printf(" %s\timage-list\t\tCreates a database file\n", cmdname);
    printf("\tdatabase-file target\tRecognize a character in target\n");
    printf("\tdatabase-file\t\tEvaluates a database file\n\n");
    printf("The system supports png and pbm.\n\n");
}

/* ============================================================
 * メイン関数
 * ============================================================ */
int 
main(int argc, char *argv[])
{
    char           *db_name, *lst_name;
    feature_db     *db;
    char           *resultstr;

    if (argc > 1 && !is_database(argv[1])) {

	// Database generation
	db = training(argv[1]);
	if (!db)
	    exit(-1);
	db_name = conv_fname(argv[1], ".db");
	db_save(db_name, db);
	free(db);

	exit(0);
    }

    switch (argc) {
    case 2:

	// Leave one out testing
	db = db_load(argv[1]);
	if (!db)
	    exit(-1);
	leave_one_out_test(db);
	break;

    case 3:

	db = kocr_init(argv[1]);
	if (!db)
	    exit(-1);

	if (!strcmp("exclude", argv[2])) {
	    // Exclude error files
	    lst_name = conv_fname(argv[1], ".lst");
	    kocr_exclude(db, lst_name);
	} else if (!strcmp("average", argv[2])) {
	    // Calcurate distance to the average
	    lst_name = conv_fname(argv[1], ".lst");
	    kocr_average(db, lst_name);
	} else if (!strcmp("distance", argv[2])) {
	    // Calcurate distance to the nearest neighbour
	    lst_name = conv_fname(argv[1], ".lst");
	    kocr_distance(db, lst_name);
	} else {
	    // Character recognition
	    resultstr = kocr_recognize_image(db, argv[2]);
	    printf("Result: %s\n", resultstr);
	    free(resultstr);
	}
	kocr_finish(db);
	break;

    case 1:
    default:
	usage(argv[0]);
	exit(0);
    }
}

