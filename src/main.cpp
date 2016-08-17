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
#define _KOCR_MAIN

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

#ifdef USE_CNN
#include "forward_cnn.h"
#include "kocr_cnn.h"
#else
#ifdef USE_SVM
#include <opencv/ml.h>
#endif
#include "kocr.h"
#include "subr.h"
#endif

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
    char           *newname = (char *) malloc(slen);
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
usage()
{
    printf("\n");
#ifdef USE_SVM
    printf("usage (SVM mode):\n\n");
#elif defined USE_CNN
    printf("usage (CNN mode):\n\n");
#else
    printf("usage:\n\n");
#endif

#ifdef USE_CNN
    printf(" kocr\tweights-file target\t\tRecognize characters in target\n");
    printf("\ttarget\t\tUse './cnn_weights.txt' as weights-file\n");
    printf("\n");

    printf("\tweights-file: *.txt\n");
#else
    printf(" kocr\timage-list\t\tCreates a database file\n");
    printf("\tdatabase-file\t\tEvaluates a database file\n");
    printf("\tdatabase-file target\tRecognize characters in target\n");
    printf("\n");

#ifdef USE_SVM
    printf("\tdatabase-file: *.xml\n");
#else
    printf("\tdatabase-file: *.db\n");
#endif

#endif
    printf("\ttarget: *.[png|pbm|jpg]\n\n");
}

/* ============================================================
 * メイン関数
 * ============================================================ */
int
main(int argc, char *argv[])
{
    char *resultstr;

#ifdef USE_CNN

    Network *net;
    char *wf_name, *target;

    switch (argc) {
    case 3:
        // set weights-file name to argv[1]
        wf_name = argv[1];
	target = argv[2];
	break;
    case 2:
        // set weights-file name to default name
        wf_name = strdup("cnn_weights.txt");
	target = argv[1];
        break;
    default:
        usage();
        exit(0);
    }

    // Recognize characters in target
    net = kocr_cnn_init(wf_name);
    if (net != NULL && !net->load_completed){
        printf("An error occured in loading weights\n");
        exit(-1);
    }

    // Character recognition
    resultstr = kocr_recognize_image(net, target);
    if (!resultstr){
        printf("An error occured in recognizing image\n");
        exit(-1);
    }

    printf("Result: %s\n", resultstr);
    free(resultstr);

    kocr_cnn_finish(net);

#else

    // using SVM or Nearest Neighbor

    char           *db_name, *lst_name;
    feature_db     *db;

#ifdef USE_SVM
    CvSVM *svm;

    if (argc > 1 && !is_database(argv[1]) && !is_opencvxml(argv[1])) {

	// Database generation
	svm = training(argv[1]);
	if (!svm)
	    exit(-1);
	db_name = conv_fname(argv[1], ".xml");
	svm->save(db_name);
	exit(0);
    }
#else
    if (argc > 1 && !is_database(argv[1]) && !is_opencvxml(argv[1])) {

	// Database generation
	db = training(argv[1]);
	if (!db)
	    exit(-1);
	db_name = conv_fname(argv[1], ".db");
	db_save(db_name, db);
	free(db);

	exit(0);
    }
#endif /* USE_SVM */

    switch (argc) {
    case 2:

	// Leave one out testing
#ifdef USE_SVM
	db_name = conv_fname(argv[1], ".db");
	if (is_database(db_name)) {
	  db = db_load(db_name);
	  if (!db)
	    exit(-1);

	  leave_one_out_test(db, argv[1]);
	} else {
	  printf("Requires old .db file, for Leave-one-out test...\n");
	}
#else
	db = db_load(argv[1]);
	if (!db)
	    exit(-1);
	leave_one_out_test(db);
#endif
	break;

    case 3:

#ifdef USE_SVM
	// argv[1] trained data (xml)
	// argv[2] target image
	svm = kocr_svm_init(argv[1]);
	if (!svm)
	    exit(-1);

	// Character recognition
	resultstr = kocr_recognize_image(svm, argv[2]);
	if (!resultstr)
	    exit(-1);

	printf("Result: %s\n", resultstr);
	free(resultstr);

	kocr_svm_finish(svm);
#else
	if (!is_database(argv[1])) {
	  printf("Requires old .db file for character recognition.\n");
	  break;
	}

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
#endif
	break;

    case 1:
    default:
	usage();
	exit(0);
    }
#endif /* USE_CNN */
    return 0;
}
