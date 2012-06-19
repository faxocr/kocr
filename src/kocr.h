/*
 * kocr.h
 *
 * Copyright (c) 2012, Seiichi Uchida. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef KOCR_H
#define KOCR_H

#define FG 0 
#define BG 255
#define N  16
#define SETS 160
#define TESTS 1
#define MAXCONTOUR 30 /* 輪郭を表す閉曲線の最大数 */
#define MAGIC_NO 0xdeadbeaf

#define TRUE 1
#define FALSE 0

#define SMOOTHING_STEP 3
#define MASKSIZE 3

#define ABS(x) (((x) > 0) ? (x) : -(x))
#define KEY_WAIT { char dummy; scanf(&dummy); }

/* DEBUG flags */
#undef DEBUG
#undef DEBUG_FILE
#undef DEBUG_DISPLAY

typedef struct 
{
    short x;
    short y;
} Contour;

typedef struct 
{
    unsigned char      I;
    unsigned char      d[4] ; 
} DIRP;

typedef struct 
{
    double      I;
    double      d[4]; 
} DIRP_D;

typedef struct
{
    int magic;
    int nitems;
    int feature_offset;
    int class_offset;
} feature_db;

typedef struct
{
    DIRP Data[N][N];
    int	status;
} datafolder;

feature_db *training(char *);
void leave_one_out_test(feature_db *);
#ifdef LIBRARY
extern "C" 
#endif
char *recognize(feature_db *, char *);
#ifdef LIBRARY
extern "C" 
#endif
char *recognize_multi(feature_db *, char *);

char *conv_fname(char *, const char *);
int is_database(const char *);

#ifdef LIBRARY
extern "C" {
#endif
void kocr_exclude(feature_db * db, char *lst_name);
void kocr_distance(feature_db * db, char *lst_name);
void kocr_average(feature_db * db, char *lst_name);

void kocr_finish(feature_db *db);
char *kocr_recognize_image(feature_db * db, char *fname);
feature_db *kocr_init(char *filename);

#ifdef LIBRARY
}
#endif

#endif /* KOCR_H */
