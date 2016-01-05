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

#ifdef THINNING
#define ANGLES 8
#define N 12
#define Y_SIZE 12
#define X_SIZE 12
int Extract_Feature_wrapper(char *, int [N][N][ANGLES]);
int Extract_Feature(cv::Mat, int [N][N][ANGLES]);
#else
#define N  16
#define Y_SIZE 16
#define X_SIZE 16
#endif

#define FG 0 
#define BG 255
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
	unsigned char      I;			//輝度値ヒストグラム?
#ifdef THINNING
	unsigned char      d[ANGLES] ; 		//方向特徴ヒストグラム?
#else
	unsigned char      d[4] ; 		//方向特徴ヒストグラム?
#endif
} DIRP;

typedef struct 
{
	double      I;		
	double      d[4]; 
} DIRP_D;

typedef struct
{
	int magic;			//データベース識別のための変数
	int nitems;			//データベースの画像数
	int feature_offset;		//データベースの特徴量の保存場所の先頭
	int class_offset;		//データベースのクラスの保存場所の先頭
} feature_db;

typedef struct
{
	DIRP Data[N][N];
	int	status;
} datafolder;		//1つのファイルを画像のピクセルごとに保存

#ifdef _KOCR_MAIN

#ifdef __cplusplus
#define _EX_DECL
#else
#define _EX_DECL extern
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef USE_SVM
	_EX_DECL char *recognize(CvSVM *, char *);
#else
	_EX_DECL char *recognize(feature_db *, char *);
#endif

#ifdef USE_SVM
	_EX_DECL char *recognize_multi(CvSVM *, char *);
#else
	_EX_DECL char *recognize_multi(feature_db *, char *);
#endif

_EX_DECL char *conv_fname(char *, const char *);
_EX_DECL int is_database(const char *);
_EX_DECL int is_opencvxml(const char *);

#ifdef USE_SVM
	_EX_DECL CvSVM *training(char *);
	_EX_DECL void leave_one_out_test(feature_db *, char *);
#else
	_EX_DECL feature_db *training(char *);
	_EX_DECL void leave_one_out_test(feature_db *);
#endif

	_EX_DECL void kocr_exclude(feature_db * db, char *lst_name);
	_EX_DECL void kocr_distance(feature_db * db, char *lst_name);
	_EX_DECL void kocr_average(feature_db * db, char *lst_name);

#ifdef USE_SVM
	_EX_DECL CvSVM *kocr_svm_init(char *);
	_EX_DECL void kocr_svm_finish(CvSVM *);
	_EX_DECL char *kocr_recognize_image(CvSVM *, char *);
#else
	_EX_DECL feature_db *kocr_init(char *filename);
	_EX_DECL void kocr_finish(feature_db *db);
	_EX_DECL char *kocr_recognize_image(feature_db *, char *);
#endif

#ifdef __cplusplus
}
#endif

#endif /* KOCR_CPP */
#endif /* KOCR_H */
