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
#define Y_SIZE 16
#define X_SIZE 16
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
	unsigned char      d[4] ; 		//方向特徴ヒストグラム?
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

#ifdef LIBRARY
extern "C" 
#endif
#ifdef USE_SVM
	char *recognize(CvSVM *, char *);
#else
	char *recognize(feature_db *, char *);
#endif

#ifdef LIBRARY
extern "C" 
#endif
#ifdef USE_SVM
	char *recognize_multi(CvSVM *, char *);
#else
	char *recognize_multi(feature_db *, char *);
#endif

char *conv_fname(char *, const char *);
int is_database(const char *);
int is_opencvxml(const char *);

#ifdef USE_SVM
#if XML_TEST
void *training(char *, char *);
#else
CvSVM *training(char *);
#endif
void leave_one_out_test(feature_db *, char *);
#else
feature_db *training(char *);
void leave_one_out_test(feature_db *);
#endif

#ifdef LIBRARY
extern "C" {
#endif
	void kocr_exclude(feature_db * db, char *lst_name);
	void kocr_distance(feature_db * db, char *lst_name);
	void kocr_average(feature_db * db, char *lst_name);

	void kocr_finish(feature_db *db);
	feature_db *kocr_init(char *filename);
#ifdef USE_SVM
	CvSVM *kocr_svm_init(char *);
	void kocr_svm_finish(CvSVM *);
	char *kocr_recognize_image(CvSVM *, char *);
#else
	char *kocr_recognize_image(feature_db *, char *);
#endif

#ifdef LIBRARY
}
#endif

#endif /* KOCR_H */
