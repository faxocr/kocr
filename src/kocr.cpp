/*
 * kocr.cpp
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

#define THRES_RATIO 2 // 画像の縦横比がコレを超えると、切り出し処理へ

/*
 * static functions
 */
static char *recog_image(feature_db *, char *);
static void exclude(feature_db * db, char *lst_name);
static void distance(feature_db * db, char *lst_name);
static void average(feature_db * db, char *lst_name);

/* ============================================================*
 * トレーニング用関数
 * ============================================================*/

/* ============================================================*
 *
 * Algorithm参考文献
 *
 * 安田 道夫, 藤沢 浩道, "文字認識のための相関法の一改良",
 * 電子通信学会論文誌. D 62(3), p217-224, 1979-03
 *
 * 水上,古賀."線素方向特徴量を用いた変位抽出を行なう
 * 手書き漢字認識システム", PRMU96-188
 *
 *
 * ETL8bの1600枚(一文字あたり160パターン)の奇数シートのうち  
 * 最初のものを使って標準パターンを作成する    
 *
 * 生成手順は次の通り
 *
 *                       カテゴリ決定 
 *                             ｜
 *                  −→ for n = 1 〜 TESTS
 *                 ｜          ｜
 *                  −−− n%2 == 0 ?    
 *                             ｜
 *         −−−−−−−−−→｜  
 *       ｜                    ↓
 *       ｜    データ読み出し(2値／ビットフィールド) 
 *       ｜                    ↓
 *       ｜   データ変換 (64x63ビット) -> (64x64 byte)
 *       ｜                    ↓
 *       ｜            パターンの位置抽出
 *       ｜                    ↓
 *       ｜      パターンの大きさ／位置正規化 (64x64)
 *       ｜                    ↓
 *       ｜                  輪郭抽出
 *       ｜                    ↓
 *       ｜             方向成分へ分解 (64x64)->(16x16)
 *       ｜                    ↓
 *       ｜                ぼけ変換 (16x16)
 *       ｜                    ↓
 *       ｜         (線素方向特徴量データ完成)
 *       ｜                    ↓
 *       ｜               ファイル出力
 *       ｜                    ｜
 *         −−−−−−−− n >= TESTS ?
 *                             ↓
 *                       (標準パタン完成)
 *                     
 * ============================================================*/
feature_db *
training(char *list_file)
{
    IplConvKernel  *element;
    int		    custom_shape[MASKSIZE * MASKSIZE];
    int		    i, j, cc, n, m, d;
    int		    num_of_char = 0;
    char	    linebuf[300];
    FILE           *listfile;
    short	    contnum;
    LabelingBS	    labeling;
    char           *classData;
    char           *tgt_dir;
    DIRP           ***CharData;
    char           *Class;
    datafolder     *df;

    if ((listfile = fopen(list_file, "rt")) == (FILE *) NULL) {
	printf("image list file is not found. aborting...\n");

	// キー入力待ち
	// cvWaitKey(10);

	return NULL;
    }

    // 読み込み文字数のカウント
    while (fgets(linebuf, sizeof(linebuf), listfile) != NULL) {
	num_of_char++;
	char           *p = linebuf;
	while (isprint(*p))
	    p++;
	if (*p != '\n') {
	    printf("invalid file format...\n");
	    return NULL;
	}
    }

    if (!num_of_char) {
	printf("no entries found...\n");
	return NULL;
    }

    // 画像ディレクトリ抽出
    tgt_dir = strdup(list_file);
    if (tgt_dir) {
	char           *p;
	p = strrchr(tgt_dir, '/');
	if (p) {
	    *p = '\0';
	} else {
	    free(tgt_dir);
	    tgt_dir = strdup("./");
	}
    } else {
	return NULL;
    }

    printf("%d images found\n", num_of_char);
    printf("extracting features...\n");

    // 全文字データ格納領域確保
    Class = (char *)malloc(sizeof(char) * num_of_char);
    CharData = (DIRP ***) malloc(sizeof(DIRP **) * num_of_char);
    for (n = 0; n < num_of_char; n++) {
	CharData[n] = (DIRP **) malloc(sizeof(DIRP *) * N);
	for (i = 0; i < N; i++) {
	    CharData[n][i] = (DIRP *) malloc(sizeof(DIRP) * N);
	}
    }

    //
    // 全文字画像データの読込
    //
    n = 0;
    rewind(listfile);
    while (fgets(linebuf, sizeof(linebuf), listfile) != NULL) {
	char		charfname [400];
	char           *p;

	// 末尾の改行文字を終端文字に置き換える
	p = strchr(linebuf, '\n');
	if (p != NULL) {
	    *p = '\0';
	}

	// ファイルフォーマットの確認
	p = strrchr(linebuf, ' ');
	if (!p && linebuf[1] == '-') {
	    Class[n] = linebuf[0];
	    sprintf(charfname, "%s/%s", tgt_dir, linebuf);
	} else if (p && isprint(*(p + 1))) {
	    *p = '\0';
	    Class[n] = *(p + 1);
	    sprintf(charfname, "%s/%s", tgt_dir, linebuf);
	} else {
	    Class[n] = '0';
	    n++;
	    continue;
	}

	// XXX: グローバル変数渡しを修正し、エラーチェックを入れる
	Extract_Feature(charfname, &df);
	if (df->status) {
	    n++;
	    continue;
	}
	for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {
		for (d = 0; d < 4; d++) {
		    CharData[n][i][j].d[d] = df->Data[i][j].d[d];
		}
		CharData[n][i][j].I = df->Data[i][j].I;
	    }
	}

	n++;
    }
    printf("extraction completed...\n");
    fclose(listfile);

    //
    // 全特徴情報のパッキング
    //
    feature_db * db = (feature_db *) malloc(sizeof(feature_db) +
					    sizeof(DIRP[N][N]) * n +
					    sizeof(char) * n);
    db->magic = MAGIC_NO;
    db->nitems = num_of_char = n;
    db->feature_offset = sizeof(feature_db);
    db->class_offset = sizeof(feature_db) + sizeof(DIRP[N][N]) * n;

    /*
     * db->mem_size =  sizeof(feature_db) + sizeof(DIRP[N][N]) * n +
     * sizeof(char) * num_of_char;
     */

    DIRP(*featData)[N][N];
    featData = (DIRP(*)[N][N]) ((char *)db + sizeof(*db));
    classData = (char *)db + sizeof(*db) + sizeof(DIRP[N][N]) * n;

    for (n = 0; n < num_of_char; n++) {
	for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {
		DIRP          **A = CharData[n];
		DIRP(*B)[N][N] = &featData[n];

		B[0][i][j].I = A[i][j].I;
		B[0][i][j].d[0] = A[i][j].d[0];
		B[0][i][j].d[1] = A[i][j].d[1];
		B[0][i][j].d[2] = A[i][j].d[2];
		B[0][i][j].d[3] = A[i][j].d[3];
	    }
	}
    }

    for (n = 0; n < num_of_char; n++) {
	classData[n] = Class[n];
    }

    /*
      mallocした領域を解放する
    */
    for (n = 0; n < num_of_char; n++) {
	if (CharData[n] == NULL) continue;
	for (i = 0; i < N; i++) {
	    if (CharData[n][i] != NULL) {
		free(CharData[n][i]);
	    }
	}
	free(CharData[n]);
    }
    free(CharData);
    free(Class);

    return db;
}

/* ============================================================
 * Leave-one-out認識テスト
 * ============================================================ */
void 
leave_one_out_test(feature_db * db)
{
    double	    min_dist, dist;
    int		    min_char_data;
    int		    i, j, n, m;
    int		    correct = 0;
    int		    miss = 0;
    int		    nitems;
    char	    fn     [300];

    IplImage       *misrecog;
    DIRP(*featData)[N][N];
    char           *classData;

    if (db->magic != MAGIC_NO) {
	return;
    }
    misrecog = cvCreateImage(cvSize(2 * N, N), IPL_DEPTH_8U, 1);

    nitems = db->nitems;
    featData = (DIRP(*)[N][N]) ((char *)db + db->feature_offset);
    classData = (char *)db + db->class_offset;

    printf("starting leave-one-out testing...\n");

    for (n = 0; n < nitems; n++) {
	min_char_data = -1;
	min_dist = 1e10;
	for (m = 0; m < nitems; m++) {
	    if (m != n) {
		dist = DIRP_Dist(&featData[n],
				 &featData[m]);
		if (dist < min_dist) {
		    min_dist = dist;
		    min_char_data = m;
		}
	    }
	}

	// printf("%c   %c\n", classData[n], classData[min_char_data]);
	if (classData[n] == classData[min_char_data]) {
	    correct++;
	} else {
	    miss++;
	    for (j = 0; j < N; j++) {
		for (i = 0; i < N; i++) {
		    misrecog->imageData[misrecog->widthStep * j + i] = 
			(char)featData[n][i][j].I;
		}
		for (i = 0; i < N; i++) {
		    misrecog->imageData[misrecog->widthStep * j + i + N] = 
			(char)featData[min_char_data][i][j].I;
		}
	    }
	    sprintf(fn, "%s/err-%d-%c-%d-%c-%d.png",
		    ERR_DIR, miss,
		    classData[n], n,
		    classData[min_char_data], min_char_data);
	    try {
	      cvSaveImage(fn, misrecog);
	    } catch (cv::Exception &e) {
	      const char* err_msg = e.what();
	      //printf("%s\n", err_msg);
	    }
	}
    }

    printf("Recog-rate = %g (= %d / %d )\n",
	   (double)correct / nitems, correct, nitems);

    // KEY_WAIT;

    cvReleaseImage(&misrecog);
}

/* ============================================================
 * 文字認識用ドライバ
 * ============================================================ */
#ifdef LIBRARY
extern "C" 
#endif
char *
recognize(feature_db * db, char *fname)
{
    double	    min_dist, dist;
    int		    min_char_data;
    int		    n, nitems;
    int		    i, j, d;
    char	    result [2];

    DIRP(*featData)[N][N];
    char           *classData;
    DIRP	    targetData[N][N];

    double	    total = 0;
    datafolder      *df;

    //
    // 特徴抽出
    //
    Extract_Feature(fname, &df);
    if (df->status) {
	return 0;
    }
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    for (d = 0; d < 4; d++) {
		targetData[i][j].d[d] = df->Data[i][j].d[d];
	    }
	    targetData[i][j].I = df->Data[i][j].I;
	}
    }

    //
    // データベース利用前処理
    //
    if (db->magic != MAGIC_NO) {
	return 0;
    }
    nitems = db->nitems;
    featData = (DIRP(*)[N][N]) ((char *)db + db->feature_offset);
    classData = (char *)db + db->class_offset;
    min_char_data = -1;
    min_dist = 1e10;

    //
    // 類似画像検索ループ
    //
    for (n = 0; n < db->nitems; n++) {
	dist = DIRP_Dist(&featData[n], &targetData);
	total += dist;
	if (dist < min_dist) {
	    min_dist = dist;
	    min_char_data = n;
	}
    }

#ifndef LIBRARY
    printf("Recogized: %c (%f)\n", classData[min_char_data], min_dist);
    printf("Credibility score %2.2f\n", 1 - n * min_dist / total);
#endif

    result[0] = classData[min_char_data];
    result[1] = '\0';

    return strdup(result);
}

#ifdef LIBRARY
extern "C" 
#endif
char *
recognize_multi(feature_db * db, char *fname)
{
    double	    min_dist, dist;
    int		    min_char_data;
    int		    n, nitems;
    int		    i, j, d;
    IplImage       *src_img = NULL, *dst_img = NULL;
    CvRect	    bb;
    IplImage       *part_img, *body;
    int		    seqnum, startx, width, nextstart;
    char	    retchar, filename[BUFSIZ], *retstr;

    DIRP(*featData)[N][N];
    char           *classData;
    DIRP	    targetData[N][N];
    datafolder     *df;

    double	    total = 0;

    // 元画像を読み込む
    src_img = cvLoadImage(fname,
			   CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (src_img == NULL)
	return NULL;


    // 白黒に変換する(0,255の二値)
    dst_img = cvCreateImage(cvSize(src_img->width, src_img->height), 8, 1);
    cvThreshold(src_img, src_img, 120, 255, CV_THRESH_BINARY);

    // 文字列全体のBB
    bb = findBB(src_img);
    body = cvCreateImage(cvSize(bb.width, bb.height), src_img->depth, 1);
    cvSet(body, CV_RGB(255, 255, 255), NULL);
    cvSetImageROI(src_img, bb);
    cvCopy(src_img, body, NULL);

    startx = 0;
    seqnum = 0;

    width = body->width;

    retstr = (char *)malloc(sizeof(char) * MAXSTRLEN);
    memset(retstr, 0, sizeof(char) * MAXSTRLEN);

    // 文字を１文字ずつ切り出して認識させる
    while (startx < width) {
	part_img = cropnum(body, startx, &nextstart);
	if (part_img == NULL || part_img->width == 0)
	    break;

	//
	// 特徴抽出
	//
	if (extract_feature2(part_img, &df) == -1) {
	    free(retstr);
	    return NULL;
	}
	if (df->status) {
	    free(retstr);
	    return NULL;
	}
	for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {
		for (d = 0; d < 4; d++) {
		    targetData[i][j].d[d] =
			df->Data[i][j].d[d];
		}
		targetData[i][j].I = df->Data[i][j].I;
	    }
	}

	//
	// データベース利用前処理
	//
	if (db->magic != MAGIC_NO) {
	    free(retstr);
	    return NULL;
	}
	nitems = db->nitems;
	featData = (DIRP(*)[N][N]) ((char *)db + db->feature_offset);
	classData = (char *)db + db->class_offset;
	min_char_data = -1;
	min_dist = 1e10;

	//
	// 類似画像検索ループ
	//
	for (n = 0; n < db->nitems; n++) {
	    dist = DIRP_Dist(&featData[n], &targetData);
	    total += dist;
	    if (dist < min_dist) {
		min_dist = dist;
		min_char_data = n;
	    }
	}

	// 結果はretchar
	retchar = classData[min_char_data];
	*(retstr + seqnum) = retchar;
	*(retstr + seqnum + 1) = 0;
#ifndef LIBRARY
	// 結果を出力する
	printf("Recogized: %c (%f)\n",
	       classData[min_char_data], min_dist);
	printf("Credibility score %2.2f\n", 1 - n * min_dist / total);
#endif

	startx = nextstart;
	seqnum++;

	// XXX: extract_feature内で開放済み...?
	// cvReleaseImage(&part_img);
    }

    return retstr;
}

static char *
recog_image(feature_db * db, char *fname)
{
    IplImage       *src_img;
    char           *ret;

    // 元画像を読み込む
    src_img = cvLoadImage(fname,
			  CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

    if (src_img->width / src_img->height > THRES_RATIO)
	ret = recognize_multi(db, fname);
    else
	ret = recognize(db, fname);

    cvReleaseImage(&src_img);
    return ret;
#undef THRES_RATIO
}

/* ============================================================
 * 精度管理用関数
 * ============================================================ */
void
print_line(char *file, int n)
{
    FILE           *fp;
    char           *linebuf = NULL;
    size_t	    len;
    int		    m = 0;

    static char   **cache_lines = NULL;
    static char    *cache_file = NULL;
    static int	    nlines = 0;

    if (cache_file && n < nlines) {
	printf("%s", cache_lines[n]);
	return;
    }

    if (!(fp = fopen(file, "r")))
	return;

    nlines = 0;
    while (getline(&linebuf, &len, fp) > 0) {
	nlines++;
    }

    if (cache_lines == NULL) {
	cache_lines = (char **)malloc(sizeof(char *) * nlines);
    }
    fclose(fp);

    if (!(fp = fopen(file, "r")))
	return;
    cache_file = strdup(file);

    do {
	if (getline(&linebuf, &len, fp) < 0) {
	    /* 途中で読み込みエラーだった場合は、キャッシュを初期状態にする */
	    fclose(fp);
	    free(cache_lines);
	    cache_lines = NULL;
	    free(cache_file);
	    cache_file = NULL;
	    nlines = 0;
	    return;
	}
	cache_lines[m++] = strdup(linebuf);
    } while (m < nlines);

    printf("%s", cache_lines[n]);
    fclose(fp);
}

void 
exclude(feature_db * db, char *lst_name)
{
    double	    min_dist, dist;
    int		    min_char_data;
    int		    i, j, n, m;
    int		    correct;
    int		    miss;
    int		    nitems;
    char	    fn     [300];
    DIRP(*featData)[N][N];
    char           *classData;
    int            *deleted;

    if (db->magic != MAGIC_NO) {
	return;
    }
    nitems = db->nitems;
    featData = (DIRP(*)[N][N]) ((char *)db + db->feature_offset);
    classData = (char *)db + db->class_offset;
    deleted = (int *)calloc(nitems, sizeof(int));

    fprintf(stderr, "# Excluding failure cases...\n");
    fprintf(stderr, "%s\n", lst_name);

    do {
	correct = miss = 0;
	for (n = 0; n < nitems; n++) {
	    if (deleted[n])
		continue;
	    min_char_data = -1;
	    min_dist = 1e10;
	    for (m = 0; m < nitems; m++) {
		if (m != n && !deleted[m]) {
		    dist = DIRP_Dist(&featData[n],
				     &featData[m]);
		    if (dist < min_dist) {
			min_dist = dist;
			min_char_data = m;
		    }
		}
	    }

	    if (classData[n] == classData[min_char_data]) {
		correct++;
	    } else {
		miss++;
		print_line(lst_name, n);
		deleted[n] = -1;
	    }
	}
    } while (miss > 0);

    fprintf(stderr, "Recog-rate = %g (= %d / %d )\n",
	    (double)correct / nitems, correct, nitems);
    free(deleted);
}

void 
distance(feature_db * db, char *lst_name)
{
    double	    min_dist, dist;
    int		    min_char_data;
    int		    i, j, n, m;
    int		    correct;
    int		    miss;
    int		    nitems;
    char	    fn     [300];
    DIRP(*featData)[N][N];
    char           *classData;

    if (db->magic != MAGIC_NO) {
	return;
    }
    nitems = db->nitems;
    featData = (DIRP(*)[N][N]) ((char *)db + db->feature_offset);
    classData = (char *)db + db->class_offset;

    fprintf(stderr, "# Measuring distance to nearest stranger...\n");
    fprintf(stderr, "%s\n", lst_name);

    correct = miss = 0;
    for (n = 0; n < nitems; n++) {
	min_char_data = -1;
	min_dist = 1e10;
	for (m = 0; m < nitems; m++) {
	    if (m == n || classData[n] == classData[m])
		continue;
	    dist = DIRP_Dist(&featData[n],
			     &featData[m]);
	    if (dist < min_dist) {
		min_dist = dist;
		min_char_data = m;
	    }
	}

	printf("%4.1f\t%c\t", min_dist, classData[min_char_data]);
	print_line(lst_name, n);
    }

    fprintf(stderr, "Recog-rate = %g (= %d / %d )\n",
	    (double)correct / nitems, correct, nitems);
}

void 
average(feature_db * db, char *lst_name)
{
    double	    dist;
    int		    i, j, m, n, d, c;
    int		    nitems;
    DIRP_D	    featSum[N][N];
    DIRP	    featAve[N][N];
    DIRP(*featData)[N][N];
    char           *classData;
    int		    ncls;
    bool	    classes[256];

    if (db->magic != MAGIC_NO) {
	return;
    }
    nitems = db->nitems;
    featData = (DIRP(*)[N][N]) ((char *)db + db->feature_offset);
    classData = (char *)db + db->class_offset;

    fprintf(stderr, "# Measuring average feature...\n");
    fprintf(stderr, "%s\n", lst_name);

    for (c = 0; c < 256; c++) {
	classes[classData[c]] = false;
    }

    for (n = 0; n < nitems; n++) {
	classes[classData[n]] = true;
    }

    for (c = 0; c < 256; c++) {
	if (classes[c] == false)
	    continue;

	// feature reset
	for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {
		for (d = 0; d < 4; d++) {
		    featSum[i][j].d[d] = 0;
		}
	    }
	}

	// sum up
	ncls = 0;
	for (n = 0; n < nitems; n++) {
	    if (classData[n] != c)
		continue;
	    ncls++;
	    for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
		    for (d = 0; d < 4; d++) {
			featSum[i][j].d[d] +=
			    featData[n][i][j].d[d];
		    }
		}
	    }
	}

	// calc average
	for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {
		for (d = 0; d < 4; d++) {
		    featAve[i][j].d[d] =
			(int)(featSum[i][j].d[d] / ncls);
		}
	    }
	}

	// print dist
	for (n = 0; n < nitems; n++) {
	    if (classData[n] != c)
		continue;

	    dist = DIRP_Dist(&featAve,
			     &featData[n]);

	    printf("%4.1f\t", dist);
	    print_line(lst_name, n);
	}
    }
}

/* ============================================================
 * DBファイル判別関数
 * ============================================================ */
int 
is_database(const char *fname)
{
    feature_db     *db;
    int		    fd, magic;

    db = (feature_db *) malloc(sizeof(*db));
    if ((fd = open(fname, O_RDONLY)) < 0) {
	return 0;
    }
    if (read(fd, (void *)db, sizeof(*db)) < sizeof(*db)) {
	return 0;
    }
    magic = db->magic;

    free(db);
    close(fd);

    return magic == MAGIC_NO ? TRUE : FALSE;

}

/*
 * API for external modules
 */
#ifdef LIBRARY
extern "C" {
#endif

void
kocr_exclude(feature_db *db, char *lst_name)
{
    if (db == NULL || lst_name == NULL) return;
    exclude(db,lst_name);
}

void
kocr_distance(feature_db *db, char *lst_name)
{
    if (db == NULL || lst_name == NULL) return;
    distance(db, lst_name);
}

void
kocr_average(feature_db *db, char *lst_name)
{
    if (db == NULL || lst_name == NULL) return;
    average(db, lst_name);
}

feature_db *
kocr_init(char *filename)
{
    if (filename == NULL) return NULL;
    return db_load(filename);
}

char *
kocr_recognize_image(feature_db *db, char *fname)
{
    if (db == NULL || fname == NULL) return NULL;
    return recog_image(db,fname);
}

void
kocr_finish(feature_db *db)
{
    if (db != NULL) {
	free(db);
    }
}

#ifdef LIBRARY
}
#endif
