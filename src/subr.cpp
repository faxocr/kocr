/*
 * subr.h
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

/*===================================================================*
 * DIRPデータ生成に必要なルーチン
 *===================================================================*/

#define _USE_MATH_DEFINES // M_PIとかを有効に

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#ifdef USE_SVM
#include <opencv/ml.h>
#endif
#include "Labeling.h"
#include "kocr.h"
#include "subr.h"

/* sigma^2=4 */
/*
 * static double Gausss[][5] = { {0.19947,0.17603,0.12099,0.06476,0.02700},
 * {0.17603,0.15535,0.10677,0.05715,0.02382},
 * {0.12099,0.10677,0.07338,0.03928,0.01637},
 * {0.06476,0.05715,0.03928,0.02102,0.00876},
 * {0.02700,0.02382,0.01637,0.00876,0.00365} };
 */

/* sigma^2=2 */
static double Gausss[][5] = {
    {0.28209, 0.21970, 0.10378, 0.02973, 0.00517},
    {0.21970, 0.17110, 0.08082, 0.02316, 0.00402},
    {0.10378, 0.08082, 0.03818, 0.01094, 0.00190},
    {0.02973, 0.02316, 0.01094, 0.00313, 0.00054},
    {0.00517, 0.00402, 0.00190, 0.00054, 0.00009}
};

/*-------------------------------------------------------------------*
 * Global Variables
 *-------------------------------------------------------------------*/
/* 分散σ^2=4のガウス分布(正規化項は無し) */
static double Gauss[][5] = {
    {1.0000, 0.8825, 0.6065, 0.3247, 0.1353},
    {0.8825, 0.7788, 0.5353, 0.2865, 0.1194},
    {0.6065, 0.5353, 0.3679, 0.1969, 0.0821},
    {0.3247, 0.2865, 0.1969, 0.1054, 0.0439},
    {0.1353, 0.1194, 0.0821, 0.0439, 0.0183}
};

typedef struct {
    unsigned char x;
    unsigned char y;
    unsigned char nu;
    double        d;
} SORT;

/*
  以下の変数はこのファイル内で、関数をまたいで利用している。
  外部のファイルへの値の受渡しは、dfのみを用いている。
 */

static unsigned char ContImg[64][64];
static unsigned char DirPat[4][N][N];
static double        Blur[4][N][N];
static short         ContLen[MAXCONTOUR];
static double        Blur_I[N][N];
static Contour*      Cont[MAXCONTOUR];
static datafolder    df;

/*===================================================================*
 * 4) 輪郭抽出
 *
 * cf. 画像解析ハンドブック,pp.579〜
 * cf. 尾崎,谷口,"画像処理--その基礎から応用まで(2nd ed)",pp.211
 *===================================================================*/
short
Contour_Detect(IplImage* Normalized)
{
    short         k, l;
    short         start_k, start_l;
    short         front_k, front_l;
    short         last_k, last_l;
    short         dk[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    short         dl[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    short         inv[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };
    unsigned char st, d, pixelval1, pixelval2;
    short         contnum = 0;
    short         contlen;

    /* Initialize */
    for (k = 0; k < 64; k++)
        for (l = 0; l < 64; l++) {
            ContImg[k][l] = BG;
        }

    /* Scanning */
    for (k = 0; k < 64; k++) {
        for (l = 0; l < 64; l++) {
            pixelval1 =
                (unsigned char)
                    Normalized->imageData[Normalized->widthStep * l + k];
            if (l != 0)
                pixelval2 =
                    (unsigned char)Normalized
                        ->imageData[Normalized->widthStep * (l - 1) + k];
            if (((pixelval1 == FG && l == 0)
                 || (pixelval1 == FG && pixelval2 == BG))
                && (ContImg[k][l] == BG)) {
                /* 端点発見 (p_0) */
                last_k = front_k = start_k = k;
                last_l = front_l = start_l = l;

                contlen = 0;
                Cont[contnum] = (Contour*)calloc(100, sizeof(Contour));
                Cont[contnum][contlen].x = start_k;
                Cont[contnum][contlen].y = start_l;
                contlen++;

                /*
                 *  d=1   0   7      → l
                 *      ＼↑／      ↓
                 *    2 ←  → 6     k
                 *      ／↓＼
                 *    3   4   5
                 */

                if (l != 0) {
                    st = 2 + 1;
                } else {
                    st = 4;
                }

                while (1) {
                    for (d = st; d < 8 + st; d++) {
                        /* 逆時計回り */
                        front_k = last_k + dk[d % 8];
                        front_l = last_l + dl[d % 8];

                        if (front_k < 0 || front_k >= 64) {
                            continue;
                        }
                        if (front_l < 0 || front_l >= 64) {
                            continue;
                        }
                        pixelval1 =
                            (unsigned char)Normalized
                                ->imageData[Normalized->widthStep * front_l
                                            + front_k];
                        if (pixelval1 == FG) {
                            ContImg[front_k][front_l] = FG;
                            last_k = front_k;
                            last_l = front_l;
                            break;
                        }
                    } /* The end of next point searching loop */
                    st = (inv[d % 8] + 1) % 8;

                    if ((last_k == start_k) && (last_l == start_l)) {
                        break; /* from while(1) */
                    } else {
                        Cont[contnum][contlen].x = front_k;
                        Cont[contnum][contlen].y = front_l;
                        contlen++;

                        if (((contlen + 1) % 100) == 0) {
                            Cont[contnum] = (Contour*)realloc(
                                Cont[contnum],
                                (((contlen + 1) / 100 + 1) * 100)
                                    * sizeof(Contour));
                        }
                    }
                } /* the end of while */

                ContLen[contnum] = contlen;
                contnum++;
                if (contnum == MAXCONTOUR) {
                    fprintf(stderr, "Too many contours (>30)are detected.\n");
                    exit(1); /* XXX */
                }
            }
        }
    }

    /* freeしていない */

    return contnum;
}

/*===================================================================*
 * 5) 微小輪郭線の方向から、方向パターンを作る
 *
 * cf. 斎藤-山田-山本,"手書漢字の方向パターン・マッチング法に
 *     よる解析", 信学論,Vol.J65-D, No.5, 1982, Section3.2.1
 *===================================================================*/
void
Contour_To_Directional_Pattern(short contnum)
{
    short  c, l, x, y;
    short  x1, y1, x2, y2;
    double theta;
    short  d, nu;

    for (nu = 0; nu < 4; nu++)
        for (x = 0; x < N; x++)
            for (y = 0; y < N; y++) {
                DirPat[nu][x][y] = 0;
            }

    for (c = 0; c < contnum; c++) {
        for (l = 0; l < ContLen[c]; l++) {
            /*
             * まず輪郭線の各点での方向θを求める。
             *
             *       (x1,y1)   l    (x2,y2)
             *     ○←●←○←◎←○←●←○
             *
             */
            x1 = Cont[c][(l + SMOOTHING_STEP) % ContLen[c]].x;
            y1 = Cont[c][(l + SMOOTHING_STEP) % ContLen[c]].y;
            x2 = Cont[c][(l - SMOOTHING_STEP + ContLen[c]) % ContLen[c]].x;
            y2 = Cont[c][(l - SMOOTHING_STEP + ContLen[c]) % ContLen[c]].y;

            theta = atan2((double)(x2 - x1), (double)(y2 - y1));
            if (ABS(theta) == M_PI) {
                theta = 0;
            }
            if (theta < 0) {
                theta += M_PI;
            }

            d = (short)floor(8.0 * theta / M_PI + 0.5) % 8;
            nu = d / 2;
            /*
             *
             *      1    2     3
             *       ＼  ｜  ／
             *         ＼｜／
             *     0 −−＋−− 0
             *         ／｜＼
             *       ／  ｜  ＼
             */
            x = (Cont[c][l].x * N) / 64;
            y = (Cont[c][l].y * N) / 64;

            if (d % 2) {
                DirPat[nu][x][y] += 2;
            } else {
                DirPat[nu][x][y] += 1;
                DirPat[(nu + 1) % 4][x][y] += 1;
            }
        }
    }
}

/*===================================================================*
 * (6) ガウス関数(σ^2=4)によるボケ変換
 *  gauss(x,y) = exp(-(x^2+y^2)/(2σ^2))
 *===================================================================*/
void
Blurring()
{
    short  x, y, i, j, nu;
    double total_weight, weight;

#ifdef DEBUG_FILE
    FILE*         fp;
    unsigned char val;
    char          fname[100];
#endif

    total_weight = 0;
    for (i = -4; i <= 4; i++)
        for (j = -4; j <= 4; j++) {
            total_weight += Gauss[ABS(i)][ABS(j)];
        }

    for (nu = 0; nu < 4; nu++) {
        for (x = 0; x < N; x++) {
            for (y = 0; y < N; y++) {
                Blur[nu][x][y] = 0.0;
                weight = 0.0;
                for (i = -4; i <= 4; i++) {
                    for (j = -4; j <= 4; j++) {
                        if ((x + i >= 0) && (x + i < N)) {
                            if ((y + j >= 0) && (y + j < N)) {
                                Blur[nu][x][y] +=
                                    (double)DirPat[nu][x + i][y + j]
                                    * Gauss[ABS(i)][ABS(j)];
                                weight += Gauss[ABS(i)][ABS(j)];
                            }
                        }
                    }
                }
                Blur[nu][x][y] *= ((double)total_weight / (double)weight);
            }
        }
    }
}

/*===================================================================*
 * 輝度値ヒストグラム平坦化
 *===================================================================*/
void
Equalize_Intensity()
{
    short i, j;
    long  k;
    short step = N * N / 256;
    SORT* sortbuf;

    sortbuf = (SORT*)calloc(N * N, sizeof(SORT));

    /* 輝度値一様化 */
    k = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sortbuf[k].x = (unsigned char)i;
            sortbuf[k].y = (unsigned char)j;
            sortbuf[k].d = Blur_I[i][j];
            k++;
        }
    }
    qsort((void*)sortbuf, N * N, sizeof(SORT), Compare);
    i = 0;
    for (k = 0; k < (N * N); k += step) {
#ifdef DEBUG
        if (i > 255) {
            fprintf(stderr, "STRANGE value\n");
        }
#endif
        for (j = 0; j < step; j++) {
            df.Data[sortbuf[k + j].x][sortbuf[k + j].y].I = (unsigned char)i;
        }
        i++;
    }

    free(sortbuf);
}

/*===================================================================*
 * 方向特徴ヒストグラム平坦化
 *===================================================================*/
void
Equalize_Directional_Pattern()
{
    short         i, j;
    unsigned char nu;
    long          k;
    short         step = N * N * 4 / 256;
    SORT*         sortbuf;

    sortbuf = (SORT*)calloc(N * N * 4, sizeof(SORT));

    /* 方向特徴ヒストグラム一様化 */
    k = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (nu = 0; nu < 4; nu++) {
                sortbuf[k].x = (unsigned char)i;
                sortbuf[k].y = (unsigned char)j;
                sortbuf[k].nu = nu;
                sortbuf[k].d = Blur[nu][i][j];
                k++;
            }
        }
    }
    qsort((char*)sortbuf, N * N * 4, sizeof(SORT), Compare);
    i = 0;
    for (k = 0; k < (N * N * 4); k += step) {
#ifdef DEBUG
        if (i > 255) {
            fprintf(stderr, "STRANGE\n");
        }
#endif
        for (j = 0; j < step; j++)
            df.Data[sortbuf[k + j].x][sortbuf[k + j].y].d[sortbuf[k + j].nu] =
                (unsigned char)i;
        i++;
    }

    free(sortbuf);
}

/*==================================================================*
 * qsort用比較関数
 *==================================================================*/
int
Compare(const void* i, const void* j)
{
    // ちょっと心配
    // printf("%g, ", ((SORT *) i)->d);

    if (((SORT*)i)->d > ((SORT*)j)->d) {
        return (1);
    }
    if (((SORT*)i)->d < ((SORT*)j)->d) {
        return (-1);
    }
    return (0);
}

/*===================================================================*
 * データ出力
 *===================================================================*/
void
Make_Intensity(IplImage* Normalized)
{
    short         i, j, x, y;
    unsigned char pixelval;

    /* 画素値に関して、64x64->16x16 */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            df.Data[i][j].I = 0;
        }

    for (x = 0; x < 64; x++) {
        i = (x * N) / 64;
        for (y = 0; y < 64; y++) {
            j = (y * N) / 64;
            pixelval = (unsigned char)
                           Normalized->imageData[Normalized->widthStep * y + x];
            if (pixelval == FG) {
                df.Data[i][j].I++;
            }
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            /* 対応する黒画素が3つ以上の時、黒画素とする */
            if (df.Data[i][j].I > 2) {
                df.Data[i][j].I = 255;
            } else {
                df.Data[i][j].I = 0;
            }
        }
    }
}

/*===================================================================*
 * (6) ガウス関数(σ^2=4)による輝度値のボケ変換
 *  Gauss(x,y) = exp(-(x^2+y^2)/(2σ^2))
 *===================================================================*/
void
Blur_Intensity()
{
    short  x, y, i, j;
    double total_weight, weight;

    total_weight = 0;
    for (i = -4; i <= 4; i++)
        for (j = -4; j <= 4; j++) {
            total_weight += Gausss[ABS(i)][ABS(j)];
        }

    for (x = 0; x < N; x++) {
        for (y = 0; y < N; y++) {
            Blur_I[x][y] = 0.0;
            weight = 0.0;
            for (i = -4; i <= 4; i++) {
                for (j = -4; j <= 4; j++) {
                    if ((x + i >= 0) && (x + i < N)) {
                        if ((y + j >= 0) && (y + j < N)) {
                            Blur_I[x][y] += (double)(df.Data[x + i][y + j].I)
                                          * Gausss[ABS(i)][ABS(j)];
                            weight += Gausss[ABS(i)][ABS(j)];
                        }
                    }
                }
            }
            Blur_I[x][y] /= (double)weight;
        }
    }
}

/*===================================================================*
 * 方向特徴表現された２パターン間の距離
 *===================================================================*/
double
DIRP_Dist(DIRP (*A)[N][N], DIRP (*B)[N][N])
{
    int    i, j, d, diff;
    double dist = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (d = 0; d < 4; d++) {
                diff = ABS(((int)A[0][i][j].d[d] - (int)B[0][i][j].d[d]));
                // printf("diff %d = ABS (%d - %d)\n", diff, (int)A[i][j].d[d],
                // (int)B[i][j].d[d]);
                dist += diff * diff;
            }
        }
    }

    return (sqrt(dist));
}

/*===================================================================*
 * 特徴抽出ルーチン
 *===================================================================*/

void
extract_feature_wrapper(char* fname, datafolder** retdf)
{
    int ret;

    IplImage* org_img =
        cvLoadImage(fname, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

    df.status = 0;
    *retdf = &df;

    if (org_img == NULL) {
        fprintf(stderr, "image file \"%s\": cannot be found.\n", fname);
        df.status = -1;
        return;
    }

    ret = extract_feature(org_img, retdf);
    if (ret) {
        df.status = -1;
        return;
    }

    return;
}

int
extract_feature(IplImage* org_img, datafolder** retdf)
{
    IplConvKernel* element;
    int            custom_shape[MASKSIZE * MASKSIZE];
    int            i, j, cc, n, m, d;
    LabelingBS     labeling;
    short          contnum;
    char*          ppp;
    int            count_pix;

    df.status = 0;
    *retdf = &df;

    // 処理後画像データの確保
    IplImage* dst_img = cvCreateImage(cvGetSize(org_img), IPL_DEPTH_8U, 1);
    IplImage* dst_img_dilate =
        cvCreateImage(cvGetSize(org_img), IPL_DEPTH_8U, 1);
    IplImage* dst_img_cc = cvCreateImage(cvGetSize(org_img), IPL_DEPTH_8U, 1);
#if LATTE_CODE
    IplImage* dst_img_erode =
        cvCreateImage(cvGetSize(org_img), IPL_DEPTH_8U, 1);
#endif

    if (org_img->nChannels > 1) {
        cvCvtColor(org_img, dst_img, CV_BGR2GRAY);
    } else {
        dst_img = org_img;
    }

#ifdef PERIFERAL
    int *    vproj, *hproj;
    CvScalar pixel;

    // 垂直射影を取る
    vproj = (int*)malloc(sizeof(int) * dst_img->width);
    for (x = 0; x < dst_img->width; x++) {
        vproj[x] = 0;
    }
    for (y = 0; y < dst_img->height; y++) {
        for (x = 0; x < dst_img->width; x++) {
            pixel = cvGet2D(dst_img, y, x);
            if ((int)pixel.val[0] == 0) {
                vproj[x]++;
            }
        }
    }
    for (x = 0; x < dst_img->width; x++) {
        printf("%d ", vproj[x]);
    }
    free(vproj);
#endif

    //=====================================================
    // 罫線除去処理
    //=====================================================
    // ラベリング前の膨張処理
    // 構造要素決定
    memset(custom_shape, 0, sizeof(int) * MASKSIZE * MASKSIZE);
    for (i = 0; i < MASKSIZE; i++) {
        for (j = 0; j < MASKSIZE; j++) {
            custom_shape[j * MASKSIZE + i] = 1;
        }
    }

    // ノイズ除去
    element = cvCreateStructuringElementEx(MASKSIZE,
                                           MASKSIZE,
                                           MASKSIZE / 2,
                                           MASKSIZE / 2,
                                           CV_SHAPE_CUSTOM,
                                           custom_shape);
#if LATTE_CODE
    cvErode(dst_img, dst_img_erode, element, 1);
    cvErode(dst_img_erode, dst_img_erode, element, 1);
    cvErode(dst_img_erode, dst_img_erode, element, 1);
    cvDilate(dst_img_erode, dst_img_dilate, element, 1);
    cvDilate(dst_img_dilate, dst_img_dilate, element, 1);
#else
    cvErode(dst_img, dst_img_dilate, element, 1);
#endif

    // ラベリング処理用の配列に膨張後の画像をコピー
    unsigned char* src = new unsigned char[dst_img->height * dst_img->width];
    for (i = 0; i < dst_img->width; i++) {
        for (j = 0; j < dst_img->height; j++) {
            // labeling.hでは「0」以外を領域とするので、色の反転を
            if (dst_img_dilate->imageData[j * dst_img->widthStep + i] == 0) {
                src[j * dst_img->width + i] = 255;
            } else {
                src[j * dst_img->width + i] = 0;
            }
        }
    }

    // ラべリング実行　(Labeling.h)
    short* cc_result = new short[dst_img->height * dst_img->width];

    // true: 領域の大きな順にソートする,しないならfalse
    // 最後の「3」:領域検出の最小領域）
    labeling.Exec(src, cc_result, dst_img->width, dst_img->height, true, 3);

    RegionInfoBS* ri;
    int           num_of_cc = labeling.GetNumOfResultRegions();
    int           size_x, size_y, top_x, top_y, bottom_x, bottom_y;
    double        aspect_ratio;

    if (!num_of_cc) {
        df.status = -1;
        return -1;
    }

    // CC画像生成
    for (i = 0; i < dst_img->width; i++) {
        for (j = 0; j < dst_img->height; j++) {
            dst_img_cc->imageData[dst_img_cc->widthStep * j + i] =
                (char)cc_result[j * dst_img_cc->width + i];
        }
    }

    // ラベリング後，CCは大きい順に並んでいる
    // そこで，「罫線らしくないCCのうちで最大サイズのCC」を文字として選択
    for (cc = 0; cc < num_of_cc; cc++) {
        ri = labeling.GetResultRegionInfo(cc);
        ri->GetSize(size_x, size_y);
        if (size_x > size_y) {
            aspect_ratio = (double)size_x / (double)size_y;
        } else {
            aspect_ratio = (double)size_y / (double)size_x;
        }

        // 罫線っぽい連結成分をスキップ
#ifdef LATTE_CODE
        if (!((aspect_ratio > 10)
              && (ri->GetNumOfPixels() > dst_img_dilate->height / 2))) {
            if (cc == 0
                && (size_y == dst_img_dilate->height
                    || size_x == dst_img_dilate->width)) {
                // printf("keisen, ");
                count_pix = 0;

                for (j = 0; j < dst_img_cc->height; j++) {
                    for (i = 0; i < dst_img_cc->width / 6; i++) {
                        if (cc_result[dst_img_cc->width * j + i] == cc + 1) {
                            cc_result[dst_img_cc->width * j + i] = 0;
                            count_pix++;
                        }
                    }
                }
                for (j = 0; j < dst_img_cc->height; j++) {
                    for (i = dst_img_cc->width - dst_img_cc->width / 6;
                         i < dst_img_cc->width;
                         i++) {
                        if (cc_result[dst_img_cc->width * j + i] == cc + 1) {
                            cc_result[dst_img_cc->width * j + i] = 0;
                            count_pix++;
                        }
                    }
                }
                for (j = 0; j < dst_img_cc->height / 6; j++) {
                    for (i = 0; i < dst_img_cc->width; i++) {
                        if (cc_result[dst_img_cc->width * j + i] == cc + 1) {
                            cc_result[dst_img_cc->width * j + i] = 0;
                            count_pix++;
                        }
                    }
                }

                if (dst_img_dilate->height > dst_img_dilate->width) {
                    if (ri->GetNumOfPixels() - count_pix
                        > dst_img_dilate->height) {
                        // printf("unified, ");
                        count_pix = 0;
                        break;
                    }
                } else {
                    if (ri->GetNumOfPixels() - count_pix
                        > dst_img_dilate->width) {
                        // printf("unified, ");
                        count_pix = 0;
                        break;
                    }
                }
                continue;
            }
            break;
        }
#else
        if (!((aspect_ratio > 10)
              && (ri->GetNumOfPixels() > dst_img_dilate->height / 2))) {
            break;
        }
#endif
    }
    ri->GetMax(top_x, top_y);
    ri->GetMin(bottom_x, bottom_y);

    for (i = 0; i < dst_img_cc->width; i++) {
        for (j = 0; j < dst_img_cc->height; j++) {
            if (cc_result[dst_img_cc->width * j + i] == (cc + 1)) {
                dst_img_cc->imageData[dst_img_cc->widthStep * j + i] = 0;
            } else {
                dst_img_cc->imageData[dst_img_cc->widthStep * j + i] =
                    (char)255; /* XXX */
            }
        }
    }

    IplImage* cropped = cvCreateImage(
        cvSize(ABS(top_x - bottom_x) + 1, ABS(top_y - bottom_y) + 1),
        dst_img_cc->depth,
        dst_img_cc->nChannels);

    // 本当は太らす前の dst_img なはずだが，dst_imgはノイズが多く，
    // 太らした方が安定しているので dst_img_ccを利用
    cvSetImageROI(dst_img_cc,
                  cvRect(bottom_x,
                         bottom_y,
                         ABS(top_x - bottom_x) + 1,
                         ABS(top_y - bottom_y) + 1));

    // 文字部外接矩形の切り出し
    cvCopy(dst_img_cc, cropped);
    cvResetImageROI(dst_img_cc);

    //==============================================================
    // アスペクト比を維持したまま，マージンをつけて正方形画像に
    int maxside = MAX((ABS(top_x - bottom_x) + 1), (ABS(top_y - bottom_y) + 1));
    IplImage* cropped_margin = cvCreateImage(cvSize(maxside, maxside),
                                             dst_img_cc->depth,
                                             dst_img_cc->nChannels);

    // 白で初期化
    for (i = 0; i < cropped_margin->width; i++) {
        for (j = 0; j < cropped_margin->height; j++) {
            cropped_margin->imageData[cropped_margin->widthStep * j + i] =
                (char)255; /* XXX */
        }
    }

    if (maxside == (ABS(top_x - bottom_x) + 1)) {
        int jstart = (cropped_margin->height - (ABS(top_y - bottom_y) + 1)) / 2;
        int jend = (cropped_margin->height + (ABS(top_y - bottom_y) + 1)) / 2;
        for (i = 0; i < cropped_margin->width; i++) {
            for (j = jstart; j < jend; j++) {
                cropped_margin->imageData[cropped_margin->widthStep * j + i] =
                    cropped->imageData[cropped->widthStep * (j - jstart) + i];
            }
        }
    } else {
        int istart = (cropped_margin->width - (ABS(top_x - bottom_x) + 1)) / 2;
        int iend = (cropped_margin->width + (ABS(top_x - bottom_x) + 1)) / 2;
        for (i = istart; i < iend; i++) {
            for (j = 0; j < cropped_margin->height; j++) {
                cropped_margin->imageData[cropped_margin->widthStep * j + i] =
                    cropped->imageData[cropped->widthStep * j + (i - istart)];
            }
        }
    }

    // 64x64の大きさに正規化
    IplImage* normalized =
        cvCreateImage(cvSize(64, 64), dst_img_cc->depth, dst_img_cc->nChannels);
    cvResize(cropped_margin, normalized, CV_INTER_NN);

    // 輪郭線抽出
    // XXX: カッコ悪いが，結果はGlobal変数で受け渡し
    contnum = Contour_Detect(normalized);
    Contour_To_Directional_Pattern(contnum);
    // 結果はCont[],contnumに入っているので、この領域をfreeする
    for (int i = 0; i < contnum; i++) {
        if (Cont[i] != NULL) {
            free(Cont[i]);
        }
    }

    // ボカシ処理
    // XXX: カッコ悪いが，結果はGlobal変数で受け渡し
    Blurring();

    // 方向ヒストグラムの正準化
    // XXX: カッコ悪いが，結果はGlobal変数で受け渡し
    Equalize_Directional_Pattern();

    // 輝度画像についても縮小＆ぼかし
    Make_Intensity(normalized);
    Blur_Intensity();

#ifdef DISPLAY_IMAGES
    IplImage* dir_image = cvCreateImage(cvSize(4 * N, N),
                                        normalized->depth,
                                        normalized->nChannels);
    IplImage* bdir_image = cvCreateImage(cvSize(4 * N, N),
                                         normalized->depth,
                                         normalized->nChannels);
    IplImage* final_image = cvCreateImage(cvSize(4 * N, N),
                                          normalized->depth,
                                          normalized->nChannels);
    IplImage* contour_image =
        cvCreateImage(cvSize(64, 64), normalized->depth, normalized->nChannels);

    /* XXXX char に255をキャストするのっておかしくね？ */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            dir_image->imageData[dir_image->widthStep * j + i] =
                (char)(DirPat[0][i][j] * 50 > 255 ? 255 : DirPat[0][i][j] * 50);

            dir_image->imageData[dir_image->widthStep * j + i + N] =
                (char)(DirPat[1][i][j] * 50 > 255 ? 255 : DirPat[1][i][j] * 50);

            dir_image->imageData[dir_image->widthStep * j + i + 2 * N] =
                (char)(DirPat[2][i][j] * 50 > 255 ? 255 : DirPat[2][i][j] * 50);

            dir_image->imageData[dir_image->widthStep * j + i + 3 * N] =
                (char)(DirPat[3][i][j] * 50 > 255 ? 255 : DirPat[3][i][j] * 50);

            bdir_image->imageData[bdir_image->widthStep * j + i] =
                (char)(Blur[0][i][j] * 10 > 255 ? 255 : Blur[0][i][j] * 10);

            bdir_image->imageData[bdir_image->widthStep * j + i + N] =
                (char)(Blur[1][i][j] * 10 > 255 ? 255 : Blur[1][i][j] * 10);

            bdir_image->imageData[bdir_image->widthStep * j + i + 2 * N] =
                (char)(Blur[2][i][j] * 10 > 255 ? 255 : Blur[2][i][j] * 10);

            bdir_image->imageData[bdir_image->widthStep * j + i + 3 * N] =
                (char)(Blur[3][i][j] * 10 > 255 ? 255 : Blur[3][i][j] * 10);

            /*
            final_image->imageData[final_image->widthStep * j + i] =
            (char)(Data[i][j].d[0] > 255 ? 255 : Data[i][j].d[0]);

            final_image->imageData[final_image->widthStep * j + i + N] =
            (char)(Data[i][j].d[1] > 255 ? 255 : Data[i][j].d[1]);

            final_image->imageData[final_image->widthStep * j + i + 2 * N] =
            (char)(Data[i][j].d[2] > 255 ? 255 : Data[i][j].d[2]);

            final_image->imageData[final_image->widthStep * j + i + 3 * N] =
            (char)(Data[i][j].d[3] > 255 ? 255 : Data[i][j].d[3]);
            */
        }
    }
    for (i = 0; i < 64; i++)
        for (j = 0; j < 64; j++)
            contour_image->imageData[contour_image->widthStep * j + i] =
                (char)(ContImg[i][j] > 0 ? 255 : 0);

    //画像の表示
    cvNamedWindow("org_img");
    cvNamedWindow("dst_img_dilate");
    cvNamedWindow("dst_img_cc");
    cvNamedWindow("cropped");
    cvShowImage("cropped", cropped);
    cvNamedWindow("contour_image");
    cvShowImage("contour_image", contour_image);
    cvNamedWindow("dir_image");
    cvShowImage("dir_image", dir_image);
    cvNamedWindow("bdir_image");
    cvShowImage("bdir_image", bdir_image);
    cvNamedWindow("final_image");
    cvShowImage("final_image", final_image);
    cvNamedWindow("cropped_margin");
    cvShowImage("cropped_margin", cropped_margin);
    cvNamedWindow("normalized");
    cvShowImage("normalized", normalized);
    cvShowImage("org_img", org_img);
    cvShowImage("dst_img_dilate", dst_img_dilate);
    cvShowImage("dst_img_cc", dst_img_cc);

    cvWaitKey(0);

    //全てのウィンドウの削除
    cvDestroyAllWindows();

    //画像データの解放
    cvReleaseImage(&dir_image);
    cvReleaseImage(&bdir_image);
    cvReleaseImage(&final_image);
    cvReleaseImage(&contour_image);
#endif

    //画像データの解放
    cvReleaseImage(&cropped);
    cvReleaseImage(&cropped_margin);
    cvReleaseImage(&normalized);
    if (dst_img == org_img) {
        // a dirty hack for mono
        cvReleaseImage(&org_img);
    } else {
        cvReleaseImage(&org_img);
        cvReleaseImage(&dst_img);
    }
    cvReleaseImage(&dst_img_dilate);
    cvReleaseImage(&dst_img_cc);
#ifdef LATTE_CODE
    cvReleaseImage(&dst_img_erode);
#endif

    return 0;
}

int
db_save(char* fname, feature_db* db)
{
    int fd, w, len;

    if ((fd = open(fname, O_WRONLY | O_CREAT, 0644)) < 0) {
        return -1;
    }
    len = sizeof(feature_db) + sizeof(DIRP[N][N]) * db->nitems
        + sizeof(char) * db->nitems;

    char* current = (char*)db;
    while ((w = write(fd, current, len)) > 0) {
        current += w;
        len -= w;
    }

    close(fd);
    if (w < 0) {
        return -1;
    }

    fprintf(stderr, "database file is generated: %s\n", fname);

    return 0;
}

feature_db*
db_load(char* fname)
{
    int         fd, len, r;
    feature_db* db;
    struct stat sb;

    fprintf(stderr, "loading database file: %s\n", fname);

    if (stat(fname, &sb) == -1) {
        fprintf(stderr, "cannot open: %s\n", fname);
        return NULL;
    }
    if ((fd = open(fname, O_RDONLY)) < 0) {
        fprintf(stderr, "cannot open: %s\n", fname);
        return NULL;
    }
    len = sb.st_size;
    db = (feature_db*)malloc(len);
    char* current = (char*)db;
    while ((r = read(fd, current, len)) > 0) {
        current += r;
        len -= r;
    }
    if (r < 0) {
        return NULL;
    }

    close(fd);

    /* XXX: この関数でmallocした領域をは呼び出し元でfreeすること */
    return db;
}
