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
#include <opencv/ml.h>

#include "kocr.h"
#include "subr.h"
#include "cropnums.h"
#include "Labeling.h"

#define ERR_DIR "../images/error"
#define OPENCVXML "<opencv_storage>\n"
#define MAXSTRLEN 1024
#define THRES_RATIO 2 // 画像の縦横比がコレを超えると、切り出し処理へ

/*
 * static functions
 */
#ifdef USE_SVM
static char *recog_image(CvSVM *, IplImage *);
#else
static char *recog_image(feature_db *, IplImage *);
#endif

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
/*
 * データベース作成関数
 */

#ifdef USE_SVM
CvSVM *
training(char *list_file)
#else
feature_db *
training(char *list_file)
#endif
{
    IplConvKernel  *element;
    int         custom_shape[MASKSIZE * MASKSIZE];
    int         i, j, k, cc, n, m, d;
    int         num_of_char = 0; // 画像数
    char        line_buf[300];
    FILE           *listfile;
    LabelingBS      labeling; // 使ってない
    char           *class_data; // データベース上の保存場所（ポインタ）
    char           *target_dir;
    DIRP           ***char_data; // 画像ごとに、16*16のbyte領域を確保
    char           *Class; // Class[num_of_char]:画像のクラスを保存
    datafolder     *df; // 特徴量保存領域

#ifdef THINNING
    int features[N][N][ANGLES];
    // int features[N][N];
#endif

#ifdef USE_SVM
    CvSVM svm, *svm_;
    CvSVMParams param;
    CvTermCriteria criteria;
    int char_count[256], class_count;

    for (i = 0; i < 256; i++)
        char_count[i] = 0;
#endif

    if ((listfile = fopen(list_file, "rt")) == (FILE *) NULL) {
        printf("image list file is not found. aborting...\n");

        // キー入力待ち
        // cvWaitKey(10);

        return NULL;
    }

    // 読み込み文字数のカウント
    while (fgets(line_buf, sizeof(line_buf), listfile) != NULL) {
        num_of_char++;
        char           *p = line_buf;
        while (isprint(*p)) p++;
        if (*p != '\n') {
            printf("invalid file format...\n n = %d, *p = %d != %d\n",num_of_char,*p,'\n');
            return NULL;
        }
    }

    if (!num_of_char) {
        printf("no entries found...\n");
        return NULL;
    }

    // 画像ディレクトリ抽出
    // char *strdup(char c):cをmallocで領域を確保して、ポインタを返す
    target_dir = strdup(list_file);
    if (target_dir) {
        // if(p):ポインタpがnullの時不成立、if(p != NULL)
        char *p;
        // char *strrch(const char *s,int c):
        // 文字列sの先頭から文字cを探し、最初に見つかった位置をポインタで返す
        p = strrchr(target_dir, '/');
        if (p) {
            *p = '\0';
        } else {
            free(target_dir);
            target_dir = strdup("./");
        }
    } else {
        return NULL;
    }

    printf("%d images found\n", num_of_char);
    printf("extracting features...\n");

    // 全文字データ格納領域確保
    Class = (char *)malloc(sizeof(char) * num_of_char);
    // char_data[num_of_char(画像数)][Y_SIZE(16)][X_SIZE(16)](Nはピクセル数)

    char_data = (DIRP ***) malloc(sizeof(DIRP **) * num_of_char);
#ifdef THINNING
    for (n = 0; n < num_of_char; n++) {
        char_data[n] = (DIRP **) malloc(sizeof(DIRP *) * N);
        for (i = 0; i < X_SIZE; i++) {
            char_data[n][i] = (DIRP *) malloc(sizeof(DIRP) * N);
        }
    }
#else
    for (n = 0; n < num_of_char; n++) {
        char_data[n] = (DIRP **) malloc(sizeof(DIRP *) * Y_SIZE);
        for (i = 0; i < X_SIZE; i++) {
            char_data[n][i] = (DIRP *) malloc(sizeof(DIRP) * X_SIZE);
        }
    }
#endif

    //
    // 全文字画像データの読込
    //
    n = 0;
    // ファイル位置指示子を先頭に戻し、エラー指示子と終端指示子をクリアする
    rewind(listfile);
    while (fgets(line_buf, sizeof(line_buf), listfile) != NULL) {
        char        char_file_name [400];
        char           *p;

        // 末尾の改行文字を終端文字に置き換える
        p = strchr(line_buf, '\n');
        if (p != NULL) {
            *p = '\0';
        }

        // ファイルフォーマットの確認
        p = strrchr(line_buf, ' ');
        if (!p && line_buf[1] == '-') {
            // 画像のクラスを保存
            Class[n] = line_buf[0];
#ifdef USE_SVM
            char_count[line_buf[0]]++;
#endif
            // int sprintf(char *str,const char *format, ...):
            // 書式formatにしたがって、printfと同様の出力を、
            // 文字列strに格納
            sprintf(char_file_name, "%s/%s", target_dir, line_buf);
        } else if (p && isprint(*(p + 1))) {
            // int isprint(int c):cが表示文字であれば真を返す
            *p = '\0';
            Class[n] = *(p + 1);
            sprintf(char_file_name, "%s/%s", target_dir, line_buf);
        } else {
            Class[n] = '0';
            n++;
            continue;
        }

#ifdef THINNING
        Extract_Feature_wrapper(char_file_name, features);
        for (i = 0; i < N; i++) {
            // printf("%3d ", i);
            for (j = 0; j < N; j++) {
                // char_data[n][j][i].d[0] = (uchar) features[j][i];
                // printf("%3d,", char_data[n][j][i].d[0]);
                for (d = 0; d < ANGLES; d++) {
                    char_data[n][j][i].d[d] = features[j][i][d];
                    // printf("[%d]", char_data[i][k][j].d[d]);
                }
                char_data[n][j][i].I = 0;
            }
            // printf("\n");
        }
#else
        extract_feature_wrapper(char_file_name, &df);
        // subr.cpp内で宣言 (特徴量をdfに保存)
        // Extract_Feature内のMake_Intensityでdf->Data[][].I,
        // Equalize_Directional_Patternでdf->Data[][].d[]を書き換えている
        if (df->status) {
            // Extract_Featureが正しく終了したとき、status=0、失敗は-1
            n++;
            continue;
        }

        //char_dataに保存
        for (i = 0; i < Y_SIZE; i++) {
            for (j = 0; j < X_SIZE; j++) {
                for (d = 0; d < 4; d++) {
                    char_data[n][i][j].d[d] = df->Data[i][j].d[d];
                }
                char_data[n][i][j].I = df->Data[i][j].I;
            }
        }
#endif

        n++;
    }
    printf("extraction completed...\n");
    fclose(listfile);

    //
    // 全特徴情報のパッキング
    //
    feature_db * db = (feature_db *) malloc(sizeof(feature_db) +
                                            sizeof(DIRP[Y_SIZE][X_SIZE]) * n +
                                            sizeof(char) * n);
    db->magic = MAGIC_NO;
    db->nitems = num_of_char = n;
    db->feature_offset = sizeof(feature_db);
    db->class_offset = sizeof(feature_db) + sizeof(DIRP[Y_SIZE][X_SIZE]) * n;

    /*
     * db->mem_size =  sizeof(feature_db) + sizeof(DIRP[Y_SIZE][X_SIZE]) * n +
     * sizeof(char) * num_of_char;
     */

    // feature_dataの宣言
    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    // dbの特徴量の先頭アドレス
    feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + sizeof(*db));
    // dbのクラスの先頭アドレス
    class_data = (char *)db + sizeof(*db) + sizeof(DIRP[Y_SIZE][X_SIZE]) * n;

    for (n = 0; n < num_of_char; n++) {
        for (i = 0; i < Y_SIZE; i++) {
            for (j = 0; j < X_SIZE; j++) {
                DIRP          **A = char_data[n];
                DIRP(*B)[Y_SIZE][X_SIZE] = &feature_data[n];

                B[0][i][j].I = A[i][j].I;
                B[0][i][j].d[0] = A[i][j].d[0];
                B[0][i][j].d[1] = A[i][j].d[1];
                B[0][i][j].d[2] = A[i][j].d[2];
                B[0][i][j].d[3] = A[i][j].d[3];
            }
        }
    }

    for (n = 0; n < num_of_char; n++) {
        // データベース上のイメージの保存場所にクラスを保存
        class_data[n] = Class[n];
    }

#ifdef USE_SVM

    for (class_count = 0, i = 0; i < 256; i++)
        if (char_count[i] > 0) {
            class_count++;
            if (char_count[i] == 1) {
                printf("The class [%c] has only one item\n", i);
                return NULL;
            }
        }

    if (class_count < 2) {
        printf("%d classe found\n", class_count);
        printf("SVM requires at least 2 classes. exiting...\n");
        return NULL;
    } else {
        printf("%d classes found\n", class_count);
    }

    //
    // SVM学習
    //
    printf("preparing the SVM module...\n");

#ifdef THINNING
    // 必要なのはCV_8UC1だが、CvSVMのバグで、CV_32FC1にする必要がある
    CvMat *Direction = cvCreateMat(db->nitems, N * N * ANGLES, CV_32FC1);
#else
    CvMat *Direction = cvCreateMat(db->nitems, Y_SIZE * X_SIZE * 4, CV_32FC1);
    CvMat *Iluminosity = cvCreateMat(db->nitems, Y_SIZE * X_SIZE, CV_32FC1);
#endif
    CvMat *Classlabel = cvCreateMat(db->nitems, 1, CV_32FC1);

    for (i = 0; i < db->nitems; ++i) {
        cvmSet(Classlabel, i, 0, (float) class_data[i]);
#ifdef THINNING
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                for (int kk = 0; kk < ANGLES; ++kk) {
                    cvmSet(Direction,
                           i, j * N * ANGLES + k * ANGLES + kk,
                           char_data[i][k][j].d[kk]);
                }
#else
        for (j = 0; j < Y_SIZE; ++j) {
            for (k = 0; k < X_SIZE; ++k) {
                cvmSet(Iluminosity,
                       i,
                       j * Y_SIZE + k,
                       (float) (feature_data[i][j][k].I));
                for (int kk = 0; kk < 4; ++kk) {
                    cvmSet(Direction,
                           i,
                           j * Y_SIZE * 4 + k * 4 + kk,
                           (float) feature_data[i][j][k].d[kk]);
                }
#endif
            }
        }
    }

    criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

    param.svm_type = CvSVM::C_SVC;
    param.kernel_type = CvSVM::LINEAR;
    // param.kernel_type = CvSVM::RBF;
    param.term_crit = criteria;

    printf("training the SVM...\n");

    try {
        svm_ = new CvSVM();
        svm_->train_auto(Direction, Classlabel, NULL, NULL, param,
                         50,
                         svm.get_default_grid(CvSVM::C),
                         svm.get_default_grid(CvSVM::GAMMA),
                         svm.get_default_grid(CvSVM::P),
                         svm.get_default_grid(CvSVM::NU),
                         svm.get_default_grid(CvSVM::COEF),
                         svm.get_default_grid(CvSVM::DEGREE));
    } catch (cv::Exception &e) {
        const char* err_msg = e.what();
        printf("%s\n", err_msg);
    }

    free(db);
#endif /* USE_SVM */

    /*
     * mallocした領域を解放する
     */
    for (n = 0; n < num_of_char; n++) {
        if (char_data[n] == NULL) continue;
        for (i = 0; i < X_SIZE; i++) {
            if (char_data[n][i] != NULL) {
                free(char_data[n][i]);
            }
        }
        free(char_data[n]);
    }
    free(char_data);
    free(Class);

#ifdef USE_SVM
    return svm_;
#else
    return db;
#endif
}

/* ============================================================
 * Leave-one-out認識テスト
 * ============================================================ */
#ifdef USE_SVM
void
leave_one_out_test(feature_db * db, char *svm_data)
#else
void
leave_one_out_test(feature_db * db)
#endif
{
    double      min_dist, dist; //最近傍法用の距離計算
    int         min_char_data;
    int         i, j, n, m;
    int         correct = 0;
    int         miss = 0;
    int         nitems;
    char        file_num     [300];

    IplImage       *miss_recog;
    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    char           *class_data;

    // データベースがこのプログラムで作られたものではないとき終了
    if (db->magic != MAGIC_NO) {
        return;
    }
    miss_recog = cvCreateImage(cvSize(2 * X_SIZE, Y_SIZE), IPL_DEPTH_8U, 1);

    nitems = db->nitems;
    feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + db->feature_offset);
    class_data = (char *)db + db->class_offset;

    //printf("starting leave-one-out testing...\n");

#ifdef USE_SVM
    cv::Mat Direction;
    Direction.create(db->nitems-1,Y_SIZE*X_SIZE*4,CV_32FC1);
    cv::Mat Classlabel;
    Classlabel.create(db->nitems-1,1,CV_32FC1);
    CvMat *Inputdata = cvCreateMat(1,Y_SIZE*X_SIZE*4,CV_32FC1);

    CvSVM svm, svm_;
    CvSVMParams param;
    char response;
    CvTermCriteria criteria;

    svm_.load(svm_data);
    param = svm_.get_params();

    int pass_n;

    for (n = 0; n < nitems; n++) {
        pass_n = 0;
        for (i = 0; i < nitems; ++i) {
            if (i != n)
                Classlabel.at<float>(i - pass_n, 0) =
                    (float) class_data[(int)(i - pass_n)];
            for (j = 0; j < Y_SIZE; ++j) {
                for (int k = 0; k < X_SIZE; ++k) {
                    for (int kk = 0; kk < 4; ++kk) {
                        if (i != n)
                            Direction.at<float>(i - pass_n, j * X_SIZE * 4 + k * 4 + kk) =
                                (float) feature_data[(int)(i - pass_n)][j][k].d[kk];
                        else
                            cvmSet(Inputdata, 0, j * X_SIZE * 4 + k * 4 + kk,
                                   (float) feature_data[i][j][k].d[kk]);
                    }
                }
            }
            if (i == n)
                pass_n = 1;
        }

        if (1) {
            // True LOOT (SLOW)
            svm.train(Direction, Classlabel, cv::Mat(), cv::Mat(), param);
            response = (char) svm.predict(Inputdata);
        } else {
            // Trained data
            response = (char) svm_.predict(Inputdata);
        }

        if (response == class_data[n]) {
            correct++;
        } else {

            miss++;
            for (j = 0; j < Y_SIZE; j++) {
                for (i = 0; i < X_SIZE; i++) {
                    miss_recog->imageData[miss_recog->widthStep * j + i] =
                        (char) feature_data[n][i][j].I;
                }
                /*
                for (i = 0; i < X_SIZE; i++) {
                  miss_recog->imageData[miss_recog->widthStep * j + i + X_SIZE] =
                    (char) feature_data[min_char_data][i][j].I;
                }
                */
            }
            sprintf(file_num, "%s/err-%d-%c-%c.png",
                    ERR_DIR, n,
                    class_data[n],
                    response);
            printf("miss image : %s\n", file_num);
            printf("class = %c, response = %c\n", class_data[n], response);
            try {
                cvSaveImage(file_num, miss_recog);
            } catch (cv::Exception &e) {
                const char* err_msg = e.what();
                // printf("%s\n", err_msg);
            }
        }

        // printf("correct = %d, miss = %d\n",correct,miss);
        printf("Recog-rate = %0.3f (= %d / %d )\n",
               (double) correct / (n + 1), correct, n + 1);
    }

#else /* USE_SVM */
    for (n = 0; n < nitems; n++) {
        // float
        min_char_data = -1;
        min_dist = 1e10;
        // 最近傍検索ループ
        for (m = 0; m < nitems; m++) {
            if (m != n) {
                dist = DIRP_Dist(&feature_data[n],
                                 &feature_data[m]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_char_data = m;
                }
            }
        }

        // printf("%c   %c\n", class_data[n], class_data[min_char_data]);
        if (class_data[(int)n] == class_data[(int)min_char_data]) {
            // 認識が正しいとき正解数増加
            correct++;
        } else {
            // 誤認識したとき
            miss++;
            for (j = 0; j < X_SIZE; j++) {
                for (i = 0; i < Y_SIZE; i++) {
                    miss_recog->imageData[miss_recog->widthStep * j + i] =
                        (char)feature_data[n][i][j].I;
                }
                for (i = 0; i < X_SIZE; i++) {
                    miss_recog->imageData[miss_recog->widthStep * j + i + X_SIZE] =
                        (char)feature_data[(int)min_char_data][i][j].I;
                }
            }
            sprintf(file_num, "%s/err-%d-%c-%d-%c-%d.png",
                    ERR_DIR, miss,
                    class_data[n], n,
                    class_data[(int)min_char_data], (int)min_char_data);
            printf("miss image : %s\n",file_num);
            try {
                cvSaveImage(file_num, miss_recog);
            } catch (cv::Exception &e) {
                const char* err_msg = e.what();
                // printf("%s\n", err_msg);
            }
        }
    }
#endif

    printf("Recog-rate = %g (= %d / %d )\n",
           (double)correct / nitems, correct, nitems);

    // KEY_WAIT;

    cvReleaseImage(&miss_recog);
}

/* ============================================================
 * 文字認識用ドライバ
 * ============================================================ */
char *
#ifdef USE_SVM
recognize(CvSVM *db, IplImage *src_img)
#else
recognize(feature_db *db,  IplImage *src_img)
#endif
{
    double      min_dist, dist;
    int         min_char_data;
    int         n, nitems;
    int         i, j, d;
    char        result [2];

    char           *class_data;
    double      total = 0;
    DIRP        target_data[Y_SIZE][X_SIZE];

#ifdef THINNING
    int features[N][N][ANGLES];

    if (Extract_Feature(cv::cvarrToMat(src_img, true), features))
        return 0;

    CvMat *feature_mat = cvCreateMat(1, N * N * ANGLES, CV_32FC1);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (int k = 0; k < ANGLES; k++) {
                cvmSet(feature_mat, 0, i * N * ANGLES + j * ANGLES + k,
                       features[j][i][k]);
            }
        }
    }

#else
    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    datafolder      *df;

    //
    // 特徴抽出
    //
    extract_feature(src_img, &df);
    if (df->status) {
        // 特徴抽出失敗で真
        return 0;
    }

    for (i = 0; i < Y_SIZE; i++) {
        for (j = 0; j < X_SIZE; j++) {
            for (d = 0; d < 4; d++) {
                target_data[i][j].d[d] = df->Data[i][j].d[d];
            }
            target_data[i][j].I = df->Data[i][j].I;
        }
    }
#endif

#ifdef USE_SVM

#ifdef THINNING
    char response = (char) db->predict(feature_mat);
#else
    /* data packing and recognization */
    int kk;
    char response;

    CvMat *Inputdata = cvCreateMat(1, Y_SIZE * X_SIZE * 4, CV_32FC1);
    for (i = 0; i < Y_SIZE; ++i) {
        for (j = 0; j < X_SIZE; ++j) {
            for (kk = 0; kk < 4; ++kk) {
                cvmSet(Inputdata, 0, i * X_SIZE * 4 + j * 4 + kk,
                       (float) target_data[i][j].d[kk]);
            }
        }
    }
    response = (char) db->predict(Inputdata);
#endif

#ifndef LIBRARY
    printf("Recogized: %c\n", response);
#endif

    result[0] = response;
    result[1] = '\0';

#else /* USE_SVM */

    //
    // データベース利用前処理
    //
    if (db->magic != MAGIC_NO) {
        return 0;
    }
    nitems = db->nitems;
    feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + db->feature_offset);
    class_data = (char *)db + db->class_offset;
    min_char_data = -1;
    min_dist = 1e10;

    //最短距離法
    //
    // 類似画像検索ループ
    //
    for (n = 0; n < db->nitems; n++) {
        dist = DIRP_Dist(&feature_data[n], &target_data);
        total += dist;
        if (dist < min_dist) {
            min_dist = dist;
            min_char_data = n;
        }
    }

#ifndef LIBRARY
    printf("Recogized: %c (%f)\n", class_data[min_char_data], min_dist);
    printf("Credibility score %2.2f\n", 1 - n * min_dist / total);
#endif

    result[0] = class_data[min_char_data];
    result[1] = '\0';
#endif

    return strdup(result);
}

char *
#ifdef USE_SVM
recognize_multi(CvSVM *db, IplImage *src_img)
#else
recognize_multi(feature_db *db, IplImage *src_img)
#endif
{
    double      min_dist, dist;
    int         min_char_data;
    int         n, nitems;
    int         i, j, d;
    IplImage       *dst_img = NULL;
    CvRect      bb;
    IplImage       *part_img, *body;
    int         seq_num, start_x, width, next_start;
    char        result_char, filename[BUFSIZ], *result_str;

    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    char           *class_data;
    DIRP        target_data[Y_SIZE][X_SIZE];
    datafolder     *df;

    double      total = 0;

    if (src_img == NULL)
        return NULL;

    // 白黒に変換する(0,255の二値)
    dst_img = cvCreateImage(cvSize(src_img->width, src_img->height), 8, 1);
    cvThreshold(src_img, src_img, 120, 255, CV_THRESH_BINARY);

    // 文字列全体のBB
    bb = findBB(src_img); // crop_nums.cpp内で宣言
    body = cvCreateImage(cvSize(bb.width, bb.height), src_img->depth, 1);
    // void cvSet( CvArr* arr, CvScalar value, const CvArr* mask=NULL ):
    // 配列arrにvalueを入れる
    cvSet(body, CV_RGB(255, 255, 255), NULL);
    // void cvSetImageROI(IplImage *img, CvRect rect):
    // imgの矩形範囲に着目(ROI)
    cvSetImageROI(src_img, bb);
    //void cvCopy( const CvArr* src, CvArr* dst, const CvArr* mask=NULL ):
    // 配列dstに配列srcの内容をコピー
    cvCopy(src_img, body, NULL);

    start_x = 0;
    seq_num = 0;

    width = body->width;

    result_str = (char *)malloc(sizeof(char) * MAXSTRLEN);
    // void *memset(void *buf, int ch, size_t n):
    // buf の先頭から n バイト分 ch をセット
    memset(result_str, 0, sizeof(char) * MAXSTRLEN);

    // 文字を１文字ずつ切り出して認識させる
    while (start_x < width) {
        part_img = cropnum(body, start_x, &next_start);
        if (part_img == NULL || part_img->width == 0)
            break;

        //
        // 特徴抽出
        //

#ifdef THINNING
        int features[N][N][ANGLES];

        if (Extract_Feature(cv::cvarrToMat(part_img, true), features))
            return 0;

        CvMat *feature_mat = cvCreateMat(1, N * N * ANGLES, CV_32FC1);
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (int k = 0; k < ANGLES; k++) {
                    cvmSet(feature_mat, 0, i * N * ANGLES + j * ANGLES + k,
                           features[j][i][k]);
                }
            }
        }
#else
        if (extract_feature(part_img, &df) == -1) {
            free(result_str);
            return NULL;
        }
        if (df->status) {
            free(result_str);
            return NULL;
        }
        for (i = 0; i < Y_SIZE; i++) {
            for (j = 0; j < X_SIZE; j++) {
                for (d = 0; d < 4; d++) {
                    target_data[i][j].d[d] =
                        df->Data[i][j].d[d];
                }
                target_data[i][j].I = df->Data[i][j].I;
            }
        }
#endif

#ifdef USE_SVM

#ifdef THINNING
        result_char = (char) db->predict(feature_mat);
#else
        /* data packing and recognization */
        int kk;
        char response;

        CvMat *Inputdata = cvCreateMat(1, Y_SIZE * X_SIZE * 4, CV_32FC1);
        for (i = 0; i < Y_SIZE; ++i) {
            for (j = 0; j < X_SIZE; ++j) {
                for (kk = 0; kk < 4; ++kk) {
                    cvmSet(Inputdata, 0, i * X_SIZE * 4 + j * 4 + kk,
                           (float) target_data[i][j].d[kk]);
                }
            }
        }
        result_char = (char) db->predict(Inputdata);
#endif

        *(result_str + seq_num) = result_char;
        *(result_str + seq_num + 1) = 0;

#ifndef LIBRARY
        printf("Recogized: %c\n", result_char);
#endif

#else
        //
        // データベース利用前処理
        //
        if (db->magic != MAGIC_NO) {
            free(result_str);
            return NULL;
        }
        nitems = db->nitems;
        feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + db->feature_offset);
        class_data = (char *)db + db->class_offset;
        min_char_data = -1;
        min_dist = 1e10;

        //
        // 類似画像検索ループ
        //
        for (n = 0; n < db->nitems; n++) {
            dist = DIRP_Dist(&feature_data[n], &target_data);
            total += dist;
            if (dist < min_dist) {
                min_dist = dist;
                min_char_data = n;
            }
        }
        // 結果はretchar
        result_char = class_data[min_char_data];
        *(result_str + seq_num) = result_char;
        *(result_str + seq_num + 1) = 0;

#ifndef LIBRARY
        // 結果を出力する
        printf("Recogized: %c (%f)\n",
               class_data[min_char_data], min_dist);
        printf("Credibility score %2.2f\n", 1 - n * min_dist / total);
#endif

#endif /* USE_SVM */

        start_x = next_start;
        seq_num++;

        // XXX: extract_feature内で開放済み...?
        // cvReleaseImage(&part_img);
    }

    return result_str;
}

#ifdef USE_SVM
static char *
recog_image(CvSVM *db, IplImage *src_img)
#else
static char *
recog_image(feature_db *db, IplImage *src_img)
#endif
{
    char           *result;

    if (src_img->width / src_img->height > THRES_RATIO)
        result = recognize_multi(db, src_img);
    else
        result = recognize(db, src_img);

    return result;
#undef THRES_RATIO
}

/* ============================================================
 * 精度管理用関数
 * ============================================================ */
void
print_line(char *file, int n)
{
    FILE           *fp;
    char           *line_buf = NULL;
    size_t      len;
    int         m = 0;

    static char   **cache_lines = NULL;
    static char    *cache_file = NULL;
    static int      n_lines = 0; //ファイルの行数

    if (cache_file && n < n_lines) {
        printf("%s", cache_lines[n]);
        return;
    }

    if (!(fp = fopen(file, "r")))
        return;

    n_lines = 0;
    while (getline(&line_buf, &len, fp) > 0) {
        n_lines++;
    }

    if (cache_lines == NULL) {
        cache_lines = (char **)malloc(sizeof(char *) * n_lines);
    }
    fclose(fp);

    if (!(fp = fopen(file, "r")))
        return;
    cache_file = strdup(file);

    do {
        if (getline(&line_buf, &len, fp) < 0) {
            // 途中で読み込みエラーだった場合は、キャッシュを初期状態にする
            fclose(fp);
            free(cache_lines);
            cache_lines = NULL;
            free(cache_file);
            cache_file = NULL;
            n_lines = 0;
            return;
        }
        // cache_lines[m]に文字列をメモリ領域を確保しつつ書き込み
        cache_lines[m++] = strdup(line_buf);
    } while (m < n_lines);

    printf("%s", cache_lines[n]);
    fclose(fp);
}

void
exclude(feature_db * db, char *lst_name)
{
    double      min_dist, dist;
    int         min_char_data;
    int         i, j, n, m;
    int         correct;
    int         miss;
    int         nitems; // 画像数
    char        file_num     [300];
    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    char           *class_data;
    int            *deleted;

    // データベースファイル識別
    if (db->magic != MAGIC_NO) {
        return;
    }

    nitems = db->nitems;
    feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + db->feature_offset);
    class_data = (char *)db + db->class_offset;
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
            // 最近傍探索
            for (m = 0; m < nitems; m++) {
                if (m != n && !deleted[m]) {
                    dist = DIRP_Dist(&feature_data[n],
                                     &feature_data[m]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_char_data = m;
                    }
                }
            }

            if (class_data[n] == class_data[min_char_data]) {
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
    double      min_dist, dist;
    int         min_char_data;
    int         i, j, n, m;
    int         correct;
    int         miss;
    int         nitems;
    char        file_num     [300];
    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    char           *class_data;

    if (db->magic != MAGIC_NO) {
        return;
    }
    nitems = db->nitems;
    feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + db->feature_offset);
    class_data = (char *)db + db->class_offset;

    fprintf(stderr, "# Measuring distance to nearest stranger...\n");
    fprintf(stderr, "%s\n", lst_name);

    correct = miss = 0;
    // 最近傍探索
    for (n = 0; n < nitems; n++) {
        min_char_data = -1;
        min_dist = 1e10;
        for (m = 0; m < nitems; m++) {
            if (m == n || class_data[n] == class_data[m])
                continue;
            dist = DIRP_Dist(&feature_data[n],
                             &feature_data[m]);
            if (dist < min_dist) {
                min_dist = dist;
                min_char_data = m;
            }
        }
        // 最小距離の表示
        printf("%4.1f\t%c\t", min_dist, class_data[min_char_data]);
        print_line(lst_name, n);
    }

    fprintf(stderr, "Recog-rate = %g (= %d / %d )\n",
            (double)correct / nitems, correct, nitems);
}

void
average(feature_db * db, char *lst_name)
{
    double      dist;
    int         i, j, m, n, d, c;
    int         nitems;
    DIRP_D      feature_sum[Y_SIZE][X_SIZE];//クラスごとの特徴量の総和
    DIRP        feature_ave[Y_SIZE][X_SIZE];//クラスごとの特徴量の平均
    DIRP(*feature_data)[Y_SIZE][X_SIZE];
    char           *class_data;
    int         n_class;
    bool        classes[256];

    if (db->magic != MAGIC_NO) {
        return;
    }
    nitems = db->nitems;
    feature_data = (DIRP(*)[Y_SIZE][X_SIZE]) ((char *)db + db->feature_offset);
    class_data = (char *)db + db->class_offset;

    fprintf(stderr, "# Measuring average feature...\n");
    fprintf(stderr, "%s\n", lst_name);

    for (c = 0; c < 256; c++) {
        classes[class_data[c]] = false;//classes[c]の間違い？
    }

    for (n = 0; n < nitems; n++) {
        classes[class_data[n]] = true;
    }

    for (c = 0; c < 256; c++) {
        if (classes[c] == false)//クラスが存在しない場合
            continue;

        // feature reset
        for (i = 0; i < Y_SIZE; i++) {
            for (j = 0; j < X_SIZE; j++) {
                for (d = 0; d < 4; d++) {
                    feature_sum[i][j].d[d] = 0;
                }
            }
        }

        // sum up
        //クラスごとの特徴量の総和
        n_class = 0;
        for (n = 0; n < nitems; n++) {
            if (class_data[n] != c)
                continue;
            n_class++;
            for (i = 0; i < Y_SIZE; i++) {
                for (j = 0; j < X_SIZE; j++) {
                    for (d = 0; d < 4; d++) {
                        feature_sum[i][j].d[d] +=
                            feature_data[n][i][j].d[d];
                    }
                }
            }
        }

        // calc average
        //クラスごとの特徴量の平均
        for (i = 0; i < Y_SIZE; i++) {
            for (j = 0; j < X_SIZE; j++) {
                for (d = 0; d < 4; d++) {
                    feature_ave[i][j].d[d] =
                        (int)(feature_sum[i][j].d[d] / n_class);
                }
            }
        }

        // print dist
        for (n = 0; n < nitems; n++) {
            if (class_data[n] != c)
                continue;

            dist = DIRP_Dist(&feature_ave,
                             &feature_data[n]);

            printf("%4.1f\t", dist);
            print_line(lst_name, n);
        }
    }
}

/* ============================================================
 * DBファイル判別関数
 * ============================================================ */
int
is_database(const char *file_name)
{
    feature_db     *db;
    int         fd, magic;

    db = (feature_db *) malloc(sizeof(*db));
    if ((fd = open(file_name, O_RDONLY)) < 0) {
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

int
is_opencvxml(const char *file_name)
{
    FILE *fp;
    size_t len;
    ssize_t read;
    char *line = NULL;

    if ((fp = fopen(file_name, "r")) == NULL) {
        return FALSE;
    }

    if ((read = getline(&line, &len, fp)) <= 0) {
        return FALSE;
    }

    if ((read = getline(&line, &len, fp)) <= 0) {
        return FALSE;
    }

    if (strcmp(line, OPENCVXML) != 0)
        return FALSE;

    fclose(fp);

    return TRUE;
}

/*
 * API for external modules
 */

void
kocr_exclude(feature_db *db, char *lst_name)
{
    if (db == NULL || lst_name == NULL)
        return;
    exclude(db,lst_name);
}

void
kocr_distance(feature_db *db, char *lst_name)
{
    if (db == NULL || lst_name == NULL)
        return;
    distance(db, lst_name);
}

void
kocr_average(feature_db *db, char *lst_name)
{
    if (db == NULL || lst_name == NULL)
        return;
    average(db, lst_name);
}

#ifdef USE_SVM
CvSVM *
kocr_svm_init(char *filename)
{
    CvSVM *svm;

    if (filename == NULL)
        return (CvSVM *) NULL;

    svm = new CvSVM();
    svm->load(filename);

    return svm;
}
#else
feature_db *
kocr_init(char *filename)
{
    if (filename == NULL) return NULL;
    return db_load(filename);
}
#endif

#ifdef USE_SVM
char *
kocr_recognize_Image(CvSVM *db, IplImage *src_img)
#else
char *
kocr_recognize_Image(feature_db *db, IplImage *src_img)
#endif
{
    if (db == NULL || src_img == NULL)
        return NULL;

    return recog_image(db, src_img);
}

#ifdef USE_SVM
char *
kocr_recognize_image(CvSVM *db, char *file_name)
#else
char *
kocr_recognize_image(feature_db *db, char *file_name)
#endif
{
    IplImage       *src_img;
    char *c;

    if (db == NULL || file_name == NULL)
        return NULL;

    // 元画像を読み込む
#if 0
    src_img = cvLoadImage(file_name,
                          CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
#else
    src_img = cvLoadImage(file_name,
                          CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_GRAYSCALE);
#endif

    // OpenCVはGIFを扱えないらしい
    if (!src_img) {
        char *p = file_name;
        for (; *p; ++p) *p = tolower(*p);
        if (strstr(file_name, ".gif")) {
            printf("This program doesn't support GIF images.\n");
        }
        return NULL;
    }

    c = recog_image(db, src_img);
    cvReleaseImage(&src_img);

    return c;
}

#ifdef USE_SVM
void
kocr_svm_finish(CvSVM *svm)
{
    if (svm != NULL) {
        delete svm;
    }
}
#else
void
kocr_finish(feature_db *db)
{
    if (db != NULL) {
        free(db);
    }
}
#endif
