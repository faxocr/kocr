/*
 * thinning.cpp
 *
 * Copyright (c) 2015, Takashi Okumura. All rights reserved.
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

#ifdef THINNING

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <stdio.h>

#include "Labeling.h"
#include "kocr.h"

#ifndef KOCR_H
// defined in kocr.h
#define ANGLES 8
#define N 12
#endif

/*
 * Extraction parameters
 */
#define SIZE_NORMALIZED 48
#define M (SIZE_NORMALIZED / N)
#define DILATION_SIZE 1
#define ELIMINATION_THRESHOLD 1.35
#define FILENAME_SIZE 1024
#define ERR_RTN -1
#define ABS(x) (((x) > 0) ? (x) : -(x))

/* for debugging */
char filename[FILENAME_SIZE];
#ifdef THINNING_MAIN
static int checkPreprocessedImage(const cv::Mat &img_src, const cv::Mat &img_bw, const cv::Mat &img_dilated, const cv::Mat &img_eroded, const cv::Mat &img_extracted, const cv::Mat &img_normalized);
static int showPreprocessedImageFlag = 0;
static char *writePreprocessedImageFileName = NULL;
#endif /* THINNING_MAIN */

// 膨張・収縮処理用マスク
static cv::Mat element = (cv::Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);

/* 分散σ^2=4のガウス分布(正規化項は無し) */
static double   Gauss[][5] = {
    {1.0000, 0.8825, 0.6065, 0.3247, 0.1353},
    {0.8825, 0.7788, 0.5353, 0.2865, 0.1194},
    {0.6065, 0.5353, 0.3679, 0.1969, 0.0821},
    {0.3247, 0.2865, 0.1969, 0.1054, 0.0439},
    {0.1353, 0.1194, 0.0821, 0.0439, 0.0183}
};

/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 *
 * Implemented by Nash, see http://opencv-code.com/quick-tips/
 */
/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat &im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows - 1; i++)
    {
        for (int j = 1; j < im.cols - 1; j++)
        {
            uchar p2 = im.at<uchar>(i - 1, j);
            uchar p3 = im.at<uchar>(i - 1, j + 1);
            uchar p4 = im.at<uchar>(i, j + 1);
            uchar p5 = im.at<uchar>(i + 1, j + 1);
            uchar p6 = im.at<uchar>(i + 1, j);
            uchar p7 = im.at<uchar>(i + 1, j - 1);
            uchar p8 = im.at<uchar>(i, j - 1);
            uchar p9 = im.at<uchar>(i - 1, j - 1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(cv::Mat &im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

#if 1
    // 細線化後、再膨張処理 (理由は未調査だが、これで大幅に性能向上)
    cv::dilate(im, prev, element, cv::Point(-1, -1), 1);
    im = prev * 255;
#else
    im *= 255;
#endif

    prev.release();
    diff.release();
}

/**
 * Deskew the image
 */
void
deskew(cv::Mat &img)
{
    cv::Moments m = cv::moments(img, true);
    cv::Mat img_deskew;
    cv::Mat rmatrix;

    if (abs(m.mu02) < 1e-2)
        return; // no op

    double skew = -(m.mu11 / m.mu02) * (180/M_PI);
    if (abs(skew) > 50)
        return; // misdetection

    rmatrix = getRotationMatrix2D(cv::Point2f(img.cols / 2.0, img.rows / 2.0),
				  skew, 1.0);

    cv::warpAffine(img, img_deskew, rmatrix, img.size(),cv::INTER_CUBIC);

    img.release();
    img = img_deskew;
}

/**
 * Detection of edge angles
 */
void
angle_detect(cv::Mat img, int angles[N][N][ANGLES])
{
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (int k = 0; k < ANGLES; k++)
	angles[i][j][k] = 0;

  for (i = 0; i < SIZE_NORMALIZED - 2; i++)
    for (j = 0; j < SIZE_NORMALIZED - 2; j++) {

      // 0 1 2
      // 7 8 3
      // 6 5 4
      int P0 = img.at<uchar>(i + 0, j + 0);
      int P1 = img.at<uchar>(i + 0, j + 1);
      int P2 = img.at<uchar>(i + 0, j + 2);
      int P3 = img.at<uchar>(i + 1, j + 2);
      int P4 = img.at<uchar>(i + 2, j + 2);
      int P5 = img.at<uchar>(i + 2, j + 1);
      int P6 = img.at<uchar>(i + 2, j + 0);
      int P7 = img.at<uchar>(i + 1, j + 0);
      int P8 = img.at<uchar>(i + 1, j + 1);

      if (!P0 && !P1 && !P2 && !P3 && !P4 && !P5 && !P6 && !P7)
	continue;

      if (!P8)
	continue;
#if ANGLES == 8
      // Angle 0
      if (P7 && P3)
	angles[j / M][i / M][0] += 1;

      // Angle 1
      if (P7 && P2 || P6 && P3)
	angles[j / M][i / M][1] += 1;

      // Angle 2
      if (P6 && P2)
	angles[j / M][i / M][2] += 1;

      // Angle 3
      if (P6 && P1 || P5 && P2)
	angles[j / M][i / M][3] += 1;

      // Angle 4
      if (P1 && P5)
	angles[j / M][i / M][4] += 1;

      // Angle 5
      if (P0 && P5 || P1 && P4)
	angles[j / M][i / M][5] += 1;

      // Angle 6
      if (P0 && P4)
	angles[j / M][i / M][6] += 1;

      // Angle 7
      if (P7 && P4 || P0 && P3)
	angles[j / M][i / M][7] += 1;
#else /* ANGLES == 4 */

      // 0 1 2    3  2  1
      // 7 8 3    ＼｜／
      // 6 5 4    ー　ー 0

      // Angle 0
      if (P7 && P3)
	angles[j / M][i / M][0] += 1;

      // Angle 1
      if (P7 && P2 || P6 && P3)
	angles[j / M][i / M][0] += 1;

      // Angle 2
      if (P6 && P2)
	angles[j / M][i / M][1] += 1;

      // Angle 3
      if (P6 && P1 || P5 && P2)
	angles[j / M][i / M][1] += 1;

      // Angle 4
      if (P1 && P5)
	angles[j / M][i / M][2] += 1;

      // Angle 5
      if (P0 && P5 || P1 && P4)
	angles[j / M][i / M][2] += 1;

      // Angle 6
      if (P0 && P4)
	angles[j / M][i / M][3] += 1;

      // Angle 7
      if (P7 && P4 || P0 && P3)
	angles[j / M][i / M][3] += 1;
#endif
    }
}

/**
 * Print angles for debug purposes
 */
void
mode_print(int angles[N][N])
{
  int i, j;

  printf("\n");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      switch(angles[j][i]) {
      case -1:
	printf("× ");
	break;
      case -2:
	printf("※ ");
	break;
      case 0:
	printf("　");
	break;
      case 1:
	printf("／");
	break;
      case 2:
	printf("／");
	break;
      case 3:
	printf("／");
	break;
      case 4:
	printf("｜");
	break;
      case 5:
	printf("＼");
	break;
      case 6:
	printf("＼");
	break;
      case 7:
	printf("＼");
	break;
      case 8:
	printf("ー");
	break;
      }
    }
    printf("|\n");
  }
}

/**
 * Detection of pixcel density
 */
void
density_detect(cv::Mat img, int density[N][N])
{
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      density[j][i] = 0;

  for (i = 0; i < SIZE_NORMALIZED; i++)
    for (j = 0; j < SIZE_NORMALIZED; j++)
      density[j / M][i / M] += img.at<uchar>(i, j) ? 1 : 0;
}

/**
 * Print angles for debug purposes
 */
void
density_print(int density[N][N])
{
  int i, j;

  printf("\n");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++)
      if (density[j][i])
	printf("%2d ", density[j][i]);
      else
	printf(" - ");
    printf("\n");
  }
  printf("\n");
}

/**
 * Detection of angle distribution
 */
void
dist_detect(int angles[N][N][ANGLES], int dist[N][N])
{
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      int dt;

      // packing
      dt = dist[j][i] =
	(angles[j][i][0] ? 0b00000001 : 0) +
	(angles[j][i][1] ? 0b00000010 : 0) +
	(angles[j][i][2] ? 0b00000100 : 0) +
	(angles[j][i][3] ? 0b00001000 : 0) +
	(angles[j][i][4] ? 0b00010000 : 0) +
	(angles[j][i][5] ? 0b00100000 : 0) +
	(angles[j][i][6] ? 0b01000000 : 0) +
	(angles[j][i][7] ? 0b10000000 : 0);

      // 共起パターン検知
      if (dt == 0) {
	dist[j][i] = 0;
	continue;
      } else if (dt == 0b10000000 ||
		 dt == 0b01000000 ||
		 dt == 0b00100000 ||
		 dt == 0b00010000 ||
		 dt == 0b00001000 ||
		 dt == 0b00000100 ||
		 dt == 0b00000010 ||
		 dt == 0b00000001) {
	dist[j][i] = 1;
      } else if (dt == 0b11000000 ||
		 dt == 0b01100000 ||
		 dt == 0b00110000 ||
		 dt == 0b00011000 ||
		 dt == 0b00001100 ||
		 dt == 0b00000110 ||
		 dt == 0b00000011 ||
		 dt == 0b10000001) {
	dist[j][i] = 2;
      } else if (dt == 0b11100000 ||
		 dt == 0b01110000 ||
		 dt == 0b00111000 ||
		 dt == 0b00011100 ||
		 dt == 0b00001110 ||
		 dt == 0b00000111 ||
		 dt == 0b10000011 ||
		 dt == 0b11000001) {
	// 変曲点疑い
	dist[j][i] = 3;
      } else {
	// 交差点疑い
	dist[j][i] = 4;
      }
    }
  }
}

/**
 * Detection of edge angles
 */
void
mode_detect(int angles[N][N][ANGLES], int mode[N][N])
{
  int i, j, total;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      int max = 0, total = 0;
      mode[j][i] = 0;

      for (int k = 0; k < 8; k++) {
	total += angles[j][i][k];

	if (max < angles[j][i][k]) {
	  max = angles[j][i][k];
	  mode[j][i] = (k == 0) ? 8 : k;
	}
      }
    }
  }
}

/**
 * Print angles for debug purposes
 */
void
angle_print(int angles[N][N][ANGLES])
{
  int i, j;

  printf("\n");

#if 0
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++)
      if (mode[j][i])
	printf("%2d ", mode[j][i]);
      else
	printf(" * ");

    printf("\n");
  }
#endif

  for (int k = 0; k < 8; k++) {
    printf("k: %d\n", k);
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++)
	if (angles[j][i][k])
	  printf("%2d ", angles[j][i][k]);
	else
	  printf(" * ");
      printf("\n");
    }
    printf("\n");
  }
}

/**
 * Detection of edge density
 */
void
edensity_detect(int angles[N][N][ANGLES], int edensity[N][N])
{
  int i, j, total;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {

      edensity[j][i] =
	(angles[j][i][0] ? 1 : 0) +
	(angles[j][i][1] ? 1 : 0) +
	(angles[j][i][2] ? 1 : 0) +
	(angles[j][i][3] ? 1 : 0) +
	(angles[j][i][4] ? 1 : 0) +
	(angles[j][i][5] ? 1 : 0) +
	(angles[j][i][6] ? 1 : 0) +
	(angles[j][i][7] ? 1 : 0);
    }
  }
}

/**
 * Blurs the angle histgram
 */
void
angle_blur(int src[N][N][ANGLES], int dst[N][N][ANGLES])
{
  short	    x, y, i, j, nu;
  double    total_weight, weight;
  double    blur[N][N][ANGLES];

  // XXX: これは定数ではないか？
  total_weight = 0;
  for (i = -2; i <= 2; i++)
    for (j = -2; j <= 2; j++)
      total_weight += Gauss[ABS(i)][ABS(j)];

  for (nu = 0; nu < 8; nu++) {
    for (x = 0; x < N; x++) {
      for (y = 0; y < N; y++) {
	blur[x][y][nu] = 0.0;
	weight = 0.0;
	for (i = -2; i <= 2; i++) {
	  for (j = -2; j <= 2; j++) {
	    if ((x + i >= 0) && (x + i < N)) {
	      if ((y + j >= 0) && (y + j < N)) {
		blur[x][y][nu] += (double) src[x + i][y + j][nu] *
		  Gauss[ABS(i)][ABS(j)];
		weight += Gauss[ABS(i)][ABS(j)];
	      }
	    }
	  }
	}
	dst[x][y][nu] = blur[x][y][nu];
	dst[x][y][nu] *= total_weight / weight;
      }
    }
  }
}

int
Extract_Feature(cv::Mat img_src, int features[N][N][ANGLES])
//Extract_Feature(cv::Mat img_src, int features[N][N])
{
    cv::Mat img_bw;
    cv::Mat img_eroded;
    cv::Mat img_dilated;
    cv::Mat img_distance;
    cv::Mat img_extracted;
    cv::Mat img_normalized;
    cv::Mat img_converted;

    int          i, j, ret = 0;
    int          cc_turn = 0;
    double       size_turn = 0;
    LabelingBS   labeling;

#ifdef DEBUG
    double       t;
    t = (double)cvGetTickCount();
#endif

    /*
     * 白黒画像取得
     */
    if (img_src.channels() > 1) {
      cv::cvtColor(img_src, img_bw, CV_BGR2GRAY);
      cv::threshold(img_bw, img_bw, 0.75 * 255, 255, CV_THRESH_BINARY);
      // cv::threshold(img_bw, img_bw, 254, 255, CV_THRESH_BINARY);
      img_bw =~ img_bw;
    } else {
      cv::threshold(img_bw, img_bw, 0.75 * 255, 255, CV_THRESH_BINARY);
      img_bw =~ img_src;
    }

    /*
     * 膨張・縮小処理
     *
     *   dilateは、白の膨張→黒の減少
     *   elode は、白の収縮→黒の膨張
     */
    double dist_min, thickness1, thickness2;
    img_dilated = cv::Mat(img_bw.size(), CV_8UC1);
    img_eroded  = cv::Mat(img_bw.size(), CV_8UC1);

    // やるべき処理
    //
    // かすれ画像: 膨張・収縮で実線化
    //   閉じるべきループを閉じる (8の右上等)
    // ちぎれ画像: 膨張で接続
    //
    // ただし：
    //   閉じるべきでないループを閉じない (6の右上)
    //   閉じるべきでないループを閉じない (8や9の小さなループ)

 backtrack:
    cv::dilate(img_bw, img_dilated, element, cv::Point(-1, -1), 1);
    cv::erode(img_dilated, img_eroded, element, cv::Point(-1, -1), 1);

    // ループの消失をチェック
    cv::distanceTransform(img_bw, img_distance, CV_DIST_L2, 3);
    cv::minMaxLoc(img_distance, &dist_min, &thickness1);
    img_distance.release(); // メモリリーク対策 (不要？)

    cv::distanceTransform(img_eroded, img_distance, CV_DIST_L2, 3);
    cv::minMaxLoc(img_distance, &dist_min, &thickness2);

    // 中心線抽出処理 (太線の際、中心線のみを抽出する)
    if (!cc_turn && thickness2 > 8) { //  && thickness2 < 14) {
      // printf("%s: %d, %d\n", filename, (int) thickness2, cc_turn);

      for (i = 0; i < img_eroded.size().height; i++)
	for (j = 0; j < img_eroded.size().width; j++) {
	  // このthicknessの閾値により、認識精度がわずかに変化する
	  if (img_distance.at<float>(i, j) > thickness2 - 1)
	    img_eroded.at<uchar>(i, j) = 0;
	  else if (img_distance.at<float>(i, j) < thickness2 / 2 - 1)
	    img_eroded.at<uchar>(i, j) = 0;
	}
    }

    if ((double) (thickness2 / thickness1) >
	(ELIMINATION_THRESHOLD + size_turn)) {
      img_eroded.release(); // メモリリーク対策 (不要？)
      img_eroded = img_bw;
    }

    /*
     * ラベリング処理 (罫線除去・ノイズ除去・文字要素の抽出)
     */
    unsigned char  *img_label = new unsigned char[img_eroded.size().height *
                                                  img_eroded.size().width];
    short          *cc_result = new short[img_eroded.size().height *
                                          img_eroded.size().width];
    int             size_x, size_y, top_x, top_y, bottom_x, bottom_y;
    int             cc, num_of_cc;
    int             src_width, src_height, dst_width, dst_height, dst_size;
    int             padding_x, padding_y;
    double          aspect_ratio;
    RegionInfoBS   *ri;

    // ラベリング用データ生成 (XXX: 非効率)
    for (i = 0; i < img_eroded.size().width; i++) {
      for (j = 0; j < img_eroded.size().height; j++) {
        img_label[j * img_eroded.size().width + i] =
          img_eroded.at<uchar>(j, i); // atは (y, x)
      }
    }

    // true: 領域の大きな順にソートする場合
    // 3: 領域検出の最小領域
    labeling.Exec(img_label, cc_result,
                  img_eroded.size().width,
                  img_eroded.size().height, true, 3);

    num_of_cc = labeling.GetNumOfResultRegions();
    if (!num_of_cc) {
      ret = ERR_RTN;
      goto finish;
    }

    //「罫線らしくないCCのうち、最大サイズのCC」を文字として選択
    for (cc = 0; cc < num_of_cc; cc++) {
        ri = labeling.GetResultRegionInfo(cc);
        ri->GetSize(size_x, size_y);
        if (size_x > size_y)
            aspect_ratio = (double)size_x / (double)size_y;
        else
            aspect_ratio = (double)size_y / (double)size_x;

        // 罫線っぽい連結成分をスキップ
        if (!(
	      (aspect_ratio > 8) &&
	      (size_x > img_eroded.size().width - 2 ||
	       size_y > img_eroded.size().height - 2)
              // (ri->GetNumOfPixels() > img_eroded.size().height / 2)
	      )
	    ) {
            break;
        }
    }

    // 文字カスレが疑われれば、1回に限りバックトラック
    if (cc_turn++ < 2 && cc + 1 < num_of_cc) {
      int size_a = ri->GetNumOfPixels();
      ri = labeling.GetResultRegionInfo(cc + 1);
      int size_b = ri->GetNumOfPixels();
      if (size_a / size_b < 10) {
	// 要素を1つのみ選択するアルゴリズムのため、膨張して結合する目的で
	// バックトラックをしているが、一定サイズの要素を複数選択する実装に
	// 変更した方が良いかもしれない
        goto backtrack;
      }
      ri = labeling.GetResultRegionInfo(cc);
    }

    // 切り出し用サイズの生成
    ri->GetMax(top_x, top_y);
    ri->GetMin(bottom_x, bottom_y);
    src_width = img_eroded.size().width;
    src_height = img_eroded.size().height;
    dst_width = ABS(top_x - bottom_x);
    dst_height = ABS(top_y - bottom_y);
    dst_size = MAX(dst_width, dst_height) + 2; // 上下マージン

    // 文字サイズが小さければ、カスレの閾値について再検証
    if (size_turn == 0 && dst_width < 40 && dst_height < 40) {
      size_turn = 0.05;
	if ((double) (thickness2 / thickness1) <
	    (ELIMINATION_THRESHOLD + size_turn)) {
	  goto backtrack;
	}
    }

    /*
     * 対象要素の切り出し
     */
    img_extracted = cv::Mat::zeros(cv::Size(dst_size, dst_size), CV_8UC1);

    // センタリング (不利な気もするが、centeringが最も性能が良い)
    padding_x = dst_size / 2 - dst_width / 2;
    padding_y = dst_size / 2 - dst_height / 2;

    for (i = 0; i < dst_size; i++) {
      for (j = 0; j < dst_size; j++) {
        if (i + bottom_x - padding_x < src_width &&
	    j + bottom_y - padding_y < src_height &&
	    0 < i + bottom_x - padding_x &&
	    0 < j + bottom_y - padding_y < src_height) {
	  if (cc_result[src_width * (j + bottom_y - padding_y) +
			i + bottom_x - padding_x] == (cc + 1)) {
	    img_extracted.at<uchar>(j, i) = (uchar) 255;
	  } else {
	    img_extracted.at<uchar>(j, i) = (uchar) 0;
	  }
	}
      }
    }

    /*
     * Deskew処理
     */
    deskew(img_extracted);

    /*
     * サイズ正規化処理
     */
    resize(img_extracted, img_normalized,
	   cv::Size(SIZE_NORMALIZED, SIZE_NORMALIZED),
           SIZE_NORMALIZED / img_extracted.size().width,
           SIZE_NORMALIZED / img_extracted.size().height, CV_INTER_AREA);

    // 外周マージン再確保 (各種アルゴリズム上、マージンがあるほうが効率が良い)
    for (i = 0; i < SIZE_NORMALIZED; i++) {
      img_normalized.at<uchar>(0, i) = (uchar) 0;
      img_normalized.at<uchar>(SIZE_NORMALIZED - 1, i) = (uchar) 0;
    }
    for (j = 0; j < SIZE_NORMALIZED; j++) {
      img_normalized.at<uchar>(j, 0) = (uchar) 0;
      img_normalized.at<uchar>(j, SIZE_NORMALIZED - 1) = (uchar) 0;
    }

    /*
     * 細線化処理
     */
    thinning(img_normalized);

    /*
     * 各種成分の抽出処理 (XXX: 性能比較のため、複数手法が残っている)
     */
    // 角度検出
    int angles[N][N][ANGLES];
    angle_detect(img_normalized, angles);

#if 0
    // 角度成分密度
    int edensity[N][N];
    edensity_detect(angles, edensity);

    // 角度分散
    int dist[N][N];
    dist_detect(angles, dist);

    // 角度最頻値
    int mode[N][N];
    mode_detect(angles, mode);

    // ピクセル密度
    int pdensity[N][N];
    density_detect(img_normalized, pdensity);
#endif

    // 抽出角度成分のぼけ変換
    int angles_f[N][N][ANGLES];
    angle_blur(angles, angles_f);

    /*
     * 特徴量の返り値作成 (XXX: 非効率)
     */
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
#if 0
	features[j][i] = pdensity[j][i];
#else
	for (int k = 0; k < ANGLES; k++)
	  features[j][i][k] = angles_f[j][i][k];
#endif
      }
    }

#ifdef DEBUG
    t = (double)cvGetTickCount() - t;
    printf("%gms\n", t / ((double) cvGetTickFrequency() * 1000.0));
#endif

#ifdef DEBUG
    /*
     * 保存処理
     */
    if (filename[0] != '\0') {
      imwrite(filename, img_normalized);
    }

    // mode_print(mode);
    // density_print(dist);
    // angle_print(angles);
    // angle_print(angles_f);

    cv::imshow("src", img_src);
    cv::imshow("bw", img_bw);
    cv::imshow("dilated", img_dilated);
    cv::imshow("eroded", img_eroded);
    cv::imshow("extracted", img_extracted);
    cv::imshow("noamlized", img_normalized);
    cv::waitKey(0);
    cvDestroyAllWindows();
#endif
#ifdef THINNING_MAIN
    checkPreprocessedImage(img_src, img_bw, img_dilated, img_eroded, img_extracted, img_normalized);
#endif /* THINNING_MAIN */

 finish:
    img_bw.release();
    img_eroded.release();
    img_dilated.release();
    img_distance.release();
    img_extracted.release();
    img_normalized.release();
    img_converted.release();

    return ret;
}

int
Extract_Feature_wrapper(char *fname, int features[N][N][ANGLES])
//Extract_Feature_wrapper(char *fname, int features[N][N])
{
  cv::Mat img_src;
  int ret;

  img_src = cv::imread(fname);
  if (img_src.empty()) {
    fprintf(stderr, "image file \"%s\": cannot be found.\n", fname);
    return ERR_RTN;
  }

  if (strlen(fname) >= FILENAME_SIZE) {
    fprintf(stderr, "file name \"%s\": too long.\n", fname);
    return ERR_RTN;
  }

  strcpy(filename, fname);
  strncpy(strrchr(filename, '.'), "-conv.png", 10);

  ret = Extract_Feature(img_src, features);
  filename[0] = '\0';

  return ret;
}

#ifdef THINNING_MAIN
/**
 * For debugging purpose
 */
#include <getopt.h>

/**
 * create an image that contains a lot of images in one image, and return it in cv::Mat
 *
 * parameters: out_img: to store a generated image
 *             other: images in cv::Mat
 *
 * return: 0: success
 *         2: cannot create temporary directory
 *         3: an error happens when executing the montage command
 *         4: cannot clean up temporary directory created in this function
 *
 * This function requires "montage" command from ImageMagick
 */
static int
getMontagedImageAfterPreprocessing(cv::Mat &out_img, const cv::Mat &img_src, const cv::Mat &img_bw, const cv::Mat &img_dilated, const cv::Mat &img_eroded, const cv::Mat &img_extracted, const cv::Mat &img_normalized)
{
    int returnCode = 0;

    char *montageCmd = (char *)malloc(sizeof(char) * 1024 + 1);
    char *montagedFile = (char *)malloc(sizeof(char) * PATH_MAX + 1);
    char *rmCmd = (char *)malloc(sizeof(char) * 1024 + 1);
    char *dirTemplate = (char *)malloc(sizeof(char) * PATH_MAX + 1);
    char *thinningTmpFile = (char *)malloc(sizeof(char) * PATH_MAX + 1);
    char *thinningTmpDir;

    do {
        /* get a prefix of temporary directory */
        const char *tmpDir = getenv("TMPDIR");
        if (tmpDir == NULL) {
            tmpDir = "/tmp";
        }

        /* make a temporary directory to store image files used in this function */
        snprintf(dirTemplate, PATH_MAX, "%s/kocrThinning.XXXXXX", tmpDir);
        thinningTmpDir = mkdtemp(dirTemplate);
        if (thinningTmpDir == NULL) {
            returnCode = 2;
            break;
        }

        /* writing the images to files */
        snprintf(thinningTmpFile, PATH_MAX, "%s/0src.png", thinningTmpDir);
        imwrite(thinningTmpFile, img_src);
        snprintf(thinningTmpFile, PATH_MAX, "%s/bw.png", thinningTmpDir);
        imwrite(thinningTmpFile, img_bw);
        snprintf(thinningTmpFile, PATH_MAX, "%s/dilated.png", thinningTmpDir);
        imwrite(thinningTmpFile, img_dilated);
        snprintf(thinningTmpFile, PATH_MAX, "%s/eroded.png", thinningTmpDir);
        imwrite(thinningTmpFile, img_eroded);
        snprintf(thinningTmpFile, PATH_MAX, "%s/extracted.png", thinningTmpDir);
        imwrite(thinningTmpFile, img_extracted);
        snprintf(thinningTmpFile, PATH_MAX, "%s/normalized.png", thinningTmpDir);
        imwrite(thinningTmpFile, img_normalized);

        /* calling montage command */
        snprintf(montagedFile, PATH_MAX, "%s/montage.png", thinningTmpDir);
        snprintf(montageCmd, 1024, "montage %s/*.png -tile x1 %s", thinningTmpDir, montagedFile);
        if (system(montageCmd) != 0) {
            returnCode = 3;
            break;
        }

        /* return the generated image */
        out_img = cv::imread(montagedFile);
    } while(0);

    /* clean up the temporary directory */
    if (thinningTmpDir != NULL) {
        snprintf(rmCmd, 1024, "rm -r %s", thinningTmpDir);
        if (system(rmCmd) != 0) {
            returnCode = 4;
        }
    }

    /* free memory space */
    free(dirTemplate);
    free(thinningTmpFile);
    free(rmCmd);
    free(montageCmd);
    free(montagedFile);

    return returnCode;
}

/**
 * check a image preprocessed in each step
 *
 * parameters: images in cv::Mat
 *
 * return: same return code from getMontagedImageAfterPreprocessing() function
 */
static int
checkPreprocessedImage(const cv::Mat &img_src, const cv::Mat &img_bw, const cv::Mat &img_dilated, const cv::Mat &img_eroded, const cv::Mat &img_extracted, const cv::Mat &img_normalized)
{
    int retCode = 0;
    cv::Mat montagedImage;

    /* return if no command line options are specified */
    if (showPreprocessedImageFlag != 1 && writePreprocessedImageFileName == NULL) {
        return 1;
    }

    retCode = getMontagedImageAfterPreprocessing(montagedImage, img_src, img_bw, img_dilated, img_eroded, img_extracted, img_normalized);
    if (retCode != 0 && retCode <= 3) {
        return retCode;
    }

    if (showPreprocessedImageFlag == 1) {
        cv::imshow("preprocessed image", montagedImage);
        cv::waitKey(0);
        cvDestroyAllWindows();
    }
    if (writePreprocessedImageFileName != NULL) {
        imwrite(writePreprocessedImageFileName , montagedImage);
    }

    return retCode;
}

static void
usage(char *cmdName)
{
    printf("%s [options] a-image-file\n", cmdName);
    printf("\t--help\tshow usage\n");
    printf("\t--show-image\tshow pre-processed images at each step\n");
    printf("\t--write-image=imagefile\twrite preprocessed images to the file in each step\n");
}

int
main(int argc, char *argv[])
{
    /* commandline option handling */
    int optindex;
    int optid;
    struct option opts[] = {
        { "help", no_argument, NULL, 'h'},
        { "show-image", no_argument, NULL, 0},
        { "write-image", required_argument, NULL, 1},
        { 0, 0, 0, 0},
    };
    while ((optid = getopt_long(argc, argv, "h", opts, &optindex)) != -1) {
        switch (optid) {
            case 'h':
                usage(argv[0]);
                return 1;
                break;
            case 0:
                showPreprocessedImageFlag = 1;
                break;
            case 1:
                writePreprocessedImageFileName = optarg;
                break;
        }
    }

    if (argv[optind] == NULL) {
        usage(argv[0]);
        return 1;
    }
    printf("# %s\n", argv[optind]);
#ifdef USE_CNN
    cv::Mat img_src;
    img_src = cv::imread(argv[optind]);
    if (img_src.empty()) {
        fprintf(stderr, "image file \"%s\": cannot be found.\n", argv[optind]);
        return 1;
    }
    preprocessing_for_cnn(img_src);
    return 0;
#else
    int features[N][N][ANGLES];
    return Extract_Feature_wrapper(argv[optind], features);
#endif /* USE_CNN */
}
#endif

#endif /* THINNING */
