/*
 * Copyright (c) 2012, Masahiko KIMOTO, Ph.D. <kimoto@ohnolab.org>
 * All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>

#define BWNOISE_THRESHOLD 95

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

CvRect
findBB(IplImage * imgSrc)
{
    int		    minX  , minY, maxX, maxY;
    CvScalar	    s , white;
    int		    x     , y;
    CvRect	    rect;

    white = cvGet2D(imgSrc, 0, 0);	/* XXX */

    minX = imgSrc->width;
    minY = imgSrc->height;
    maxX = 0;
    maxY = 0;

    for (x = 0; x < imgSrc->width - 1; x++) {
	for (y = 0; y < imgSrc->height - 1; y++) {
	    s = cvGet2D(imgSrc, y, x);
	    if (s.val[0] != white.val[0]) {
		minX = min(minX, x);
		minY = min(minY, y);
		maxX = max(maxX, x);
		maxY = max(maxY, y);
	    }
	}
    }

    rect = cvRect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    return rect;
}

IplImage       *
cropnum(IplImage * src_img, int startx, int *nextstart)
{
    IplImage       *result;
    CvMat	    dataA;
    CvRect	    bba , rect;
    //bounding box
	CvScalar maxVal;
    CvScalar	    val = cvRealScalar(0);

    int		    size  , x, y;
    int		    allwhite, state;

    state = 0;
    bba.y = 0;
    bba.height = src_img->height;
    bba.x = startx;
    bba.width = src_img->width - startx;

    if (startx >= src_img->width)
	return NULL;

    maxVal = cvRealScalar(src_img->height * ((1 << (src_img->depth)) - 1));

    for (x = startx; x < src_img->width; x++) {
	CvMat		data;
#ifdef DEBUG
	printf("x:%d %d\n", x, src_img->width);
#endif
	cvGetCol(src_img, &data, x);
	val = cvSum(&data);

#ifdef DEBUG
	printf("val %d/%d\n", (int)val.val[0], (int)maxVal.val[0]);
#endif

	if ((val.val[0] * 100) < (maxVal.val[0] * BWNOISE_THRESHOLD)) {
	    /* black is present in this columun */
	    allwhite = 0;
	} else {
	    /* the line is almost white */
	    allwhite = 1;
	}

	if (state == 0 && allwhite == 1) {
	    bba.x = x;
	    continue;
	}
	if (state == 0 && allwhite == 0) {
	    state = 1;
	    bba.x = x;
	    continue;
	}
	if (state == 1 && allwhite == 1) {
	    bba.width = (x - bba.x);
	    break;
	}
	if (state == 1 && allwhite == 0) {
	    bba.width = (x - bba.x);
	    continue;
	}
    }

    *nextstart = x;

    //bba is rectangle of a Number
#ifdef DEBUG
	printf("BB: %d,%d - %d,%d from %d,%d\n",
	       bba.x, bba.y, bba.width, bba.height,
	       src_img->width, src_img->height);
#endif

    cvSetImageROI(src_img, bba);

    rect = cvGetImageROI(src_img);
#ifdef DEBUG
    printf("src:  x=%d, y=%d, width=%d, height=%d\n",
	   rect.x, rect.y,
	   rect.width, rect.height);
#endif

    //create a image with bba
	result = cvCreateImage(cvSize(rect.width, rect.height),
			       src_img->depth, 1);

    if (result == NULL)
	return NULL;

    cvSet(result, CV_RGB(255, 255, 255), NULL);

#ifdef DEBUG
    printf("dst: depth=%d, width=%d, height=%d\n",
	   result->depth, result->width, result->height);
#endif

    //copy rectangle part of src_img to result image
	cvCopy(src_img, result, NULL);
    cvResetImageROI(src_img);

    return result;

}

static void
do_split(IplImage * src_img)
{
    CvRect	    bb;
    IplImage       *part_img, *body;
    int		    seqnum, startx, width, nextstart;
    char	    filename[BUFSIZ];

    //crop with BB of all chars
	bb = findBB(src_img);
    body = cvCreateImage(cvSize(bb.width, bb.height), src_img->depth, 1);
    cvSet(body, CV_RGB(255, 255, 255), NULL);
    cvSetImageROI(src_img, bb);
    cvCopy(src_img, body, NULL);

    startx = 0;
    seqnum = 0;

    width = body->width;

    while (startx < width) {
	part_img = cropnum(body, startx, &nextstart);
	if (part_img == NULL || part_img->width == 0)
	    break;
        //NOTE:do something other than saving image here
#ifdef DEBUG
	sprintf(filename, "file%03d.png", seqnum);
	cvSaveImage(filename, part_img, 0);
#endif

	startx = nextstart;
	seqnum++;
	cvReleaseImage(&part_img);
    }
}

#ifdef UNIT_TEST
static void
err_exit() {
    fprintf(stderr, "error\n");
    exit(1);
}

int
main(int argc, char **argv)
{
    IplImage       *src_img = NULL, *dst_img = NULL;
    int		    newwidth = 0;
    IplImage       *prs_img;
    
    if (argc < 2) {
	fprintf(stderr, "usage: cropnums inputfile\n");
	exit(1);
    }
    
    src_img = cvLoadImage(argv[1], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR
	);
    if (src_img == NULL)
	err_exit();
    
    dst_img = cvCreateImage(cvSize(src_img->width, src_img->height),
			    8, 1);
#if 0
    //pre - processing
    {
	IplImage       *grayImage;
	IplImage       *binImage;
	
	grayImage = cvCreateImage(cvSize(src_img->width, src_img->height), 8, 1);
	binImage = cvCreateImage(cvSize(src_img->width, src_img->height), 8, 1);
	
	//cvCvtColor(src_img, grayImage, CV_BGR2GRAY);
	cvDilate(src_img, grayImage, NULL, 1);
	cvSmooth(grayImage, grayImage, CV_GAUSSIAN, 21, 21, 0, 0);
	cvThreshold(grayImage, binImage, 120, 255, CV_THRESH_BINARY);
	cvNormalize(binImage, dst_img, 0, 1, CV_MINMAX, 0);
    }
#endif
    
    cvThreshold(src_img, dst_img, 120, 255, CV_THRESH_BINARY);
    //dst_img is BW format
    do_split(dst_img);
    
    cvReleaseImage(&src_img);
    
    return 1;
}

#endif
