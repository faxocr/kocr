#include <vector>
#include <string>
#include <string.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "forward_cnn.h"
#include "cropnums.h"
#include "kocr_cnn.h"


char *recognize(Network *net, IplImage *src_img){
    std::vector<int> src_shape(4);
    src_shape[0] = 1;  // num of images
    src_shape[1] = 1;  // channel
    src_shape[2] = 48;  // row
    src_shape[3] = 48;  // column

    Tensor<float> src_tensor(src_shape);
    cv::Mat src_mat = preprocessing_for_cnn(cv::cvarrToMat(src_img, true));
    if (src_mat.data == NULL){
      // some error occured in preprocessing,
      // it may be that the input image has no black area.
      return NULL;
    }

    for(int i=0;i<src_tensor.n;i++){
        src_tensor.ix(i) = (float)src_mat.data[i] / 255;
    }
    src_mat.release();

    std::string response = net->predict_labels(src_tensor)[0];

#ifndef LIBRARY
    printf("Recogized: %s\n", response.c_str());
#endif

    return strdup(response.c_str());
}


char *recognize_multi(Network *net, IplImage *src_img) {
    IplImage    *dst_img = NULL;
    CvRect	    bb;
    IplImage    *part_img, *body;
    int		    seq_num, start_x, width, next_start;
    char	    *result_str;

    // prepare Tensor as input for Network
    std::vector<int> src_shape(4);
    src_shape[0] = 1;  // num of images
    src_shape[1] = 1;  // channel
    src_shape[2] = 48;  // row
    src_shape[3] = 48;  // column

    Tensor<float> src_tensor(src_shape);  // reuse this Tensor for all images

    // 白黒に変換する(0,255の二値)
    dst_img = cvCreateImage(cvSize(src_img->width, src_img->height), 8, 1);
    cvThreshold(src_img, src_img, 120, 255, CV_THRESH_BINARY);

    // 文字列全体のBB
    bb = findBB(src_img); // cropnums.cpp内で宣言
    body = cvCreateImage(cvSize(bb.width, bb.height), src_img->depth, src_img->nChannels);
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

        cv::Mat src_mat = preprocessing_for_cnn(cv::cvarrToMat(part_img, true));
        for(int i=0;i<src_tensor.n;i++){
            src_tensor.ix(i) = (float)src_mat.data[i] / 255;
        }
        src_mat.release();

        int response = net->predict_classes(src_tensor)[0];
        result_str[seq_num] = (char)(response + '0');
        result_str[seq_num + 1] = 0;

#ifndef LIBRARY
        printf("Recogized: %d\n", response);
#endif

        start_x = next_start;
        seq_num++;
    }

    return result_str;
}


char *recog_image(Network *net, IplImage *src_img){
    char *result;

    if (src_img->width / src_img->height > THRES_RATIO)
        result = recognize_multi(net, src_img);
    else
        result = recognize(net, src_img);
    return result;
}

int read_int(std::ifstream& ifs){
    int n;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int));
    return n;
}

Network *kocr_cnn_init(char *filename){
    if (filename == NULL)
        return (Network *) NULL;

    std::ifstream ifs(filename);
    int nb_classes = read_int(ifs);

    std::vector<std::string> unique_labels(nb_classes);
    for(int i=0;i<nb_classes;i++){
        int str_len = read_int(ifs);
        std::vector<char> str(str_len);
        ifs.read(str.data(), sizeof(char) * str_len);
        unique_labels[i].assign(str.begin(), str.end());
    }

    Network *net;
    net = new Network();

    std::vector<int> input_shape(3);
    input_shape[0] = 1;
    input_shape[1] = 48;
    input_shape[2] = 48;

    net->add(new Convolution2D(32, 9, 9, input_shape));
    net->add(new Relu());
    net->add(new MaxPooling2D(2, 2));
    net->add(new Dropout(0.5));

    net->add(new Convolution2D(64, 5, 5));
    net->add(new Relu());
    net->add(new MaxPooling2D(2, 2));
    net->add(new Dropout(0.5));

    net->add(new Convolution2D(128, 3, 3));
    net->add(new Relu());
    net->add(new MaxPooling2D(2, 2));
    net->add(new Dropout(0.5));

    net->add(new Flatten());
    net->add(new Dense(128));
    net->add(new Relu());
    net->add(new Dropout(0.5));

    net->add(new Dense(nb_classes));
    net->add(new Softmax());

    net->build();
    net->load_weights(ifs);
    net->set_label(unique_labels);

    return net;
}

void kocr_cnn_finish(Network *net){
    if (net != NULL) {
        delete net;
    }
}

char *kocr_recognize_image(Network *net, char *file_name){
    IplImage       *src_img;
    char *c;

    if (net == NULL || file_name == NULL){
      printf("test point 1\n");
      return NULL;
    }

    src_img = cvLoadImage(file_name);

    // OpenCV does not support GIF format
    if (!src_img) {
        char *p = file_name;
        for (; *p; ++p) *p = tolower(*p);
        if (strstr(file_name, ".gif")) {
            printf("This program doesn't support GIF images.\n");
        }else{
            printf("An error occurred in loading images. Please check that the file exists.\n");
        }
        return NULL;
    }

    c = recog_image(net, src_img);
    cvReleaseImage(&src_img);

    return c;
}


char *kocr_recognize_Image(Network *net, IplImage *src_img){
    if (net == NULL || src_img == NULL)
        return NULL;

    return recog_image(net, src_img);
}
