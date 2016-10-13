#include <cstdio>
#include <opencv2/highgui/highgui.hpp>

cv::Mat preprocessing_for_cnn(cv::Mat);

void usage(){
    printf("usage:\n");
    printf(" $ preprocess target");
    printf(" (target: *.[png|pbm|jpg])\n");
}

int main(int argc, char *argv[]){
    if (argc != 2){
        usage();
        return 0;
    }

    IplImage *src_img = cvLoadImage(argv[1]);
    if (!src_img) {
        return 1;
    }

    cv::Mat src_mat = preprocessing_for_cnn(cv::cvarrToMat(src_img, true));
    if (src_mat.data == NULL){
        return 1;
    }

    strncpy(strrchr(argv[1], '.'), "-conv.png", 10);
    imwrite(argv[1], src_mat);

    return 0;
}
