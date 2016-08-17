#ifndef KOCR_CNN_H
#define KOCR_CNN_H

#define THRES_RATIO 2
#define MAXSTRLEN 1024

// Implemented in thinning.cpp
cv::Mat preprocessing_for_cnn(cv::Mat);

char *recognize(Network *, IplImage *);
char *recognize_multi(Network *, IplImage *);
char *recog_image(Network *, IplImage *);

// training is not implemented yet
/*
char *conv_fname(char *, const char *);
int is_database(const char *);
int is_opencvxml(const char *);

Network *training(char *);
void leave_one_out_test(feature_db *, char *);

void kocr_exclude(feature_db * db, char *lst_name);
void kocr_distance(feature_db * db, char *lst_name);
void kocr_average(feature_db * db, char *lst_name);
*/

Network *kocr_cnn_init(char *);
void kocr_cnn_finish(Network *);
char *kocr_recognize_image(Network *, char *);
char *kocr_recognize_Image(Network *, IplImage *);

#endif /* KOCR_CNN_H */
