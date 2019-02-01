#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include<string>
#include <ml.h>

using namespace std;
using namespace cv;

void rgb_to_hsi(cv::Mat src);
void kmeans_clustering(cv::Mat hsi_img, int n);
void feature_extraction(cv::Mat Infected_Cluster, float* features);
void svm_classifier(float* features, int mode);
