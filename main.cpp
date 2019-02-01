#include "machine_learning.hpp"
int iteration;

int main(int argc, char **argv)
{
    int inf;
    int n_cluster = atoi( argv[2] );
    float features[10];
    char infected_file[20] = "";
    int mode = atoi( argv[3] );

    iteration = 0;

    if( argc != 4 )
    {
        std::cerr << "Usage: " << argv[0] << "<InputImage> <NumberOfClusters> <Learn - 0 / Test - 1>" << std::endl;
        return EXIT_FAILURE;
    }

    // load a specified file as a 3-channel color image
    cv::Mat src_img = cv::imread( argv[1] );
    cv::Mat hsi_img;

    rgb_to_hsi(src_img);
    hsi_img = cv::imread("PROCESS_PIC/h.jpg");

    kmeans_clustering(hsi_img, n_cluster);
    cout <<"Enter the infected cluster number:"<<endl;
    cin >> inf;

    sprintf(infected_file,"%s%d%s","PROCESS_PIC/", inf, ".bmp");
    cv::Mat Infected_Cluster = imread(infected_file);
    feature_extraction(Infected_Cluster, features);

    svm_classifier(features, mode);

    return TRUE;

}



