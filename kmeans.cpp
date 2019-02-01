#include "machine_learning.hpp"

#define BACKGROND_COLOR_CHANNELS (3)
#define BACKGROND_COLOR_RED (2)
#define BACKGROND_COLOR_GREEN (0)
#define BACKGROND_COLOR_BLUE (1)
#define RESULT_FILE_WORDS (200)


void kmeans_clustering(cv::Mat hsi_img, int NumberOfClusters)
{
    int i,j, size;
    char file_name[] = "";
    char file_extension[] = ".bmp";
    int background_color[BACKGROND_COLOR_CHANNELS] = {255, 255, 255}; //{255,135,60};


    if(!hsi_img.data)
    {
        cerr << "Error: Loading image" << endl;
    }

    // (2)reshape the image to be a 1 column matrix
    cv::Mat points;
    hsi_img.convertTo(points, CV_32FC3);
    size = hsi_img.rows*hsi_img.cols;
    points = points.reshape(3, size);

    // (3)run k-means clustering algorithm to segment pixels in RGB color space
    cv::Mat_<int> clusters(points.size(), CV_32SC1);
    cv::Mat centers;
    cv::kmeans(points, NumberOfClusters, clusters,
    cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);

    // (4)make a each centroid represent all pixels in the cluster
    cv::Mat dst_img(hsi_img.size(), hsi_img.type());
    cv::MatIterator_<cv::Vec3f> itf = centers.begin<cv::Vec3f>();
    cv::MatIterator_<cv::Vec3b> itd = dst_img.begin<cv::Vec3b>(), itd_end = dst_img.end<cv::Vec3b>();

#if 1
    for(int i=0; itd != itd_end; ++itd, ++i) {
        cv::Vec3f color = itf[clusters(1,i)];
        (*itd)[0] = cv::saturate_cast<uchar>(color[0]);
        (*itd)[1] = cv::saturate_cast<uchar>(color[1]);
        (*itd)[2] = cv::saturate_cast<uchar>(color[2]);
    }
#endif
    cv::imwrite("PROCESS_PIC/cluster.bmp", dst_img);


#if 1
    for(j=0; j<NumberOfClusters; j++)
    {
        char file[RESULT_FILE_WORDS] = "";//DON'T DELETE THIS! OR VARIABLE file ISN'T INITIALIZE.
        sprintf(file,"%s%s%d%s","PROCESS_PIC/",file_name,j,file_extension);
        itd = dst_img.begin<cv::Vec3b>();
        for(int i=0; itd != itd_end; ++itd, ++i)
        {
            int idx = clusters(1,i);
            cv::Vec3f color = itf[clusters(1,i)];
            if(j == idx)
            {
                (*itd)[0] = cv::saturate_cast<uchar>(color[0]);
                (*itd)[1] = cv::saturate_cast<uchar>(color[1]);
                (*itd)[2] = cv::saturate_cast<uchar>(color[2]);
            }
            else
            {
                (*itd)[0] = background_color[BACKGROND_COLOR_BLUE];
                (*itd)[1] = background_color[BACKGROND_COLOR_GREEN];
                (*itd)[2] = background_color[BACKGROND_COLOR_RED];
            }
        }
        cv::imwrite(file, dst_img);
        printf("cluster %d image save completed.\n",j);
    }
#endif

points.release();
hsi_img.release();
clusters.release();
centers.release();
dst_img.release();



}
