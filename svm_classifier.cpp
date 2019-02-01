#include "machine_learning.hpp"

extern int iteration;

const int numTrainingPoints=15;
const int attributes_per_sample = 10;
const int numTestPoints = 1;

int write_data_to_txt(float *matrix)
{
    int i = 0;;

    std::fstream outputFile;
    outputFile.open( "FILES/file.txt", std::ios::out ) ;

    for(int j=0; j<10; j++)
    {
        outputFile << matrix[j] <<", ";
    }
    outputFile.close( );
    return 1; // all OK
}

int write_data_to_csv(Mat &matrix)
{
    std::fstream outputFile;
    outputFile.open( "FILES/file.csv", std::ios::out ) ;

    for(int i=0; i<matrix.rows; i++)
    {
        for(int j=0; j<matrix.cols; j++)
        {
            outputFile << matrix.at<float>(i,j) << ", ";
        }
        outputFile << endl;
    }
    outputFile.close( );
    return 1; // all OK
}

void svm_classifier(float* features, int mode)
{
    static float trainsamples[numTrainingPoints][attributes_per_sample] = {

	{0.256358, -0.0213387, 255, 190.967, 255, 24, 255, -0.987266, 103.399, 10691.4},
	{0.22598, 0.975163, 255, 214.39, 255, 66, 255, -1.41048, 77.6283, 6026.16},
	{0.291632, -0.819279, 255, 161.063, 255, 18, 255, -0.435894, 115.926, 13438.9},
	{0.251281, 0.132531, 255, 208.561, 255, 80, 255, -1.06708, 77.2676, 5970.27},
	{0.287432, -0.725677, 255, 193.412, 255, 91, 255, -0.523495, 79.4189, 6307.35},
	{0.268393, -0.300596, 255, 200.373, 255, 78, 255, -0.84024, 81.761, 6684.86},
	{0.25848, -0.0737439, 255, 201.916, 255, 67, 255, -0.960004, 84.628, 7161.91},
	{0.276667, -0.512145, 255, 194.861, 255, 75, 255, -0.696319, 84.902, 7208.35},
	{0.273636, -0.419202, 255, 198.247, 255, 80, 255, -0.766928, 81.9199, 6710.88},
	{0.225073, 1.00867, 255, 213.2, 255, 59, 255, -1.42458, 80.2842, 6445.55},
	{0.247863, 0.259858, 255, 209.115, 255, 77, 255, -1.13068, 77.8597, 6062.13},
	{0.273394, -0.41391, 255, 192.232, 255, 61, 255, -0.769851, 90.7588, 8237.15},
	{0.119706, 10.0558, 255, 253.189, 255, 232, 255, -3.4225, 6.19395, 38.3651},
	{0.253352, 0.0851284, 255, 190.444, 255, 16, 255, -1.04421, 106.12, 11261.4},
	{0.182016, 3.01112, 255, 229.112, 255, 80, 255, -2.01372, 62.1309, 3860.24},

    };

    //Anthracnose = (1.0), Gall midge = (-1.0)
    float labels[numTrainingPoints] = {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0};

    if (mode == 0) 
    {
        cout <<"Learning mode activated"<<endl;
        for (int i=0; i<10; i++)
        write_data_to_txt(features);
        return;
    }

    else if (mode == 1) {
    cout <<"Testing mode activated"<<endl<<endl;
    cv::Mat trainingData(numTrainingPoints, attributes_per_sample, CV_32FC1, &trainsamples);
    cv::Mat testData(numTestPoints, attributes_per_sample, CV_32FC1, features);
    cv::Mat trainingClasses(numTrainingPoints, 1, CV_32FC1, &labels);

    write_data_to_csv(trainingData);


    // Set up SVM's parameters
    CvSVMParams param;
    //params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::LINEAR;
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    param.svm_type = CvSVM::C_SVC;
    param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
    param.degree = 0; // for poly
    param.gamma = 20; // for poly/rbf/sigmoid
    param.coef0 = 0; // for poly/sigmoid

    param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    param.p = 0.0; // for CV_SVM_EPS_SVR

    param.class_weights = NULL; // for CV_SVM_C_SVC
    param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;

    
    //trainingData = trainingData.reshape(1, 1000);
    CvSVM svm;
    svm.train(trainingData, trainingClasses, Mat(), Mat(), param);
    svm.save("FILES/learn.txt"); // saving
    svm.load("FILES/learn.txt"); // loading

    int x = svm.predict(testData);
    if (x == 1) printf("Predicted disease is %s", "Anthracnose\n");
    if (x == -1) printf("Predicted disease is %s", "GallMidge\n");

    trainingData.release();
    testData.release();
    trainingClasses.release();
    }
    else cout <<"Invalid mode entered."<<endl;
}
