#include "machine_learning.hpp"

void rgb_to_hsi(Mat src)
{

  if(src.empty())
  cerr << "Error: Loading image" << endl;
  Mat hsi(src.rows, src.cols, src.type());

  float r, g, b, h, s, in;

  for(int i = 0; i < src.rows; i++)
    {
      for(int j = 0; j < src.cols; j++)
        {
          b = src.at<Vec3b>(i, j)[0];
          g = src.at<Vec3b>(i, j)[1];
          r = src.at<Vec3b>(i, j)[2];

          in = (b + g + r) / 3;

          int min_val = 0;
          min_val = std::min(r, std::min(b,g));

          s = 1 - 3*(min_val/(b + g + r));
          if(s < 0.00001)
            {
                  s = 0;
            }else if(s > 0.99999){
                  s = 1;
            }

          if(s != 0)
            {
              h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g)*(r - g)) + ((r - b)*(g - b)));
              h = acos(h);

              if(b <= g)
                {
                  h = h;
                } else{
                  h = ((360 * 3.14159265) / 180.0) - h;
                }
            }

          hsi.at<Vec3b>(i, j)[0] = (h * 180) / 3.14159265;
          hsi.at<Vec3b>(i, j)[1] = s*100;
          hsi.at<Vec3b>(i, j)[2] = in;
        }
    }

  //namedWindow("RGB image", CV_WINDOW_AUTOSIZE);
  //namedWindow("HSI image", CV_WINDOW_AUTOSIZE);

  //imshow("RGB image", src);
  //imshow("HSI image", hsi);


  vector<Mat> channels;
  split(hsi, channels);

  //imshow("H image", channels[0]);
  //imshow("S image", channels[1]);
  //imshow("I image", channels[2]);

  //char file[3] = "hsi";
  imwrite("PROCESS_PIC/hsi.jpg",hsi);
  imwrite("PROCESS_PIC/h.jpg",channels[0]);
  hsi.release();
  //waitKey(0);
  //return channels[0];
}
