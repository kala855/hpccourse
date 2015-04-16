#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <time.h>

int main (int argc, char **argv){
    try
    {
        cv::gpu::setDevice(0);
        clock_t start, end;
        char *imageName = argv[1];
        cv::Mat src_host = cv::imread(imageName, 1);
        cv::gpu::GpuMat dst, src, gray;
        if(argc!=2 || !src_host.data){
            std::cout<<"No image Data"<<std::endl;
            return -1;
        }

        start = clock();
        src.upload(src_host);
        cv::gpu::cvtColor(src,gray,CV_BGR2GRAY);
        cv::gpu::Sobel(gray, dst, CV_8UC1, 1,0,3, 1,0);
        cv::Mat result_host(dst);
        end = clock();
        //cv::namedWindow("Result", CV_WINDOW_NORMAL);
        //cv::imshow("Result", result_host);
        //cv::waitKey();
        cv::imwrite("./Sobel_Image.jpg",result_host);
        std::cout <<((double)(end-start))/CLOCKS_PER_SEC<<std::endl;
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
