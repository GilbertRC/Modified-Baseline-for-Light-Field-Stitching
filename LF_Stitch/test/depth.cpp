//深度图测试，用于预处理或归一化查看
//***************************************************************//
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/highgui/highgui.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <cyl_projection.hpp>
#include <cyl_stitch.hpp>

#include <opencv2/ximgproc.hpp>


int main(int argc, char** argv)
{
    cv::Mat img_0 = cv::imread("./result_resize.bmp",CV_LOAD_IMAGE_COLOR);
    cv::Mat img_0_d = cv::imread("./result_WMF_3.bmp",CV_LOAD_IMAGE_GRAYSCALE);
    //std::cout<<img_0_d<<std::endl;
    
    cv::Mat dst;
    cv::ximgproc::jointBilateralFilter(img_0, img_0_d, dst, -1, 6, 120);
    
    imshow("src", img_0_d);
    imshow("joint", img_0);
    imshow("jointBilateralFilter", dst);
    imwrite("result_WMF_3_JBF.bmp",dst);
    
    /*
    cv::Mat img_0_resize = cv::Mat::zeros(img_0_d.rows,img_0_d.cols,CV_8UC3);
    for(size_t y=0; y<img_0_resize.rows;y++ )
        for(size_t x=0; x<img_0_resize.cols;x++ )
        {
            img_0_resize.at<cv::Vec3b>(y,x)[0]=img_0.at<cv::Vec3b>(y+11,x+11)[0];
            img_0_resize.at<cv::Vec3b>(y,x)[1]=img_0.at<cv::Vec3b>(y+11,x+11)[1];
            img_0_resize.at<cv::Vec3b>(y,x)[2]=img_0.at<cv::Vec3b>(y+11,x+11)[2];
        }
    cv::imshow("11",img_0);
    cv::imwrite("result_resize.bmp",img_0_resize);*/
    
    //cv::imshow("11_d",img_0_d);
    //cv::imshow("11_d",img_0_d);
    //cv::imwrite("1_depth_resize.bmp",img_0_d);
    /*
    cv::Mat dst;
    cv::ximgproc::jointBilateralFilter(img_0, img_0_d, dst, -1, 6, 20);
    
    imshow("src", img_0_d);
    imshow("joint", img_0);
    imshow("jointBilateralFilter", dst);
    imwrite("1_depth_new.bmp",dst);*/
    
    /*
    double MaxValue, MinValue;
    cv::minMaxLoc(img_0_d, &MinValue,&MaxValue);
    std::cout<<MinValue<<" "<<MaxValue<<std::endl;
    
    cv::Mat Normdepth=cv::Mat::zeros(img_0_d.rows,img_0_d.cols,CV_8UC1);
    for(size_t y=0; y<img_0_d.rows;y++ )
        for(size_t x=0; x<img_0_d.cols;x++ )
        {
            Normdepth.at<uchar>(y,x)=((int)img_0_d.at<uchar>(y,x)-MinValue)*255/(MaxValue-MinValue);
        }
    
    cv::imshow("111",Normdepth);
    std::cout<<Normdepth<<std::endl;
    
    cv::minMaxLoc(aaa, &MinValue,&MaxValue);
    std::cout<<MinValue<<" "<<MaxValue;
    //std::cout<<aaa<<std::endl;
    cv::Mat result=cv::Mat::zeros(img_0.rows,img_0.cols,CV_8UC3);
    for(size_t y=0; y<img_0.rows;y++ )
        for(size_t x=0; x<img_0.cols;x++ )
        {
            //if(y>img_0_resize.rows-1 || x>img_0_resize.cols-1) continue;
            if((int)Normdepth.at<uchar>(y,x)<180)
            {
                //result.at<cv::Vec3b>(y+11,x+11)[0]=img_0_resize.at<cv::Vec3b>(y,x)[0];
                //result.at<cv::Vec3b>(y+11,x+11)[1]=img_0_resize.at<cv::Vec3b>(y,x)[1];
                //result.at<cv::Vec3b>(y+11,x+11)[2]=img_0_resize.at<cv::Vec3b>(y,x)[2];
                result.at<cv::Vec3b>(y,x)[0]=img_0.at<cv::Vec3b>(y,x)[0];
                result.at<cv::Vec3b>(y,x)[1]=img_0.at<cv::Vec3b>(y,x)[1];
                result.at<cv::Vec3b>(y,x)[2]=img_0.at<cv::Vec3b>(y,x)[2];
            }
        }
    
    cv::imshow("result",result);
    //std::cout<<img_0_d<<std::endl;*/
    
    cv::waitKey(0);
    
    return 0;
}


