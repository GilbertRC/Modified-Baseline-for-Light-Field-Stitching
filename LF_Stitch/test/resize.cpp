//调整输出的尺寸
//***************************************************************//
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <cyl_projection.hpp>
#include <cyl_stitch.hpp>
#include <cyl_types.hpp>

int main(int argc, char** argv)
{
    
    int result_height, result_width;
    std::vector<cv::Mat> LF_result;
    for(size_t v=5;v<6;v++)
        for(size_t u=1;u<9;u++)
        {
            std::string result_name;
            std::stringstream r_uu,r_vv;
            std::string r_uutr,r_vvtr;
            r_uu<<(u);
            r_uu>>r_uutr;
            r_vv<<(v);
            r_vv>>r_vvtr;
            result_name = "/home/richardson/桌面/result_0328_0390_NISwGSP/"+ r_vvtr + "_" + r_uutr + "-[NISwGSP][3D][BLEND_LINEAR].png";
            std::cout<<result_name<<std::endl;
            cv::Mat result = cv::imread(result_name,CV_LOAD_IMAGE_COLOR);
            
            LF_result.push_back(result);
            if(u==1) {
                result_height=result.rows; 
                result_width=result.cols; 
            }
            else{
                if(result.rows < result_height) result_height=result.rows;
                if(result.cols < result_width) result_width=result.cols;
            }
        }
        
        for(size_t v=5;v<6;v++)
            for(size_t u=1;u<9;u++)
            {
                std::string result_name;
                std::stringstream r_uu,r_vv;
                std::string r_uutr,r_vvtr;
                r_uu<<(u);
                r_uu>>r_uutr;
                r_vv<<(v);
                r_vv>>r_vvtr;
                cv::Mat result=cv::Mat::zeros(result_height, result_width, CV_8UC3);
                LF_result[u-1](cv::Rect(LF_result[u-1].cols-result_width, LF_result[u-1].rows/2-result_height/2, result_width, result_height-1)).copyTo(result);
                resize(result,result,cv::Size(result.cols*376/result.rows,376));
                result_name = "/home/richardson/桌面/result_0328_0390_NISwGSP/resize_"+ r_vvtr + "_" + r_uutr + ".bmp";
                cv::imwrite(result_name,result);
            }
            
            return 0;
}




