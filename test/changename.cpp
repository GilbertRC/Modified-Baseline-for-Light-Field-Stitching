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
    int count=0;
    for(size_t v=0;v<9;v++)
        for(size_t u=0;u<9;u++)
        {
            std::string img0_name;
            std::stringstream uu,vv;
            std::string uutr,vvtr;
            uu<<(u+1);
            uu>>uutr;
            vv<<(v+1);
            vv>>vvtr;
            img0_name = "/home/richardson/桌面/result_0430_0490(7)_9x9/"+ vvtr + "_" + uutr + ".bmp";
            cv::Mat img_0 = cv::imread(img0_name,CV_LOAD_IMAGE_COLOR);
            
            std::string result_name;
            std::stringstream r_uv;
            std::string r_uvtr;
            r_uv<<count;
            r_uv>>r_uvtr;
            result_name = "/media/richardson/Richardson/陈亦雷/硕士/光场/开源/epinet-master/eslf/result_0430_0490(7)_9x9/input_Cam"+ r_uvtr + ".png";
            
            cv::imwrite(result_name,img_0);
            count++;
        }
        
        return 0;
}
