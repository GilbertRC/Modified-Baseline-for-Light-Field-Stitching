#ifndef CYL_PROJECTION_H
#define CYL_PROJECTION_H
#include <iostream>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>


namespace cyl
{
    
const double EPSILON = 1e-6;
//柱面投影
cv::Mat Cylinder_projection(cv::Mat& input);

template<class T>
inline void bilinearInterpolate(float* result, const cv::Point2f& p, T* img, int width, int height, int channels)
{
    if (!img)
        return;
    float xx = p.x - floor(p.x);
    float yy = p.y - floor(p.y);
    int x = int(p.x), y = int(p.y);
    for(int c=0;c<channels;c++)
    {
        if( (x+1)>width-1 || (y+1)>height-1)
            result[c] = 0;
        else
        {
            result[c] = img[(y*width+x)*channels+c]*(1-xx)*(1-yy)+
                    img[(y*width+x+1)*channels+c]*xx*(1-yy)+
                    img[((y+1)*width+x)*channels+c]*(1-xx)*yy+
                    img[((y+1)*width+x+1)*channels+c]*xx*yy;
        }
    }
}


}

#endif
