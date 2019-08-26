#include <cyl_projection.hpp>

namespace cyl
{
    
//双线性插值
  inline float getPixelValue(cv::Mat* image_, float x, float y, int k)
  {
    uchar* data = & image_->data[int(y)*image_->step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
      (1-xx)*(1-yy)*data[0]+
      xx*(1-yy)*data[1]+
      (1-xx)*yy*data[image_->step]+
      xx*yy*data[image_->step+1]
    );
  }
    
    
//柱面投影
cv::Mat Cylinder_projection(cv::Mat& input) 
{
    double width = input.cols;
    double height = input.rows;
    double centerX = width / 2;
    double centerY = height / 2;
    
    double alpha = M_PI / 4;//视角设为45度
    double f = width / (2 * tan(alpha/ 2));
    cv::Mat output = cv::Mat::zeros(height,f*alpha,CV_8UC3);
    
    double theta, pointX, pointY;
    for (int y = 0; y < height; y ++) {
        for (int x = 0; x < f*(M_PI/4); x ++) {
            theta = (x-f*alpha/2) / f;
            pointX = f * tan( (x - f*atan(width/(2*f))) / f) + centerX;
            pointY = (y - centerY) / cos(theta) +  centerY;
            for (int k = 0; k < input.channels(); k ++) {
                if (pointY >= 0 && pointY <= height && pointX >= 0 && pointX <= width)
                    output.at<cv::Vec3b>(y, x)[k] = input.at<cv::Vec3b>(pointY, pointX)[k];
                else 
                    output.at<cv::Vec3b>(y, x)[k] = 0;
            }
        }
    }
    return output;
}


}
