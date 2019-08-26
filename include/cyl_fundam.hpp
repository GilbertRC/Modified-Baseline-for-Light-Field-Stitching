#ifndef CYL_FUNDAM_HPP
#define CYL_FUNDAM_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
#include <precomp.hpp>
#include <cyl_types.hpp>
#include <iostream>
namespace cv
{

class HomographyEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const;
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const;
    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const;
};

class HomographyRefineCallback : public LMSolver::Callback
{
public:
    HomographyRefineCallback(InputArray _src, InputArray _dst);
    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const;
    Mat src, dst;
};

cv::Mat findHomography( InputArray _points1, InputArray _points2,
                           OutputArray _mask, int method, double ransacReprojThreshold );

void convertPointsFromHomogeneous( InputArray _src, OutputArray _dst );
void convertPointsToHomogeneous( InputArray _src, OutputArray _dst );
void convertPointsHomogeneous( InputArray _src, OutputArray _dst );
double sampsonDistance(InputArray _pt1, InputArray _pt2, InputArray _F);

}

namespace cyl
{

class HomographyRefineCallback_LF : public cv::LMSolver::Callback
{
public:
    HomographyRefineCallback_LF(cv::InputArray _src, cv::InputArray _dst);
    bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const;
    cv::Mat src, dst;
};

class HomographyRefineCallback_LF_old : public cv::LMSolver::Callback
{
public:
    HomographyRefineCallback_LF_old(cv::InputArray _src, cv::InputArray _dst);
    bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const;
    cv::Mat src, dst;
};

    
cv::Mat findHomography_re( cv::InputArray srcPoints, cv::InputArray dstPoints,
                            int method = 0, double ransacReprojThreshold = 3,
                            cv::OutputArray mask=cv::noArray(), const int maxIters = 2000,
                            const double confidence = 0.995);

int runKernel_LF( cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model );
}
    
    
    
    
#endif //CYL_TYPES_HPP
