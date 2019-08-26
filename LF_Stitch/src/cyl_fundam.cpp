/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <cyl_fundam.hpp>


namespace cv
{
    bool HomographyEstimatorCallback::checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        if( haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count) )
            return false;
        
        // We check whether the minimal set of points for the homography estimation
        // are geometrically consistent. We check if every 3 correspondences sets
        // fulfills the constraint.
        //
        // The usefullness of this constraint is explained in the paper:
        //
        // "Speeding-up homography estimation in mobile devices"
        // Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
        // Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
        if( count == 4 )
        {
            static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
            const Point2f* src = ms1.ptr<Point2f>();
            const Point2f* dst = ms2.ptr<Point2f>();
            int negative = 0;
            
            for( int i = 0; i < 4; i++ )
            {
                const int* t = tt[i];
                Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
                Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);
                
                negative += determinant(A)*determinant(B) < 0;
            }
            if( negative != 0 && negative != 4 )
                return false;
        }
        
        return true;
    }
    
    int HomographyEstimatorCallback::runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(2);
        const Point2f* M = m1.ptr<Point2f>();
        const Point2f* m = m2.ptr<Point2f>();
        
        double LtL[9][9], W[9][1], V[9][9];
        Mat _LtL( 9, 9, CV_64F, &LtL[0][0] );
        Mat matW( 9, 1, CV_64F, W );
        Mat matV( 9, 9, CV_64F, V );
        Mat _H0( 3, 3, CV_64F, V[8] );
        Mat _Htemp( 3, 3, CV_64F, V[7] );
        Point2d cM(0,0), cm(0,0), sM(0,0), sm(0,0);
        
        for( i = 0; i < count; i++ )
        {
            cm.x += m[i].x; cm.y += m[i].y;
            cM.x += M[i].x; cM.y += M[i].y;
        }
        
        cm.x /= count;
        cm.y /= count;
        cM.x /= count;
        cM.y /= count;
        
        for( i = 0; i < count; i++ )
        {
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }
        
        if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
            return 0;
        sm.x = count/sm.x; sm.y = count/sm.y;
        sM.x = count/sM.x; sM.y = count/sM.y;
        
        double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
        double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
        Mat _invHnorm( 3, 3, CV_64FC1, invHnorm );
        Mat _Hnorm2( 3, 3, CV_64FC1, Hnorm2 );
        
        _LtL.setTo(Scalar::all(0));
        for( i = 0; i < count; i++ )
        {
            double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
            double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
            double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
            double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
            int j, k;
            for( j = 0; j < 9; j++ )
                for( k = j; k < 9; k++ )
                    LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
        }
        completeSymm( _LtL );
        
        eigen( _LtL, matW, matV );
        _Htemp = _invHnorm*_H0;
        _H0 = _Htemp*_Hnorm2;
        _H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(2,2) );
        
        return 1;
    }
    
    void HomographyEstimatorCallback::computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        int i, count = m1.checkVector(2);
        const Point2f* M = m1.ptr<Point2f>();
        const Point2f* m = m2.ptr<Point2f>();
        const double* H = model.ptr<double>();
        float Hf[] = { (float)H[0], (float)H[1], (float)H[2], (float)H[3], (float)H[4], (float)H[5], (float)H[6], (float)H[7] };
        
        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();
        
        for( i = 0; i < count; i++ )
        {
            float ww = 1.f/(Hf[6]*M[i].x + Hf[7]*M[i].y + 1.f);
            float dx = (Hf[0]*M[i].x + Hf[1]*M[i].y + Hf[2])*ww - m[i].x;
            float dy = (Hf[3]*M[i].x + Hf[4]*M[i].y + Hf[5])*ww - m[i].y;
            err[i] = (float)(dx*dx + dy*dy);
        }
    }



    HomographyRefineCallback::HomographyRefineCallback(InputArray _src, InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool HomographyRefineCallback::compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
    {
        int i, count = src.checkVector(2);
        Mat param = _param.getMat();
        _err.create(count*2, 1, CV_64F);
        Mat err = _err.getMat(), J;
        if( _Jac.needed())
        {
            _Jac.create(count*2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert( J.isContinuous() && J.cols == 8 );
        }

        const Point2f* M = src.ptr<Point2f>();
        const Point2f* m = dst.ptr<Point2f>();
        const double* h = param.ptr<double>();
        double* errptr = err.ptr<double>();
        double* Jptr = J.data ? J.ptr<double>() : 0;

        for( i = 0; i < count; i++ )
        {
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            errptr[i*2] = xi - m[i].x;
            errptr[i*2+1] = yi - m[i].y;

            if( Jptr )
            {
                Jptr[0] = Mx*ww; Jptr[1] = My*ww; Jptr[2] = ww;
                Jptr[3] = Jptr[4] = Jptr[5] = 0.;
                Jptr[6] = -Mx*ww*xi; Jptr[7] = -My*ww*xi;
                Jptr[8] = Jptr[9] = Jptr[10] = 0.;
                Jptr[11] = Mx*ww; Jptr[12] = My*ww; Jptr[13] = ww;
                Jptr[14] = -Mx*ww*yi; Jptr[15] = -My*ww*yi;

                Jptr += 16;
            }
        }

        return true;
    }


}
namespace cyl
{

    
HomographyRefineCallback_LF::HomographyRefineCallback_LF(cv::InputArray _src, cv::InputArray _dst)
{
    src = _src.getMat();
    dst = _dst.getMat();
}
/*
bool HomographyRefineCallback_LF::compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const
{
    int i, count = src.checkVector(4);
    cv::Mat param = _param.getMat();
    _err.create(count*4, 1, CV_64F);
    cv::Mat err = _err.getMat(), J;
    if( _Jac.needed())
    {
        _Jac.create(count*4, param.rows, CV_64F);
        J = _Jac.getMat();
        CV_Assert( J.isContinuous() && J.cols == 24 );
    }
    
    const cv::Point4f* M = src.ptr<cv::Point4f>();
    const cv::Point4f* m = dst.ptr<cv::Point4f>();
    const double* h = param.ptr<double>();
    double* errptr = err.ptr<double>();
    double* Jptr = J.data ? J.ptr<double>() : 0;
    
    for( i = 0; i < count; i++ )
    {
        double Mu = M[i].u, Mv = M[i].v;
        double Mx = M[i].x, My = M[i].y;
        double ww = h[20]*Mu + h[21]*Mv + h[22]*Mx + h[23]*My + 1.;
        ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
        double ui = (h[0]*Mu + h[1]*Mv + h[2]*Mx + h[3]*My + h[4])*ww;
        double vi = (h[5]*Mu + h[6]*Mv + h[7]*Mx + h[8]*My + h[9])*ww;
        double xi = (h[10]*Mu + h[11]*Mv + h[12]*Mx + h[13]*My + h[14])*ww;
        double yi = (h[15]*Mu + h[16]*Mv + h[17]*Mx + h[18]*My + h[19])*ww;
        errptr[i*4] = ui - m[i].u;
        errptr[i*4+1] = vi - m[i].v;
        errptr[i*4+2] = xi - m[i].x;
        errptr[i*4+3] = yi - m[i].y;
        
        if( Jptr )
        {
            Jptr[0] = Mu*ww; Jptr[1] = Mv*ww; Jptr[2] = Mx*ww; Jptr[3] = My*ww; Jptr[4] = ww;
            Jptr[5] = Jptr[6] = Jptr[7] = Jptr[8] = Jptr[9] =0.;
            Jptr[10] = Jptr[11] = Jptr[12] = Jptr[13] = Jptr[14] = 0.;
            Jptr[15] = Jptr[16] = Jptr[17] = Jptr[18] = Jptr[19] = 0.;
            Jptr[20] = -Mu*ww*ui; Jptr[21] = -Mv*ww*ui; Jptr[22] = -Mx*ww*ui; Jptr[23] = -My*ww*ui;
            Jptr[24] = Jptr[25] = Jptr[26] = Jptr[27] = Jptr[28] = 0.;
            Jptr[29] = Mu*ww; Jptr[30] = Mv*ww; Jptr[31] = Mx*ww; Jptr[32] = My*ww; Jptr[33] = ww;
            Jptr[34] = Jptr[35] = Jptr[36] = Jptr[37] = Jptr[38] = 0.;
            Jptr[39] = Jptr[40] = Jptr[41] = Jptr[42] = Jptr[43] = 0.;
            Jptr[44] = -Mu*ww*vi; Jptr[45] = -Mv*ww*vi; Jptr[46] = -Mx*ww*vi; Jptr[47] = -My*ww*vi;
            Jptr[48] = Jptr[49] = Jptr[50] = Jptr[51] = Jptr[52] = 0.;
            Jptr[53] = Jptr[54] = Jptr[55] = Jptr[56] = Jptr[57] = 0.;
            Jptr[58] = Mu*ww; Jptr[59] = Mv*ww; Jptr[60] = Mx*ww; Jptr[61] = My*ww; Jptr[62] = ww;
            Jptr[63] = Jptr[64] = Jptr[65] = Jptr[66] = Jptr[67] = 0.;
            Jptr[68] = -Mu*ww*xi; Jptr[69] = -Mv*ww*xi; Jptr[70] = -Mx*ww*xi; Jptr[71] = -My*ww*xi;
            Jptr[72] = Jptr[73] = Jptr[74] = Jptr[75] = Jptr[76] = 0.;
            Jptr[77] = Jptr[78] = Jptr[79] = Jptr[80] = Jptr[81] = 0.;
            Jptr[82] = Jptr[83] = Jptr[84] = Jptr[85] = Jptr[86] = 0.;
            Jptr[87] = Mu*ww; Jptr[88] = Mv*ww; Jptr[89] = Mx*ww; Jptr[90] = My*ww; Jptr[91] = ww;
            Jptr[92] = -Mu*ww*yi; Jptr[93] = -Mv*ww*yi; Jptr[94] = -Mx*ww*yi; Jptr[95] = -My*ww*yi;
            
            Jptr += 96;
        }
    }
    return true;
}
*/

bool HomographyRefineCallback_LF::compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const
{
    int i, count = src.checkVector(4);
    cv::Mat param = _param.getMat();
    _err.create(count*2, 1, CV_64F);
    cv::Mat err = _err.getMat(), J;
    if( _Jac.needed())
    {
        _Jac.create(count*2, param.rows, CV_64F);
        J = _Jac.getMat();
        CV_Assert( J.isContinuous() && J.cols == 24 );
    }
    
    const cv::Point4f* M = src.ptr<cv::Point4f>();
    const cv::Point4f* m = dst.ptr<cv::Point4f>();
    const double* h = param.ptr<double>();
    double* errptr = err.ptr<double>();
    double* Jptr = J.data ? J.ptr<double>() : 0;
    
    for( i = 0; i < count; i++ )
    {
        double Mu = M[i].u, Mv = M[i].v;
        double Mx = M[i].x, My = M[i].y;
        double ww = h[20]*Mu + h[21]*Mv + h[22]*Mx + h[23]*My + 1.;
        ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
        //double ui = (h[0]*Mu + h[1]*Mv + h[2]*Mx + h[3]*My + h[4])*ww;
        //double vi = (h[5]*Mu + h[6]*Mv + h[7]*Mx + h[8]*My + h[9])*ww;
        double xi = (h[10]*Mu + h[11]*Mv + h[12]*Mx + h[13]*My + h[14])*ww;
        double yi = (h[15]*Mu + h[16]*Mv + h[17]*Mx + h[18]*My + h[19])*ww;
        errptr[i*2] = xi - m[i].x;
        errptr[i*2+1] = yi - m[i].y;
        
        if( Jptr )
        {
            Jptr[0] = Jptr[1] = Jptr[2] = Jptr[3] = Jptr[4] = 0.;
            Jptr[5] = Jptr[6] = Jptr[7] = Jptr[8] = Jptr[9] = 0.;
            Jptr[10] = Mu*ww; Jptr[11] = Mv*ww; Jptr[12] = Mx*ww; Jptr[13] = My*ww; Jptr[14] = ww;
            Jptr[15] = Jptr[16] = Jptr[17] = Jptr[18] = Jptr[19] = 0.;
            Jptr[20] = -Mu*ww*xi; Jptr[21] = -Mv*ww*xi; Jptr[22] = -Mx*ww*xi; Jptr[23] = -My*ww*xi;
            Jptr[24] = Jptr[25] = Jptr[26] = Jptr[27] = Jptr[28] = 0.;
            Jptr[29] = Jptr[30] = Jptr[31] = Jptr[32] = Jptr[33] = 0.;
            Jptr[34] = Jptr[35] = Jptr[36] = Jptr[37] = Jptr[38] = 0.;
            Jptr[39] = Mu*ww; Jptr[40] = Mv*ww; Jptr[41] = Mx*ww; Jptr[42] = My*ww; Jptr[43] = ww;
            Jptr[44] = -Mu*ww*yi; Jptr[45] = -Mv*ww*yi; Jptr[46] = -Mx*ww*yi; Jptr[47] = -My*ww*yi;
            
            Jptr += 48;
        }
    }
    return true;
}

cv::Mat findHomography_re( cv::InputArray srcPoints, cv::InputArray dstPoints,
                                 int method, double ransacReprojThreshold,
                                 cv::OutputArray mask, const int maxIters,
                                 const double confidence)
{
    //CV_INSTRUMENT_REGION()

    const double defaultRANSACReprojThreshold = 3;
    bool result = false;

    cv::Mat points1 = srcPoints.getMat(), points2 = dstPoints.getMat();
    std::cout<<points1<<std::endl;
    std::cout<<points2<<std::endl;
    cv::Mat src, dst, H, tempMask;
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        cv::Mat& p = i == 1 ? points1 : points2;
        cv::Mat& m = i == 1 ? src : dst;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            if( npoints < 0 )
                CV_Error(cv::Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if( npoints == 0 )
                return cv::Mat();
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
        //std::cout<<m<<std::endl;
    }

    CV_Assert( src.checkVector(2) == dst.checkVector(2) );

    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    cv::Ptr<cv::PointSetRegistrator::Callback> cb = cv::makePtr<cv::HomographyEstimatorCallback>();

    if( method == 0 || npoints == 4 )
    {
        tempMask = cv::Mat::ones(npoints, 1, CV_8U);
        result = cb->runKernel(src, dst, H) > 0;
    }
    else if( method == cv::RANSAC )
        result = createRANSACPointSetRegistrator(cb, 4, ransacReprojThreshold, confidence, maxIters)->run(src, dst, H, tempMask);
    else if( method == cv::LMEDS )
        result = createLMeDSPointSetRegistrator(cb, 4, confidence, maxIters)->run(src, dst, H, tempMask);
    //else if( method == cv::RHO )
        //result = createAndRunRHORegistrator(confidence, maxIters, ransacReprojThreshold, npoints, src, dst, H, tempMask);
    else
        CV_Error(cv::Error::StsBadArg, "Unknown estimation method");

    if( result && npoints > 4 && method != cv::RHO)
    {
        compressElems( src.ptr<cv::Point2f>(), tempMask.ptr<uchar>(), 1, npoints );
        npoints = compressElems( dst.ptr<cv::Point2f>(), tempMask.ptr<uchar>(), 1, npoints );
        if( npoints > 0 )
        {
            cv::Mat src1 = src.rowRange(0, npoints);
            cv::Mat dst1 = dst.rowRange(0, npoints);
            src = src1;
            dst = dst1;
            std::cout<<"最小二乘前"<<H<<std::endl;
            if( method == cv::RANSAC || method == cv::LMEDS )
                std::cout<<"aa="<<npoints<<std::endl;
                cb->runKernel( src, dst, H );
            std::cout<<"优化前"<<H<<std::endl;
            cv::Mat H8(8, 1, CV_64F, H.ptr<double>());
            createLMSolver(cv::makePtr<cv::HomographyRefineCallback>(src, dst), 10)->run(H8);
        }
    }

    if( result )
    {
        if( mask.needed() )
            tempMask.copyTo(mask);
    }
    else
    {
        H.release();
        if(mask.needed() ) {
            tempMask = cv::Mat::zeros(npoints >= 0 ? npoints : 0, 1, CV_8U);
            tempMask.copyTo(mask);
        }
    }

    return H;
}

int runKernel_LF( cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model )
    {
        cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(4);
        std::cout<<"count="<<count<<std::endl;
        const cv::Point4f* M = m1.ptr<cv::Point4f>();
        const cv::Point4f* m = m2.ptr<cv::Point4f>();
        
        double LtL[25][25], W[25][1], V[25][25];
        cv::Mat _LtL( 25, 25, CV_64F, &LtL[0][0] );
        cv::Mat matW( 25, 1, CV_64F, W );
        cv::Mat matV( 25, 25, CV_64F, V );
        cv::Mat _H0( 5, 5, CV_64F, V[24] );
        cv::Mat _Htemp( 5, 5, CV_64F, V[23] );
        cv::Point4d cM(0,0,0,0), cm(0,0,0,0), sM(0,0,0,0), sm(0,0,0,0);
        
        for( i = 0; i < count; i++ )
        {
            cm.u += m[i].u; cm.v += m[i].v;
            cm.x += m[i].x; cm.y += m[i].y;
            cM.u += M[i].u; cM.v += M[i].v;
            cM.x += M[i].x; cM.y += M[i].y;
        }
        
        cm.u /= count;
        cm.v /= count;
        cm.x /= count;
        cm.y /= count;
        cM.u /= count;
        cM.v /= count;
        cM.x /= count;
        cM.y /= count;
        
        for( i = 0; i < count; i++ )
        {
            sm.u += fabs(m[i].u - cm.u);
            sm.v += fabs(m[i].v - cm.v);
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.u += fabs(M[i].u - cM.u);
            sM.v += fabs(M[i].v - cM.v);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }
        
        if( fabs(sm.u) < DBL_EPSILON || fabs(sm.v) < DBL_EPSILON ||
            fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.u) < DBL_EPSILON || fabs(sM.v) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
            return 0;
        sm.u = count/sm.u; sm.v = count/sm.v;
        sm.x = count/sm.x; sm.y = count/sm.y;
        sM.u = count/sM.u; sM.v = count/sM.v;
        sM.x = count/sM.x; sM.y = count/sM.y;
        
        double invHnorm[25] = { 1./sm.u, 0, 0, 0, cm.u, 0, 1./sm.v, 0, 0, cm.v, 0, 0, 1./sm.x, 0, cm.x, 0, 0, 0, 1./sm.y, cm.y, 0, 0, 0, 0, 1 };
        double Hnorm2[25] = { sM.u, 0, 0, 0, -cM.u*sM.u, 0, sM.v, 0, 0, -cM.v*sM.v, 0, 0, sM.x, 0, -cM.x*sM.x, 0, 0, 0, sM.y, -cM.y*sM.y, 0, 0, 0, 0, 1 };
        cv::Mat _invHnorm( 5, 5, CV_64FC1, invHnorm );
        cv::Mat _Hnorm2( 5, 5, CV_64FC1, Hnorm2 );
        
        _LtL.setTo(cv::Scalar::all(0));
        for( i = 0; i < count; i++ )
        {
            double u = (m[i].u - cm.u)*sm.u, v = (m[i].v - cm.v)*sm.v;
            double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
            double U = (M[i].u - cM.u)*sM.u, V = (M[i].v - cM.v)*sM.v;
            double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
            double Lu[] = { U, V, X, Y, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -u*U, -u*V, -u*X, -u*Y, -u };
            double Lv[] = { 0, 0, 0, 0, 0, U, V, X, Y, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -v*U, -v*V, -v*X, -v*Y, -v };
            double Lx[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, U, V, X, Y, 1, 0, 0, 0, 0, 0, -x*U, -x*V, -x*X, -x*Y, -x };
            double Ly[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, U, V, X, Y, 1, -y*U, -y*V, -y*X, -y*Y, -y };
            int j, k;
            for( j = 0; j < 25; j++ )
                for( k = j; k < 25; k++ )
                    LtL[j][k] += Lu[j]*Lu[k] + Lv[j]*Lv[k] + Lx[j]*Lx[k] + Ly[j]*Ly[k];
        }
        completeSymm( _LtL );
        
        eigen( _LtL, matW, matV );
        _Htemp = _invHnorm*_H0;
        _H0 = _Htemp*_Hnorm2;
        
        _H0=_H0/_H0.at<double>(4,4);
        std::cout<<"_H0="<<_H0<<std::endl;
        _H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(4,4));
        
        return 1;
    }

    
HomographyRefineCallback_LF_old::HomographyRefineCallback_LF_old(cv::InputArray _src, cv::InputArray _dst)
{
    src = _src.getMat();
    dst = _dst.getMat();
}

bool HomographyRefineCallback_LF_old::compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const
{
    int i, count = src.checkVector(4);
    cv::Mat param = _param.getMat();
    _err.create(count*4, 1, CV_64F);
    cv::Mat err = _err.getMat(), J;
    if( _Jac.needed())
    {
        _Jac.create(count*4, param.rows, CV_64F);
        J = _Jac.getMat();
        CV_Assert( J.isContinuous() && J.cols == 24 );
    }
    
    const cv::Point4f* M = src.ptr<cv::Point4f>();
    const cv::Point4f* m = dst.ptr<cv::Point4f>();
    const double* h = param.ptr<double>();
    double* errptr = err.ptr<double>();
    double* Jptr = J.data ? J.ptr<double>() : 0;
    
    for( i = 0; i < count; i++ )
    {
        double Mu = M[i].u, Mv = M[i].v;
        double Mx = M[i].x, My = M[i].y;
        double ww = h[20]*Mu + h[21]*Mv + h[22]*Mx + h[23]*My + 1.;
        ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
        double ui = (h[0]*Mu + h[1]*Mv + h[2]*Mx + h[3]*My + h[4])*ww;
        double vi = (h[5]*Mu + h[6]*Mv + h[7]*Mx + h[8]*My + h[9])*ww;
        double xi = (h[10]*Mu + h[11]*Mv + h[12]*Mx + h[13]*My + h[14])*ww;
        double yi = (h[15]*Mu + h[16]*Mv + h[17]*Mx + h[18]*My + h[19])*ww;
        errptr[i*4] = ui - m[i].u;
        errptr[i*4+1] = vi - m[i].v;
        errptr[i*4+2] = xi - m[i].x;
        errptr[i*4+3] = yi - m[i].y;
        
        if( Jptr )
        {
            Jptr[0] = Mu*ww; Jptr[1] = Mv*ww; Jptr[2] = Mx*ww; Jptr[3] = My*ww; Jptr[4] = ww;
            Jptr[5] = Jptr[6] = Jptr[7] = Jptr[8] = Jptr[9] =0.;
            Jptr[10] = Jptr[11] = Jptr[12] = Jptr[13] = Jptr[14] = 0.;
            Jptr[15] = Jptr[16] = Jptr[17] = Jptr[18] = Jptr[19] = 0.;
            Jptr[20] = -Mu*ww*ui; Jptr[21] = -Mv*ww*ui; Jptr[22] = -Mx*ww*ui; Jptr[23] = -My*ww*ui;
            Jptr[24] = Jptr[25] = Jptr[26] = Jptr[27] = Jptr[28] = 0.;
            Jptr[29] = Mu*ww; Jptr[30] = Mv*ww; Jptr[31] = Mx*ww; Jptr[32] = My*ww; Jptr[33] = ww;
            Jptr[34] = Jptr[35] = Jptr[36] = Jptr[37] = Jptr[38] = 0.;
            Jptr[39] = Jptr[40] = Jptr[41] = Jptr[42] = Jptr[43] = 0.;
            Jptr[44] = -Mu*ww*vi; Jptr[45] = -Mv*ww*vi; Jptr[46] = -Mx*ww*vi; Jptr[47] = -My*ww*vi;
            Jptr[48] = Jptr[49] = Jptr[50] = Jptr[51] = Jptr[52] = 0.;
            Jptr[53] = Jptr[54] = Jptr[55] = Jptr[56] = Jptr[57] = 0.;
            Jptr[58] = Mu*ww; Jptr[59] = Mv*ww; Jptr[60] = Mx*ww; Jptr[61] = My*ww; Jptr[62] = ww;
            Jptr[63] = Jptr[64] = Jptr[65] = Jptr[66] = Jptr[67] = 0.;
            Jptr[68] = -Mu*ww*xi; Jptr[69] = -Mv*ww*xi; Jptr[70] = -Mx*ww*xi; Jptr[71] = -My*ww*xi;
            Jptr[72] = Jptr[73] = Jptr[74] = Jptr[75] = Jptr[76] = 0.;
            Jptr[77] = Jptr[78] = Jptr[79] = Jptr[80] = Jptr[81] = 0.;
            Jptr[82] = Jptr[83] = Jptr[84] = Jptr[85] = Jptr[86] = 0.;
            Jptr[87] = Mu*ww; Jptr[88] = Mv*ww; Jptr[89] = Mx*ww; Jptr[90] = My*ww; Jptr[91] = ww;
            Jptr[92] = -Mu*ww*yi; Jptr[93] = -Mv*ww*yi; Jptr[94] = -Mx*ww*yi; Jptr[95] = -My*ww*yi;
            
            Jptr += 96;
        }
    }
    return true;
}


}

cv::Mat cv::findHomography( InputArray _points1, InputArray _points2,
                           OutputArray _mask, int method, double ransacReprojThreshold )
{
    return cv::findHomography(_points1, _points2, method, ransacReprojThreshold, _mask);
}


void cv::convertPointsFromHomogeneous( InputArray _src, OutputArray _dst )
{
    //CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    if( !src.isContinuous() )
        src = src.clone();
    int i, npoints = src.checkVector(3), depth = src.depth(), cn = 3;
    if( npoints < 0 )
    {
        npoints = src.checkVector(4);
        CV_Assert(npoints >= 0);
        cn = 4;
    }
    CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F || depth == CV_64F));

    int dtype = CV_MAKETYPE(depth <= CV_32F ? CV_32F : CV_64F, cn-1);
    _dst.create(npoints, 1, dtype);
    Mat dst = _dst.getMat();
    if( !dst.isContinuous() )
    {
        _dst.release();
        _dst.create(npoints, 1, dtype);
        dst = _dst.getMat();
    }
    CV_Assert( dst.isContinuous() );

    if( depth == CV_32S )
    {
        if( cn == 3 )
        {
            const Point3i* sptr = src.ptr<Point3i>();
            Point2f* dptr = dst.ptr<Point2f>();
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i].z != 0 ? 1.f/sptr[i].z : 1.f;
                dptr[i] = Point2f(sptr[i].x*scale, sptr[i].y*scale);
            }
        }
        else
        {
            const Vec4i* sptr = src.ptr<Vec4i>();
            Point3f* dptr = dst.ptr<Point3f>();
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i][3] != 0 ? 1.f/sptr[i][3] : 1.f;
                dptr[i] = Point3f(sptr[i][0]*scale, sptr[i][1]*scale, sptr[i][2]*scale);
            }
        }
    }
    else if( depth == CV_32F )
    {
        if( cn == 3 )
        {
            const Point3f* sptr = src.ptr<Point3f>();
            Point2f* dptr = dst.ptr<Point2f>();
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i].z != 0.f ? 1.f/sptr[i].z : 1.f;
                dptr[i] = Point2f(sptr[i].x*scale, sptr[i].y*scale);
            }
        }
        else
        {
            const Vec4f* sptr = src.ptr<Vec4f>();
            Point3f* dptr = dst.ptr<Point3f>();
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i][3] != 0.f ? 1.f/sptr[i][3] : 1.f;
                dptr[i] = Point3f(sptr[i][0]*scale, sptr[i][1]*scale, sptr[i][2]*scale);
            }
        }
    }
    else if( depth == CV_64F )
    {
        if( cn == 3 )
        {
            const Point3d* sptr = src.ptr<Point3d>();
            Point2d* dptr = dst.ptr<Point2d>();
            for( i = 0; i < npoints; i++ )
            {
                double scale = sptr[i].z != 0. ? 1./sptr[i].z : 1.;
                dptr[i] = Point2d(sptr[i].x*scale, sptr[i].y*scale);
            }
        }
        else
        {
            const Vec4d* sptr = src.ptr<Vec4d>();
            Point3d* dptr = dst.ptr<Point3d>();
            for( i = 0; i < npoints; i++ )
            {
                double scale = sptr[i][3] != 0.f ? 1./sptr[i][3] : 1.;
                dptr[i] = Point3d(sptr[i][0]*scale, sptr[i][1]*scale, sptr[i][2]*scale);
            }
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");
}


void cv::convertPointsToHomogeneous( InputArray _src, OutputArray _dst )
{
    //CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    if( !src.isContinuous() )
        src = src.clone();
    int i, npoints = src.checkVector(2), depth = src.depth(), cn = 2;
    if( npoints < 0 )
    {
        npoints = src.checkVector(3);
        CV_Assert(npoints >= 0);
        cn = 3;
    }
    CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F || depth == CV_64F));

    int dtype = CV_MAKETYPE(depth, cn+1);
    _dst.create(npoints, 1, dtype);
    Mat dst = _dst.getMat();
    if( !dst.isContinuous() )
    {
        _dst.release();
        _dst.create(npoints, 1, dtype);
        dst = _dst.getMat();
    }
    CV_Assert( dst.isContinuous() );

    if( depth == CV_32S )
    {
        if( cn == 2 )
        {
            const Point2i* sptr = src.ptr<Point2i>();
            Point3i* dptr = dst.ptr<Point3i>();
            for( i = 0; i < npoints; i++ )
                dptr[i] = Point3i(sptr[i].x, sptr[i].y, 1);
        }
        else
        {
            const Point3i* sptr = src.ptr<Point3i>();
            Vec4i* dptr = dst.ptr<Vec4i>();
            for( i = 0; i < npoints; i++ )
                dptr[i] = Vec4i(sptr[i].x, sptr[i].y, sptr[i].z, 1);
        }
    }
    else if( depth == CV_32F )
    {
        if( cn == 2 )
        {
            const Point2f* sptr = src.ptr<Point2f>();
            Point3f* dptr = dst.ptr<Point3f>();
            for( i = 0; i < npoints; i++ )
                dptr[i] = Point3f(sptr[i].x, sptr[i].y, 1.f);
        }
        else
        {
            const Point3f* sptr = src.ptr<Point3f>();
            Vec4f* dptr = dst.ptr<Vec4f>();
            for( i = 0; i < npoints; i++ )
                dptr[i] = Vec4f(sptr[i].x, sptr[i].y, sptr[i].z, 1.f);
        }
    }
    else if( depth == CV_64F )
    {
        if( cn == 2 )
        {
            const Point2d* sptr = src.ptr<Point2d>();
            Point3d* dptr = dst.ptr<Point3d>();
            for( i = 0; i < npoints; i++ )
                dptr[i] = Point3d(sptr[i].x, sptr[i].y, 1.);
        }
        else
        {
            const Point3d* sptr = src.ptr<Point3d>();
            Vec4d* dptr = dst.ptr<Vec4d>();
            for( i = 0; i < npoints; i++ )
                dptr[i] = Vec4d(sptr[i].x, sptr[i].y, sptr[i].z, 1.);
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");
}


void cv::convertPointsHomogeneous( InputArray _src, OutputArray _dst )
{
    //CV_INSTRUMENT_REGION()

    int stype = _src.type(), dtype = _dst.type();
    CV_Assert( _dst.fixedType() );

    if( CV_MAT_CN(stype) > CV_MAT_CN(dtype) )
        convertPointsFromHomogeneous(_src, _dst);
    else
        convertPointsToHomogeneous(_src, _dst);
}

double cv::sampsonDistance(InputArray _pt1, InputArray _pt2, InputArray _F)
{
    //CV_INSTRUMENT_REGION()

    CV_Assert(_pt1.type() == CV_64F && _pt2.type() == CV_64F && _F.type() == CV_64F);
    CV_DbgAssert(_pt1.rows() == 3 && _F.size() == Size(3, 3) && _pt1.rows() == _pt2.rows());

    Mat pt1(_pt1.getMat());
    Mat pt2(_pt2.getMat());
    Mat F(_F.getMat());

    Vec3d F_pt1 = *F.ptr<Matx33d>() * *pt1.ptr<Vec3d>();
    Vec3d Ft_pt2 = F.ptr<Matx33d>()->t() * *pt2.ptr<Vec3d>();

    double v = pt2.ptr<Vec3d>()->dot(F_pt1);

    // square
    Ft_pt2 = Ft_pt2.mul(Ft_pt2);
    F_pt1 = F_pt1.mul(F_pt1);

    return v*v / (F_pt1[0] + F_pt1[1] + Ft_pt2[0] + Ft_pt2[1]);
}


/* End of file. */
