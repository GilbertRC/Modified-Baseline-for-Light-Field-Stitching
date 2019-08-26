#ifndef CYL_STITCH_H
#define CYL_STITCH_H

#include <iostream>
#include <vector>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <cyl_projection.hpp>
#include <cyl_types.hpp>
#include <gms_matcher.h>
#include <cyl_fundam.hpp>
#include <cyl_gcSeamFinder.hpp>

//#define USE_GPU 

namespace cyl
{
    
void detectAndMatch(const cv::Mat& img0, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches);
void detectAndMatch_withDepth(const cv::Mat& img0, const cv::Mat& img0_d, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches);
void detectAndMatch_GMSwithDepth(const cv::Mat& img0, const cv::Mat& img0_d, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches);
void detectAndMatch_GMS(const cv::Mat& img0, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches);
bool isHomoFromMatches(cv::detail::MatchesInfo& m, const cv::detail::ImageFeatures& f1, const cv::detail::ImageFeatures& f2);
bool isHomoFromMatches(cv::detail::MatchesInfo& m, const cv::detail::ImageFeatures& f1, const cv::detail::ImageFeatures& f2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
void calcPoint_after_H(const cv::Point2f& p_src, cv::Point2f& p_dst, const cv::Mat& H);
cv::Mat WarpImg(const cv::Mat img_src, cv::Point2i& TL_rect, const cv::Mat& H);
void warpImageAndMask(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& H, std::vector<cv::UMat>& images_warped, std::vector<cv::UMat>& masks_warped, std::vector<cv::Point>& corners);
void findSeamAndBlend_LF(std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, std::vector<std::vector<std::vector<cv::UMat>>>& masks_warped_LF, std::vector<std::vector<std::vector<cv::Point>>>& corners_LF, const size_t& v, const size_t& u, cv::Mat& result, const int blend_type);
void BlendwithoutCut_LF(std::vector<cv::UMat>& images_warped, std::vector<cv::UMat>& masks_warped, std::vector<cv::Point>& corners, cv::Mat& result, const int blend_type);
void findSeamAndBlend(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& H, std::vector<cv::UMat>& images_warped, std::vector<cv::UMat>& masks_warped, std::vector<cv::Point>& corners, cv::Mat& result, const int blend_type);
void Blend(cv::Mat& result, cv::Mat& result_mask, std::vector<cv::Mat>& Ims, std::vector<cv::Mat>& Mks, std::vector<cv::Point>& corners, int blend_type);
void feature_withLFCenterConstraint(const cv::Mat& img_c0, const cv::Mat& img_0, const cv::Mat& img_1, const std::vector<cv::detail::ImageFeatures>& features_c, const std::vector<cv::detail::MatchesInfo>& matches_c, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& matches);
cv::Mat computeHomo5x5(std::vector<cv::Point4f>& points1_LF,std::vector<cv::Point4f>& points2_LF);

}

#endif
