#ifndef CYL_GCSEAMFINDER_HPP
#define CYL_GCSEAMFINDER_HPP

#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util_inl.hpp>
#include <precomp.hpp>
#include <map>
#include <gcgraph.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace cyl {

class CV_EXPORTS SeamFinder_LF
{
public:
    virtual ~SeamFinder_LF() {}
    /** @brief Estimates seams.

    @param src Source images
    @param corners Source image top-left corners
    @param masks Source image masks to update
     */
    virtual void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners, std::vector<cv::UMat> &masks) = 0;
    virtual void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners, std::vector<cv::UMat> &masks, std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, const size_t& v, const size_t& u) = 0;
};

/** @brief Base class for all pairwise seam estimators.
 */
class CV_EXPORTS PairwiseSeamFinder_LF : public SeamFinder_LF
{
public:
    virtual void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
                      std::vector<cv::UMat> &masks);
    virtual void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners, std::vector<cv::UMat> &masks, std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, const size_t& v, const size_t& u) = 0;

protected:
    void run();
    /** @brief Resolves masks intersection of two specified images in the given ROI.

    @param first First image index
    @param second Second image index
    @param roi Region of interest
     */
    virtual void findInPair(size_t first, size_t second, cv::Rect roi) = 0;

    std::vector<cv::UMat> images_;
    std::vector<cv::Size> sizes_;
    std::vector<cv::Point> corners_;
    std::vector<cv::UMat> masks_;
};
    
/** @brief Base class for all minimum graph-cut-based seam estimators.
 */
class CV_EXPORTS GraphCutSeamFinderBase_LF
{
public:
    enum CostType { COST_COLOR, COST_COLOR_GRAD, COST_COLOR_EDGE };
};

/** @brief Minimum graph cut-based seam estimator. See details in @cite V03 .
 */
class CV_EXPORTS GraphCutSeamFinder_LF : public GraphCutSeamFinderBase_LF, public SeamFinder_LF
{
public:
    GraphCutSeamFinder_LF(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                       float bad_region_penalty = 1000.f);

    ~GraphCutSeamFinder_LF();

    void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
              std::vector<cv::UMat> &masks);
    void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
              std::vector<cv::UMat> &masks, std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, const size_t& v, const size_t& u);

private:
    // To avoid GCGraph dependency
    class Impl;
    cv::Ptr<PairwiseSeamFinder_LF> impl_;

};

} // namespace cyl

#endif // CYL_GCSEAMFINDER_HPP

