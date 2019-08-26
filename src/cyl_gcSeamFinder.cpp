#include <cyl_gcSeamFinder.hpp>



namespace cyl {
    
void PairwiseSeamFinder_LF::find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
                              std::vector<cv::UMat> &masks)
{
    if (src.size() == 0)
        return;

    images_ = src;
    sizes_.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        sizes_[i] = src[i].size();
    corners_ = corners;
    masks_ = masks;
    run();

}

void PairwiseSeamFinder_LF::run()
{
    for (size_t i = 0; i < sizes_.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < sizes_.size(); ++j)
        {
            cv::Rect roi;
            if (cv::detail::overlapRoi(corners_[i], corners_[j], sizes_[i], sizes_[j], roi))
                findInPair(i, j, roi);
        }
    }
}    
    
class GraphCutSeamFinder_LF::Impl : public PairwiseSeamFinder_LF
{
public:
    Impl(int cost_type, float terminal_cost, float bad_region_penalty)
        : cost_type_(cost_type), terminal_cost_(terminal_cost), bad_region_penalty_(bad_region_penalty) {}

    ~Impl() {}

    void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners, std::vector<cv::UMat> &masks, std::vector<std::vector<std::vector<cv::UMat>>> &images_warped_LF, const size_t& v, const size_t& u);
    void findInPair(size_t first, size_t second, cv::Rect roi);

private:
    void setGraphWeightsColor(const cv::Mat &img1, const cv::Mat &img2,
                              const cv::Mat &mask1, const cv::Mat &mask2, GCGraph<float> &graph);
    void setGraphWeightsColorGrad(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2,
                                  const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2,
                                  GCGraph<float> &graph);
    void setGraphWeightsColorEdge(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2,
                                  const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2,
                                  GCGraph<float> &graph, cv::Rect roi, const int& gap, const cv::Point& tl1);

    std::vector<cv::Mat> dx_, dy_;
    
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
    std::vector<std::vector<std::vector<cv::Mat>>> images_warped_LF_;
    size_t v_;
    size_t u_;
};

void GraphCutSeamFinder_LF::Impl::find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners, std::vector<cv::UMat> &masks, std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, const size_t& v, const size_t& u)
{
    
    v_ = v;
    u_ = u;
    images_warped_LF_.resize(images_warped_LF.size());
    for(size_t i=0;i<images_warped_LF_.size();i++)
    {
        images_warped_LF_[i].resize(images_warped_LF[i].size());
        for(size_t j=0;j<images_warped_LF_[i].size();j++)
        {
            images_warped_LF_[i][j].resize(images_warped_LF[i][j].size());
            images_warped_LF[i][j][0].convertTo(images_warped_LF_[i][j][0], CV_32F);
            images_warped_LF[i][j][1].convertTo(images_warped_LF_[i][j][1], CV_32F);
        }
    }
    
    std::cout<<"v_="<<v_<<std::endl;
    std::cout<<"u_="<<u_<<std::endl;
    // Compute gradients
    dx_.resize(src.size());
    dy_.resize(src.size());
    cv::Mat dx, dy;
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].channels() == 3);
        Sobel(src[i], dx, CV_32F, 1, 0);
        Sobel(src[i], dy, CV_32F, 0, 1);
        dx_[i].create(src[i].size(), CV_32F);
        dy_[i].create(src[i].size(), CV_32F);
        for (int y = 0; y < src[i].rows; ++y)
        {
            const cv::Point3f* dx_row = dx.ptr<cv::Point3f>(y);
            const cv::Point3f* dy_row = dy.ptr<cv::Point3f>(y);
            float* dx_row_ = dx_[i].ptr<float>(y);
            float* dy_row_ = dy_[i].ptr<float>(y);
            for (int x = 0; x < src[i].cols; ++x)
            {
                dx_row_[x] = cv::detail::normL2(dx_row[x]);
                dy_row_[x] = cv::detail::normL2(dy_row[x]);
            }
        }
    }
    PairwiseSeamFinder_LF::find(src, corners, masks);
}


void GraphCutSeamFinder_LF::Impl::setGraphWeightsColor(const cv::Mat &img1, const cv::Mat &img2,
                                                    const cv::Mat &mask1, const cv::Mat &mask2, GCGraph<float> &graph)
{
    const cv::Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                float weight = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x)) +
                               cv::detail::normL2(img1.at<cv::Point3f>(y, x + 1), img2.at<cv::Point3f>(y, x + 1)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float weight = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x)) +
                               cv::detail::normL2(img1.at<cv::Point3f>(y + 1, x), img2.at<cv::Point3f>(y + 1, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}


void GraphCutSeamFinder_LF::Impl::setGraphWeightsColorGrad(
        const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2,
        const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2,
        GCGraph<float> &graph)
{
    const cv::Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                float grad = dx1.at<float>(y, x) + dx1.at<float>(y, x + 1) +
                             dx2.at<float>(y, x) + dx2.at<float>(y, x + 1) + weight_eps;
                float weight = (cv::detail::normL2(img1.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x)) +
                                cv::detail::normL2(img1.at<cv::Point3f>(y, x + 1), img2.at<cv::Point3f>(y, x + 1))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float grad = dy1.at<float>(y, x) + dy1.at<float>(y + 1, x) +
                             dy2.at<float>(y, x) + dy2.at<float>(y + 1, x) + weight_eps;
                float weight = (cv::detail::normL2(img1.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x)) +
                                cv::detail::normL2(img1.at<cv::Point3f>(y + 1, x), img2.at<cv::Point3f>(y + 1, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}

static inline
float normL2_f(const float& a)
{
    return a*a;
}

static inline
float normL2_f(const float& a, const float& b)
{
    return normL2_f(a - b);
}

void GraphCutSeamFinder_LF::Impl::setGraphWeightsColorEdge(
        const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2,
        const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2,
        GCGraph<float> &graph, cv::Rect roi, const int& gap, const cv::Point& tl1)
{
    //cv::imshow("images_warped_LF_[v_][u_+1][0]",images_warped_LF_[v_][u_+1][0]);
    const cv::Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }
    
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                //float grad = normL2_f(dx1.at<float>(y, x), dx2.at<float>(y, x)) +
                //             normL2_f(dx1.at<float>(y, x + 1), dx2.at<float>(y, x + 1));
                float grad = dx1.at<float>(y, x) + dx1.at<float>(y, x + 1) +
                             dx2.at<float>(y, x) + dx2.at<float>(y, x + 1);
                float edge = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img1.at<cv::Point3f>(y, x + 1)) +
                             cv::detail::normL2(img2.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x + 1));
                //float edge = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img1.at<cv::Point3f>(y, x + 1));
                //float grad = dx1.at<float>(y, x) + dx2.at<float>(y, x);
                float color = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x)) +
                                cv::detail::normL2(img1.at<cv::Point3f>(y, x + 1), img2.at<cv::Point3f>(y, x + 1));
                float LF = 0;
                float edge_LF = 0;
                for(size_t v=0;v<images_warped_LF_.size();v++)
                    for(size_t u=0;u<images_warped_LF_[0].size();u++)
                    {
                        if(v==v_ && u==u_)continue;
                        if(roi.y-tl1.y+y-gap >= 0 && roi.y-tl1.y+y-gap <= MIN(images_warped_LF_[v][u][0].rows,images_warped_LF_[v][u][1].rows)-1 && roi.x-tl1.x+x-gap >= 0 && roi.x-tl1.x+x-gap <= MIN(images_warped_LF_[v][u][0].cols,images_warped_LF_[v][u][1].cols) -2 )
                        {
                            LF += cv::detail::normL2(images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap)) +
                            cv::detail::normL2(images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x+1-gap),images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x+1-gap));
                            
                            edge_LF += cv::detail::normL2(images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x+1-gap)) +
                            cv::detail::normL2(images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x+1-gap));
                        }
                          
                    }
                //float weight = 0.2*color + 0.8*LF/63 + weight_eps;
                float weight = 0.17*(color*color/(grad+weight_eps)) + 0.83*LF/(images_warped_LF_.size()*images_warped_LF_[0].size()-1) + weight_eps;
               
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float grad = dy1.at<float>(y, x) + dy1.at<float>(y + 1, x) +
                             dy2.at<float>(y, x) + dy2.at<float>(y + 1, x);
                //float grad = normL2_f(dy1.at<float>(y, x), dy2.at<float>(y, x)) +
                //             normL2_f(dy1.at<float>(y + 1, x), dy2.at<float>(y + 1, x));
                float edge = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img1.at<cv::Point3f>(y + 1, x)) + 
                             cv::detail::normL2(img2.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y + 1, x));
                //float edge = cv::detail::normL2(img1.at<cv::Point3f>(y, x), img1.at<cv::Point3f>(y + 1, x));         
                //float grad = dy1.at<float>(y, x) + dy2.at<float>(y, x);
                float color = (cv::detail::normL2(img1.at<cv::Point3f>(y, x), img2.at<cv::Point3f>(y, x)) +
                                cv::detail::normL2(img1.at<cv::Point3f>(y + 1, x), img2.at<cv::Point3f>(y + 1, x)));
                float LF = 0;
                float edge_LF = 0;
                for(size_t v=0;v<images_warped_LF_.size();v++)
                    for(size_t u=0;u<images_warped_LF_[0].size();u++)
                    {
                        if(v==v_ && u==u_)continue;
                        if(roi.y-tl1.y+y-gap >= 0 && roi.y-tl1.y+y-gap <= MIN(images_warped_LF_[v][u][0].rows,images_warped_LF_[v][u][1].rows)-2 && roi.x-tl1.x+x-gap >= 0 && roi.x-tl1.x+x-gap <= MIN(images_warped_LF_[v][u][0].cols,images_warped_LF_[v][u][1].cols) -1 )
                        {
                            LF += cv::detail::normL2(images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap)) +
                            cv::detail::normL2(images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y+1-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y+1-gap, roi.x-tl1.x+x-gap));
                            
                            edge_LF += cv::detail::normL2(images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][0].at<cv::Point3f>(roi.y-tl1.y+y+1-gap, roi.x-tl1.x+x-gap)) +
                            cv::detail::normL2(images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y-gap, roi.x-tl1.x+x-gap),images_warped_LF_[v][u][1].at<cv::Point3f>(roi.y-tl1.y+y+1-gap, roi.x-tl1.x+x-gap));
                        }
                    }
                //float weight = 0.2*color + 0.8*LF/63 + weight_eps;
                //float weight = 0.3*(color*color/(grad+weight_eps)) + 0.7*LF/63 + weight_eps;
                float weight = 0.17*(color*color/(grad+weight_eps)) + 0.83*LF/(images_warped_LF_.size()*images_warped_LF_[0].size()-1) + weight_eps;
                
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}


void GraphCutSeamFinder_LF::Impl::findInPair(size_t first, size_t second, cv::Rect roi)
{
    cv::Mat img1 = images_[first].getMat(cv::ACCESS_READ), img2 = images_[second].getMat(cv::ACCESS_READ);
    cv::Mat dx1 = dx_[first], dx2 = dx_[second];
    cv::Mat dy1 = dy_[first], dy2 = dy_[second];
    cv::Mat mask1 = masks_[first].getMat(cv::ACCESS_RW), mask2 = masks_[second].getMat(cv::ACCESS_RW);
    cv::Point tl1 = corners_[first], tl2 = corners_[second];

    const int gap = 10;
    cv::Mat subimg1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    cv::Mat subimg2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    cv::Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    cv::Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    cv::Mat subdx1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    cv::Mat subdy1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    cv::Mat subdx2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    cv::Mat subdy2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);

    // Cut subimages and submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.rows && x1 < img1.cols)
            {
                subimg1.at<cv::Point3f>(y + gap, x + gap) = img1.at<cv::Point3f>(y1, x1);
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
                subdx1.at<float>(y + gap, x + gap) = dx1.at<float>(y1, x1);
                subdy1.at<float>(y + gap, x + gap) = dy1.at<float>(y1, x1);
            }
            else
            {
                subimg1.at<cv::Point3f>(y + gap, x + gap) = cv::Point3f(0, 0, 0);
                submask1.at<uchar>(y + gap, x + gap) = 0;
                subdx1.at<float>(y + gap, x + gap) = 0.f;
                subdy1.at<float>(y + gap, x + gap) = 0.f;
            }

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
            {
                subimg2.at<cv::Point3f>(y + gap, x + gap) = img2.at<cv::Point3f>(y2, x2);
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
                subdx2.at<float>(y + gap, x + gap) = dx2.at<float>(y2, x2);
                subdy2.at<float>(y + gap, x + gap) = dy2.at<float>(y2, x2);
            }
            else
            {
                subimg2.at<cv::Point3f>(y + gap, x + gap) = cv::Point3f(0, 0, 0);
                submask2.at<uchar>(y + gap, x + gap) = 0;
                subdx2.at<float>(y + gap, x + gap) = 0.f;
                subdy2.at<float>(y + gap, x + gap) = 0.f;
            }
        }
    }

    const int vertex_count = (roi.height + 2 * gap) * (roi.width + 2 * gap);
    const int edge_count = (roi.height - 1 + 2 * gap) * (roi.width + 2 * gap) +
                           (roi.width - 1 + 2 * gap) * (roi.height + 2 * gap);
    GCGraph<float> graph(vertex_count, edge_count);

    switch (cost_type_)
    {
    case GraphCutSeamFinder_LF::COST_COLOR:
        setGraphWeightsColor(subimg1, subimg2, submask1, submask2, graph);
        break;
    case GraphCutSeamFinder_LF::COST_COLOR_GRAD:
        setGraphWeightsColorGrad(subimg1, subimg2, subdx1, subdx2, subdy1, subdy2,
                                 submask1, submask2, graph);
        break;
    case GraphCutSeamFinder_LF::COST_COLOR_EDGE:
        setGraphWeightsColorEdge(subimg1, subimg2, subdx1, subdx2, subdy1, subdy2,
                                 submask1, submask2, graph,roi,gap,tl1);
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "unsupported pixel similarity measure");
    }

    graph.maxFlow();

    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (graph.inSourceSegment((y + gap) * (roi.width + 2 * gap) + x + gap))
            {
                if (mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x))
                    mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            }
            else
            {
                if (mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x))
                    mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
            }
        }
    }
}


GraphCutSeamFinder_LF::GraphCutSeamFinder_LF(int cost_type, float terminal_cost, float bad_region_penalty)
    : impl_(new Impl(cost_type, terminal_cost, bad_region_penalty)) {}

GraphCutSeamFinder_LF::~GraphCutSeamFinder_LF() {}

void GraphCutSeamFinder_LF::find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
                              std::vector<cv::UMat> &masks, std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, const size_t& v, const size_t& u)
{
    impl_->find(src, corners, masks,images_warped_LF, v, u);
}

void GraphCutSeamFinder_LF::find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
                              std::vector<cv::UMat> &masks)
{
    impl_->find(src, corners, masks);
}


} // namespace cyl
