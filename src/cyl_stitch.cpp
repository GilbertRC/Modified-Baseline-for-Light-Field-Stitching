#include <cyl_stitch.hpp>

namespace cyl
{

void detectAndMatch(const cv::Mat& img0, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches)
{
    cv::Ptr <cv::Feature2D> sift = cv::xfeatures2d :: SIFT :: create(3200);
    
    cv::Mat gray;
    cv::cvtColor(img0, gray, CV_BGR2GRAY);
    sift->detect(img0,features[0].keypoints);
    sift->compute(img0,features[0].keypoints,features[0].descriptors);
    features[0].img_idx = 0;
    features[0].img_size = gray.size();
    cv::cvtColor(img1, gray, CV_BGR2GRAY);
    sift->detect(img1,features[1].keypoints);
    sift->compute(img1,features[1].keypoints,features[1].descriptors);
    features[1].img_idx = 1;
    features[1].img_size = gray.size();
    
    cv::detail::BestOf2NearestMatcher matcher;
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    /*
     // GMS filter
    std::vector<bool> vbInliers;
    gms_matcher gms(features[0].keypoints, img0.size(), features[1].keypoints, img1.size(), pairwise_matches[1].matches);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    
    // collect matches
    std::vector<cv::DMatch> matches_gms;
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(pairwise_matches[1].matches[i]);
        }
    }
    pairwise_matches[1].matches.swap(matches_gms);*/
}

void detectAndMatch_withDepth(const cv::Mat& img0, const cv::Mat& img0_d, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches)
{
    //深度图归一化
    double MaxValue, MinValue;
    cv::minMaxLoc(img0_d, &MinValue,&MaxValue);
    
    cv::Mat Normdepth=cv::Mat::zeros(img0_d.rows,img0_d.cols,CV_8UC1);
    for(size_t y=0; y<img0_d.rows;y++ )
        for(size_t x=0; x<img0_d.cols;x++ )
        {
            Normdepth.at<uchar>(y,x)=((int)img0_d.at<uchar>(y,x)-MinValue)*255/(MaxValue-MinValue);
        }
    
    cv::Mat mask = cv::Mat::zeros(img0.rows,img0.cols,CV_8UC1);
    for(size_t y=0; y<mask.rows;y++ )
        for(size_t x=0; x<mask.cols;x++ )
        {
            if(y-11<0 || x-11<0 ||y-11>img0_d.rows-1 || x-11>img0_d.cols-1) continue;
            if((int)Normdepth.at<uchar>(y-11,x-11)<180)
            {
                mask.at<uchar>(y,x) = 255;
            }
        }
        //cv::imshow("aaa",mask);
    
    cv::Ptr <cv::Feature2D> sift = cv::xfeatures2d :: SIFT :: create(3200);
    cv::Mat gray;
    cv::cvtColor(img0, gray, CV_BGR2GRAY);
    sift->detectAndCompute(img0, mask, features[0].keypoints, features[0].descriptors);
    features[0].img_idx = 0;
    features[0].img_size = gray.size();
    cv::cvtColor(img1, gray, CV_BGR2GRAY);
    sift->detectAndCompute(img1, cv::Mat(), features[1].keypoints, features[1].descriptors);
    features[1].img_idx = 1;
    features[1].img_size = gray.size();
    
    cv::detail::BestOf2NearestMatcher matcher;
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
}

void detectAndMatch_GMSwithDepth(const cv::Mat& img0, const cv::Mat& img0_d, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches)
{
    //深度图归一化
    double MaxValue, MinValue;
    cv::minMaxLoc(img0_d, &MinValue,&MaxValue);
    
    cv::Mat Normdepth=cv::Mat::zeros(img0_d.rows,img0_d.cols,CV_8UC1);
    for(size_t y=0; y<img0_d.rows;y++ )
        for(size_t x=0; x<img0_d.cols;x++ )
        {
            Normdepth.at<uchar>(y,x)=((int)img0_d.at<uchar>(y,x)-MinValue)*255/(MaxValue-MinValue);
        }
    
    cv::Mat mask = cv::Mat::zeros(img0.rows,img0.cols,CV_8UC1);
    for(size_t y=0; y<mask.rows;y++ )
        for(size_t x=0; x<mask.cols;x++ )
        {
            if(y-11<0 || x-11<0 ||y-11>img0_d.rows-1 || x-11>img0_d.cols-1) continue;
            if((int)Normdepth.at<uchar>(y-11,x-11)<180)
            {
                mask.at<uchar>(y,x) = 255;
            }
        }
        //cv::imshow("aaa",mask);
    
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create(10000);
    orb->setFastThreshold(0);
    orb->detectAndCompute(img0, mask, features[0].keypoints, features[0].descriptors);
    features[0].img_idx = 0;
    features[0].img_size = img0.size();
    orb->detectAndCompute(img1, cv::Mat(), features[1].keypoints, features[1].descriptors);
    features[1].img_idx = 1;
    features[1].img_size = img1.size();
#ifdef USE_GPU
    cv::cuda::GpuMat gd1(features[0].descriptors), gd2(features[1].descriptors);
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    matcher->match(gd1, gd2, pairwise_matches[1].matches);
#else
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(features[0].descriptors, features[1].descriptors, pairwise_matches[1].matches);
#endif
    
    // GMS filter
    std::vector<bool> vbInliers;
    gms_matcher gms(features[0].keypoints, img0.size(), features[1].keypoints, img1.size(), pairwise_matches[1].matches);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    
    // collect matches
    std::vector<cv::DMatch> matches_gms;
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(pairwise_matches[1].matches[i]);
        }
    }
    pairwise_matches[1].matches.swap(matches_gms);
}

void detectAndMatch_GMS(const cv::Mat& img0, const cv::Mat& img1, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(10000);
    orb->setFastThreshold(0);
    
    orb->detectAndCompute(img0, cv::Mat(), features[0].keypoints, features[0].descriptors);
    features[0].img_idx = 0;
    features[0].img_size = img0.size();
    orb->detectAndCompute(img1, cv::Mat(), features[1].keypoints, features[1].descriptors);
    features[1].img_idx = 1;
    features[1].img_size = img1.size();
#ifdef USE_GPU
    cv::cuda::GpuMat gd1(features[0].descriptors), gd2(features[1].descriptors);
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    matcher->match(gd1, gd2, pairwise_matches[1].matches);
#else
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(features[0].descriptors, features[1].descriptors, pairwise_matches[1].matches);
#endif
    
    // GMS filter
    std::vector<bool> vbInliers;
    gms_matcher gms(features[0].keypoints, img0.size(), features[1].keypoints, img1.size(), pairwise_matches[1].matches);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    
    // collect matches
    std::vector<cv::DMatch> matches_gms;
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(pairwise_matches[1].matches[i]);
        }
    }
    pairwise_matches[1].matches.swap(matches_gms);
}


bool isHomoFromMatches(cv::detail::MatchesInfo& m, const cv::detail::ImageFeatures& f1, const cv::detail::ImageFeatures& f2)
{
    // compute other elements of MatchesInfo
    if((int)m.matches.size() < 5)
    {
        m = cv::detail::MatchesInfo();
        return false;
    }
    m.src_img_idx = f1.img_idx;
    m.dst_img_idx = f2.img_idx;
    
    // calculate Homography
    std::vector<cv::Point2f> points1, points2;
    std::vector<cv::DMatch> a;
    for(int i=0;i<m.matches.size();i++)
    {
        points1.push_back(f1.keypoints[m.matches[i].queryIdx].pt);
        points2.push_back(f2.keypoints[m.matches[i].trainIdx].pt);
    }
    
    m.H = cv::findHomography(points1, points2, cv::RANSAC, 3, m.inliers_mask,2000,0.995);
    std::cout<<"RANSAC得到的H:"<<std::endl<<m.H<<std::endl;
    //m.H = findHomography_re(points1, points2, cv::RANSAC, 3, m.inliers_mask,2000,0.995);
    //std::cout<<"RANSAC_RE得到的H:"<<std::endl<<m.H<<std::endl;
    if(std::abs(determinant(m.H)) < std::numeric_limits<double>::epsilon())
    {
        m = cv::detail::MatchesInfo();
        return false;
    }
    // num of inliers and refine Homography
    m.num_inliers = 0;
    std::vector<cv::DMatch> inliers;
    for(int i=0; i <(int)m.inliers_mask.size();i++)
    {
        if (m.inliers_mask[i])
        {
            m.num_inliers++;
            inliers.push_back(m.matches[i]);
        }
    }
    m.matches.swap(inliers);
    points1.clear();
    points2.clear();
    for(int i=0;i<m.matches.size();i++)
    {
        points1.push_back(f1.keypoints[m.matches[i].queryIdx].pt);
        points2.push_back(f2.keypoints[m.matches[i].trainIdx].pt);
    }
    
    //m.H = cv::findHomography(points1, points2, cv::RANSAC, 1, m.inliers_mask,2000,0.995);
    cv::Mat Ai_2N9=cv::Mat::zeros(m.matches.size()*2,9,CV_64FC1);
    for(size_t i=0;i<m.matches.size();i++)
    {  
        Ai_2N9.at<double>(2*i+0,3)=-points1[i].x;
        Ai_2N9.at<double>(2*i+0,4)=-points1[i].y;
        Ai_2N9.at<double>(2*i+0,5)=-1;
        Ai_2N9.at<double>(2*i+0,6)=points2[i].y*points1[i].x;
        Ai_2N9.at<double>(2*i+0,7)=points2[i].y*points1[i].y;
        Ai_2N9.at<double>(2*i+0,8)=points2[i].y;
        
        Ai_2N9.at<double>(2*i+1,0)=points1[i].x;
        Ai_2N9.at<double>(2*i+1,1)=points1[i].y;
        Ai_2N9.at<double>(2*i+1,2)=1;
        Ai_2N9.at<double>(2*i+1,6)=-points2[i].x*points1[i].x;
        Ai_2N9.at<double>(2*i+1,7)=-points2[i].x*points1[i].y;
        Ai_2N9.at<double>(2*i+1,8)=-points2[i].x;
    }
    //std::cout<<Ai_2N9.row(1)<<std::endl;
    
    cv::Mat w, U, Vt;
    cv::SVD::compute(Ai_2N9, w, U, Vt);
    cv::Mat ans=Vt.row(Vt.rows-1);
    //std::cout<<"ans="<<ans<<std::endl;
    
    cv::Mat H = ans.reshape(0,3);
    //m.H = H / H.at<double>(2,2);
    //cv::Mat H8(8, 1, CV_64F, m.H.ptr<double>());
    //cv::createLMSolver(cv::makePtr<cv::HomographyRefineCallback>(points1, points2), 10)->run(H8);
    
    return true;
}

bool isHomoFromMatches(cv::detail::MatchesInfo& m, const cv::detail::ImageFeatures& f1, const cv::detail::ImageFeatures& f2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
    // compute other elements of MatchesInfo
    if((int)m.matches.size() < 5)
    {
        m = cv::detail::MatchesInfo();
        return false;
    }
    m.src_img_idx = f1.img_idx;
    m.dst_img_idx = f2.img_idx;
    
    // calculate Homography
    std::vector<cv::DMatch> a;
    for(int i=0;i<m.matches.size();i++)
    {
        points1.push_back(f1.keypoints[m.matches[i].queryIdx].pt);
        points2.push_back(f2.keypoints[m.matches[i].trainIdx].pt);
        //a.push_back(m.matches[i]);
    }
    //m.matches.swap(a);
    m.H = cv::findHomography(points1, points2, cv::RANSAC, 3, m.inliers_mask,2000,0.995);
    std::cout<<"RANSAC得到的H:"<<std::endl<<m.H<<std::endl;
    //m.H = cv::findHomography(points1, points2, cv::RHO, 3, m.inliers_mask,2000,0.995);
    //std::cout<<"RHO得到的H:"<<std::endl<<m.H<<std::endl;
    //cv::Mat img_0 = cv::imread("./1.bmp",CV_LOAD_IMAGE_COLOR);
    
    if(std::abs(determinant(m.H)) < std::numeric_limits<double>::epsilon())
    {
        m = cv::detail::MatchesInfo();
        return false;
    }
    // num of inliers and refine Homography
    m.num_inliers = 0;
    std::vector<cv::DMatch> inliers;
    for(int i=0; i <(int)m.inliers_mask.size();i++)
    {
        if (m.inliers_mask[i])
        {
            m.num_inliers++;
            inliers.push_back(m.matches[i]);
        }
    }
    m.matches.swap(inliers);
    points1.clear();
    points2.clear();
    for(int i=0;i<m.matches.size();i++)
    {
        points1.push_back(f1.keypoints[m.matches[i].queryIdx].pt);
        points2.push_back(f2.keypoints[m.matches[i].trainIdx].pt);
    }
    
    return true;
}

void calcPoint_after_H(const cv::Point2f& p_src, cv::Point2f& p_dst, const cv::Mat& H)
//输入p_src，输出p_dst
{
    double xx = p_src.x*H.at<double>(0,0) + p_src.y*H.at<double>(0,1) + H.at<double>(0,2);
    double yy = p_src.x*H.at<double>(1,0) + p_src.y*H.at<double>(1,1) + H.at<double>(1,2);
    double zz = p_src.x*H.at<double>(2,0) + p_src.y*H.at<double>(2,1) + H.at<double>(2,2);
    if (zz){
        p_dst.x = xx / zz;
        p_dst.y = yy / zz;
    }
    else{
        p_dst.x = xx;
        p_dst.y = yy;
    }
    
}

cv::Mat WarpImg(const cv::Mat img_src, cv::Point2i& TL_rect, const cv::Mat& H)//根据_H计算出img_wraped和TL_rect
{
    //计算四个角点经H变换后的点
    cv::Mat invH = H.inv();
    
    int width = img_src.cols, height = img_src.rows;
    cv::Point2f TL(0, 0), TR(width - 1, 0), BL(0, height - 1), BR(width - 1, height - 1);
    cv::Point2f TL_dst, TR_dst, BL_dst, BR_dst;
    calcPoint_after_H(TL, TL_dst, H);
    calcPoint_after_H(TR, TR_dst, H);
    calcPoint_after_H(BL, BL_dst, H);
    calcPoint_after_H(BR, BR_dst, H);
    float x_min, x_max, y_min, y_max;
    x_min = MIN(TL_dst.x, MIN(TR_dst.x, MIN(BL_dst.x, BR_dst.x)));
    y_min = MIN(TL_dst.y, MIN(TR_dst.y, MIN(BL_dst.y, BR_dst.y)));
    x_max = MAX(TL_dst.x, MAX(TR_dst.x, MAX(BL_dst.x, BR_dst.x)));
    y_max = MAX(TL_dst.y, MAX(TR_dst.y, MAX(BL_dst.y, BR_dst.y)));
    cv::Mat img_wraped;
    cv::warpPerspective(img_src, img_wraped, H, cv::Size(MIN(TR_dst.x,BR_dst.x),img_src.rows));
    TL_rect.x = 0;
    TL_rect.y = 0;
    /*
    //构建包络左上角点
    TL_rect.x = x_min;
    TL_rect.y = y_min;
    
    int rect_w = x_max - x_min + 1;
    int rect_h = y_max - y_min + 1;
    int nchannels = img_src.channels();
    cv::Mat img_wraped;
    if(nchannels == 3)
    {
        img_wraped.create(rect_h, rect_w, CV_8UC3);
        img_wraped.setTo(cv::Scalar(0, 0, 0));
    }
    else if(nchannels == 1)
    {
        img_wraped.create(rect_h, rect_w, CV_8U);
        img_wraped.setTo(cv::Scalar(0));
    }
    cv::Point2f offset(0 - TL_rect.x, 0 - TL_rect.y);//img_src坐标系原点(0,0)相对于img_wraped坐标系原点TL_rect的相对位置
    uchar *im_warp_data = img_wraped.data;
    //int top = (int)(0.05 *img_src.rows); int bottom = (int)(0.05*img_src.rows); int left = (int)(0.05 *img_src.cols); int right = (int)(0.05*img_src.cols);
    //cv::Mat img_srcbord;
    //copyMakeBorder( img_src, img_srcbord, top, bottom, left, right, cv::BORDER_REPLICATE, cv::Scalar() );
    //cv::imshow("11",img_src);
    //cv::imshow("12",img_srcbord);
    const uchar *img_src_data = img_src.data;
    
    //构建img_wraped：将img_wraped的点反映射到img_src上，然后双线性插值
    float *cl = new float[nchannels];//(x,y)插值结果
    for(int y=0; y<rect_h; y++)
        for(int x=0; x<rect_w; x++)//(x,y)为img_wraped上坐标点
        {
            cv::Point2f pt_imdst(x - offset.x, y - offset.y);//转换到img_src坐标系
            cv::Point2f pt_imsrc;//(x,y)反映射到imsrc上的坐标为小数
            calcPoint_after_H(pt_imdst, pt_imsrc, invH);
            if (pt_imsrc.x<0 || pt_imsrc.x>width - 1 || pt_imsrc.y<0 || pt_imsrc.y>height - 1)
                continue;//如果超出原图像边界，则(x,y)为全黑
            
            //双线性插值
            bilinearInterpolate(cl, pt_imsrc, img_src_data, width, height, nchannels);
            for (int c = 0; c < nchannels; ++c)
                im_warp_data[(y*rect_w + x)*nchannels + c] = uchar(cl[c]);
        }
    delete[] cl;
    //cv::copyMakeBorder( img_wraped, img_wraped, 100, 100, 100, 100,cv::BORDER_CONSTANT, cv::Scalar() );*/
    return img_wraped;
}

void warpImageAndMask(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& H, std::vector<cv::UMat>& images_warped, std::vector<cv::UMat>& masks_warped, std::vector<cv::Point>& corners)
{
    //两幅图像的蒙版
    cv::Mat masks_0(img0.rows, img0.cols, CV_8U);
    masks_0.setTo(cv::Scalar::all(255));
    cv::Mat masks_1(img1.rows, img1.cols, CV_8U);
    masks_1.setTo(cv::Scalar::all(255));
    
    cv::Mat img0_w,mask0_w;
    cv::Point2i corner_0;
    img0_w = cyl::WarpImg(img0, corner_0, H);
    mask0_w = cyl::WarpImg(masks_0, corner_0, H);
    cv::imshow("0",img0_w.getUMat(cv::ACCESS_RW));
    
    images_warped.push_back(img0_w.getUMat(cv::ACCESS_RW));
    images_warped.push_back(img1.getUMat(cv::ACCESS_RW));

    cv::Point2i corner_1;
    corner_1.x=0;corner_1.y=0;
    corners.push_back(corner_0);
    corners.push_back(corner_1);
    
    masks_warped.push_back(mask0_w.getUMat(cv::ACCESS_RW));
    masks_warped.push_back(masks_1.getUMat(cv::ACCESS_RW));
}

void findSeamAndBlend_LF(std::vector<std::vector<std::vector<cv::UMat>>>& images_warped_LF, std::vector<std::vector<std::vector<cv::UMat>>>& masks_warped_LF, std::vector<std::vector<std::vector<cv::Point>>>& corners_LF, const size_t& v, const size_t& u, cv::Mat& result, const int blend_type)
{
    
    std::vector<cv::UMat> images_warped_f(images_warped_LF[v][u].size());
    images_warped_LF[v][u][0].convertTo(images_warped_f[0], CV_32F);
    images_warped_LF[v][u][1].convertTo(images_warped_f[1], CV_32F);
    cv::Ptr<cyl::SeamFinder_LF> seam_finder = cv::makePtr<cyl::GraphCutSeamFinder_LF>(cyl::GraphCutSeamFinderBase_LF::COST_COLOR_EDGE);
    seam_finder->find(images_warped_f, corners_LF[v][u], masks_warped_LF[v][u], images_warped_LF,v,u);
    //seam_finder->find(imagesd_warped_f, corners, masks_warped);
    //cv::imshow("3",masks_warped[0]);
    //cv::imshow("4",masks_warped[1]);
    
    
    cv::Mat result_mask;
    std::vector<cv::Mat> images_w;
    images_w.push_back(images_warped_LF[v][u][0].getMat(cv::ACCESS_RW));
    images_w.push_back(images_warped_LF[v][u][1].getMat(cv::ACCESS_RW));
    std::vector<cv::Mat> masks_w;
    masks_w.push_back(masks_warped_LF[v][u][0].getMat(cv::ACCESS_RW));
    masks_w.push_back(masks_warped_LF[v][u][1].getMat(cv::ACCESS_RW));
    Blend(result, result_mask, images_w, masks_w, corners_LF[v][u], blend_type);
}

void BlendwithoutCut_LF(std::vector<cv::UMat>& images_warped, std::vector<cv::UMat>& masks_warped, std::vector<cv::Point>& corners, cv::Mat& result, const int blend_type)
{
    
    cv::Mat result_mask;
    std::vector<cv::Mat> images_w;
    images_w.push_back(images_warped[0].getMat(cv::ACCESS_RW));
    images_w.push_back(images_warped[1].getMat(cv::ACCESS_RW));
    std::vector<cv::Mat> masks_w;
    masks_w.push_back(masks_warped[0].getMat(cv::ACCESS_RW));
    masks_w.push_back(masks_warped[1].getMat(cv::ACCESS_RW));
    Blend(result, result_mask, images_w, masks_w, corners, blend_type);
}


//二维
void findSeamAndBlend(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& H, std::vector<cv::UMat>& images_warped, std::vector<cv::UMat>& masks_warped, std::vector<cv::Point>& corners, cv::Mat& result, const int blend_type)
{
    //两幅图像的蒙版
    cv::Mat masks_0(img0.rows, img0.cols, CV_8U);
    masks_0.setTo(cv::Scalar::all(255));
    cv::Mat masks_1(img1.rows, img1.cols, CV_8U);
    masks_1.setTo(cv::Scalar::all(255));
    
    cv::Mat img0_w,mask0_w;
    cv::Point2i corner_0;
    img0_w = cyl::WarpImg(img0, corner_0, H);
    mask0_w = cyl::WarpImg(masks_0, corner_0, H);
    cv::imshow("0",img0_w.getUMat(cv::ACCESS_RW));
    
    images_warped.push_back(img0_w.getUMat(cv::ACCESS_RW));
    images_warped.push_back(img1.getUMat(cv::ACCESS_RW));
    std::vector<cv::UMat> images_warped_f(images_warped.size());
    images_warped[0].convertTo(images_warped_f[0], CV_32F);
    images_warped[1].convertTo(images_warped_f[1], CV_32F);

    
    cv::Point2i corner_1;
    corner_1.x=0;corner_1.y=0;
    corners.push_back(corner_0);
    corners.push_back(corner_1);
    
    masks_warped.push_back(mask0_w.getUMat(cv::ACCESS_RW));
    masks_warped.push_back(masks_1.getUMat(cv::ACCESS_RW));
    
    cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    //cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cyl::GraphCutSeamFinder_LF>(cyl::GraphCutSeamFinderBase_LF::COST_COLOR_EDGE);
    seam_finder->find(images_warped_f, corners, masks_warped);
    //seam_finder->find(imagesd_warped_f, corners, masks_warped);
    //cv::imshow("3",masks_warped[0]);
    //cv::imshow("4",masks_warped[1]);
    
    cv::Mat result_mask;
    std::vector<cv::Mat> images_w;
    images_w.push_back(img0_w);
    images_w.push_back(img1);
    std::vector<cv::Mat> masks_w;
    masks_w.push_back(masks_warped[0].getMat(cv::ACCESS_RW));
    masks_w.push_back(masks_warped[1].getMat(cv::ACCESS_RW));
    Blend(result, result_mask, images_w, masks_w, corners, blend_type);
}

void Blend(cv::Mat& result, cv::Mat& result_mask, std::vector<cv::Mat>& Ims, std::vector<cv::Mat>& Mks, std::vector<cv::Point>& corners, int blend_type)
{
    bool try_gpu = false;
    //int blend_type = cv::detail::Blender::NO; //MULTI_BAND, FEATHER, NO
    float blend_strength = 5;
    int num_images = Ims.size();
    if (num_images < 2)
    {
        std::cout << "please load more images" << std::endl;
        return;
    }
    
    std::vector<cv::Size> sizes(num_images);
    for (int i = 0; i < num_images; i++)
        sizes[i] = Ims[i].size();
    
    cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(blend_type, try_gpu);
    cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
    if (blend_type == cv::detail::Blender::NO || blend_width < 1.f)
        blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, try_gpu);
    else if (blend_type == cv::detail::Blender::MULTI_BAND)
    {
        cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
        mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
    }
    else if (blend_type == cv::detail::Blender::FEATHER)
    {
        cv::detail::FeatherBlender* fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get());
        fb->setSharpness(1.f / blend_width);
    }
    blender->prepare(corners, sizes);
    for (int i = 0; i < num_images; i++)
    {
        cv::Mat img_s;
        Ims[i].convertTo(img_s, CV_16S);
        blender->feed(img_s, Mks[i], corners[i]);
    }
    blender->blend(result, result_mask);
    result.convertTo(result, CV_8UC3);
    result_mask.convertTo(result_mask, CV_8U);
}

cv::Mat computeHomo5x5(std::vector<cv::Point4f>& points1_LF,std::vector<cv::Point4f>& points2_LF)
{
    cv::Mat Ai_4N25=cv::Mat::zeros(points1_LF.size()*4,25,CV_64FC1);
    for(size_t i=0;i<points1_LF.size();i++)
    {  
        Ai_4N25.at<double>(4*i+0,15)=-points1_LF[i].u;
        Ai_4N25.at<double>(4*i+0,16)=-points1_LF[i].v;
        Ai_4N25.at<double>(4*i+0,17)=-points1_LF[i].x;
        Ai_4N25.at<double>(4*i+0,18)=-points1_LF[i].y;
        Ai_4N25.at<double>(4*i+0,19)=-1;
        Ai_4N25.at<double>(4*i+0,20)=points2_LF[i].y*points1_LF[i].u;
        Ai_4N25.at<double>(4*i+0,21)=points2_LF[i].y*points1_LF[i].v;
        Ai_4N25.at<double>(4*i+0,22)=points2_LF[i].y*points1_LF[i].x;
        Ai_4N25.at<double>(4*i+0,23)=points2_LF[i].y*points1_LF[i].y;
        Ai_4N25.at<double>(4*i+0,24)=points2_LF[i].y;
        
        Ai_4N25.at<double>(4*i+1,10)=points1_LF[i].u; 
        Ai_4N25.at<double>(4*i+1,11)=points1_LF[i].v;
        Ai_4N25.at<double>(4*i+1,12)=points1_LF[i].x;
        Ai_4N25.at<double>(4*i+1,13)=points1_LF[i].y;
        Ai_4N25.at<double>(4*i+1,14)=1;
        Ai_4N25.at<double>(4*i+1,20)=-points2_LF[i].x*points1_LF[i].u;
        Ai_4N25.at<double>(4*i+1,21)=-points2_LF[i].x*points1_LF[i].v;
        Ai_4N25.at<double>(4*i+1,22)=-points2_LF[i].x*points1_LF[i].x;
        Ai_4N25.at<double>(4*i+1,23)=-points2_LF[i].x*points1_LF[i].y;
        Ai_4N25.at<double>(4*i+1,24)=-points2_LF[i].x;
        
        Ai_4N25.at<double>(4*i+2,5)=-points1_LF[i].u;
        Ai_4N25.at<double>(4*i+2,6)=-points1_LF[i].v;
        Ai_4N25.at<double>(4*i+2,7)=-points1_LF[i].x;
        Ai_4N25.at<double>(4*i+2,8)=-points1_LF[i].y;
        Ai_4N25.at<double>(4*i+2,9)=-1;
        Ai_4N25.at<double>(4*i+2,20)=points2_LF[i].v*points1_LF[i].u;
        Ai_4N25.at<double>(4*i+2,21)=points2_LF[i].v*points1_LF[i].v;
        Ai_4N25.at<double>(4*i+2,22)=points2_LF[i].v*points1_LF[i].x;
        Ai_4N25.at<double>(4*i+2,23)=points2_LF[i].v*points1_LF[i].y;
        Ai_4N25.at<double>(4*i+2,24)=points2_LF[i].v;
        
        Ai_4N25.at<double>(4*i+3,0)=points1_LF[i].u;
        Ai_4N25.at<double>(4*i+3,1)=points1_LF[i].v;
        Ai_4N25.at<double>(4*i+3,2)=points1_LF[i].x;
        Ai_4N25.at<double>(4*i+3,3)=points1_LF[i].y;
        Ai_4N25.at<double>(4*i+3,4)=1;
        Ai_4N25.at<double>(4*i+3,20)=-points2_LF[i].u*points1_LF[i].u;
        Ai_4N25.at<double>(4*i+3,21)=-points2_LF[i].u*points1_LF[i].v;
        Ai_4N25.at<double>(4*i+3,22)=-points2_LF[i].u*points1_LF[i].x;
        Ai_4N25.at<double>(4*i+3,23)=-points2_LF[i].u*points1_LF[i].y;
        Ai_4N25.at<double>(4*i+3,24)=-points2_LF[i].u;
    }
    //std::cout<<Ai_4N25.row(0)<<std::endl;
    
    cv::Mat w, U, Vt;
    cv::SVD::compute(Ai_4N25, w, U, Vt);
    cv::Mat ans=Vt.row(Vt.rows-1);
    ans = ans/ans.at<double>(0,24);
    ans = ans.reshape(0,5);
    
    return ans;
}

void feature_withLFCenterConstraint(const cv::Mat& img_c0, const cv::Mat& img_0, const cv::Mat& img_1, const std::vector<cv::detail::ImageFeatures>& features_c, const std::vector<cv::detail::MatchesInfo>& matches_c, std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& matches)
{
    for(int i=0;i<matches_c[1].matches.size();i++)
    {
        features[2].keypoints.push_back(features_c[0].keypoints[matches_c[1].matches[i].queryIdx]);
    }
    //其他子孔径图像特征点先和筛选后的中心子孔径特征进行匹配
    cv::Ptr <cv::Feature2D> sift = cv::xfeatures2d :: SIFT :: create(3200);
    sift->detectAndCompute(img_0, cv::Mat(), features[0].keypoints, features[0].descriptors);
    sift->compute(img_c0,features[2].keypoints,features[2].descriptors);
    cv::detail::BestOf2NearestMatcher matcher;
    matcher(features,matches);
    matcher.collectGarbage();
    
    //用Ransac筛选同光场两张图之间的匹配关系
    std::vector<cv::Point2f> points1_temp, points2_temp;
    for(int i=0;i<matches[2].matches.size();i++)
    {
        points1_temp.push_back(features[0].keypoints[matches[2].matches[i].queryIdx].pt);
        points2_temp.push_back(features[2].keypoints[matches[2].matches[i].trainIdx].pt);
    }
    
    matches[2].H = cv::findHomography(points1_temp, points2_temp, cv::RANSAC, 3, matches[2].inliers_mask,2000,0.995);
    std::cout<<"同光场RANSAC得到的H:"<<std::endl<<matches[2].H<<std::endl;
    matches[2].num_inliers = 0;
    std::vector<cv::DMatch> inliers;
    for(int i=0; i <(int)matches[2].inliers_mask.size();i++)
    {
        if (matches[2].inliers_mask[i])
        {
            matches[2].num_inliers++;
            inliers.push_back(matches[2].matches[i]);
        }
    }
    matches[2].matches.swap(inliers);

    std::cout<<"同光场筛选后匹配点数为："<<matches[2].matches.size()<<std::endl;
    //cv::Mat img_match_center;
    //cv::drawMatches(img_0,features[0].keypoints,img_c0,features[2].keypoints,matches[2].matches,img_match_center);
    //cv::imshow("同光场匹配点对中心2",img_match_center);
    
    std::vector<cv::KeyPoint> keypoints_temp;
    for(int i=0;i<matches[2].matches.size();i++)
    {
        keypoints_temp.push_back(features[0].keypoints[matches[2].matches[i].queryIdx]);
    }
    features[0].keypoints.swap(keypoints_temp);
    //cv::Mat imgan0_feature;
    //cv::drawKeypoints(img_an_0,features_an[0].keypoints,imgan0_feature);
    //cv::imshow("imgan0_feature",imgan0_feature);
    
    //读入另一个光场对应位置的子孔径图像img_an_1
    sift->detectAndCompute(img_1, cv::Mat(), features[1].keypoints, features[1].descriptors);
    sift->compute(img_0,features[0].keypoints,features[0].descriptors);
    matcher(features,matches);
    matcher.collectGarbage();
}

}

