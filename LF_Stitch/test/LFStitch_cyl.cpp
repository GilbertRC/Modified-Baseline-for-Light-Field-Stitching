//更改了图割
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
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int LFrow_number=8;
    int LFcol_number=8;
    std::vector<std::vector<cv::Mat>> LF_0(LFrow_number), LF_1(LFrow_number);
    std::vector<cv::Point4f> points0_LF, points1_LF;
    //先读取中心子孔径做匹配
    //cv::Mat img_c0 = cv::imread("./IMG_0390_eslf/8_8.bmp",CV_LOAD_IMAGE_COLOR);
    //cv::Mat img_c1 = cv::imread("./IMG_0328_eslf/8_8.bmp",CV_LOAD_IMAGE_COLOR);
    //cv::Mat img_c0_d = cv::imread("./1_depth_re_WMF_1_cornerok.bmp",CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_c0 = cv::imread("./IMG_0490_eslf/8_8.bmp",CV_LOAD_IMAGE_COLOR);
    cv::Mat img_c1 = cv::imread("./IMG_0430_eslf/8_8.bmp",CV_LOAD_IMAGE_COLOR);
    cv::Mat img_c0_d = cv::imread("./9_depth_WMF_4_re_2.bmp",CV_LOAD_IMAGE_GRAYSCALE);
    std::vector<cv::detail::ImageFeatures> features_c(2);
    std::vector<cv::detail::MatchesInfo> matches_c(4);
    cyl::detectAndMatch_withDepth(img_c0, img_c0_d, img_c1, features_c, matches_c);//带有深度图信息的特征点提取与匹配
    //cyl::detectAndMatch_GMSwithDepth(img_c0, img_c0_d, img_c1, features_c, matches_c);
    //cyl::detectAndMatch(img_c0, img_c1, features_c, matches_c);
    std::cout<<"原匹配点数为："<<matches_c[1].matches.size()<<std::endl;
    std::vector<cv::Point2f> points_c0, points_c1;
    if(cyl::isHomoFromMatches(matches_c[1],features_c[0],features_c[1],points_c0,points_c1))
    {
        std::cout<<"内点数为："<<matches_c[1].matches.size()<<std::endl;
        cv::Mat img_match1;
        cv::drawMatches(img_c0,features_c[0].keypoints,img_c1,features_c[1].keypoints,matches_c[1].matches,img_match1);
        cv::imshow("匹配点对",img_match1);
    }
    
    //对除了中心子孔径的图像进行特征点筛选与匹配
    for(size_t v=0;v<LFrow_number;v++)
        for(size_t u=0;u<LFcol_number;u++)
        {
            std::cout<<"processing v="<<v<<", u="<<u<<std::endl;
            std::string img0_name, img1_name;
            std::stringstream uu,vv;
            std::string uutr,vvtr;
            uu<<(u+4);
            uu>>uutr;
            vv<<(v+4);
            vv>>vvtr;
            //img0_name = "./IMG_0390_eslf/"+ vvtr + "_" + uutr + ".bmp";
            //img1_name = "./IMG_0328_eslf/"+ vvtr + "_" + uutr + ".bmp";
            img0_name = "./IMG_0490_eslf/"+ vvtr + "_" + uutr + ".bmp";
            img1_name = "./IMG_0430_eslf/"+ vvtr + "_" + uutr + ".bmp"; 
            cv::Mat img_0 = cv::imread(img0_name,CV_LOAD_IMAGE_COLOR);
            cv::Mat img_1 = cv::imread(img1_name,CV_LOAD_IMAGE_COLOR);
            cv::imshow("img_0",img_0);
            cv::imshow("img_1",img_1);
            //cv::waitKey();
            LF_0[v].push_back(img_0);
            LF_1[v].push_back(img_1);
            
            if(v==4 && u==4)
            {
                for(int i=0;i<(int)points_c0.size();i++)
                {
                    points0_LF.push_back(cv::Point4f(u,v,points_c0[i].x,points_c0[i].y));
                    points1_LF.push_back(cv::Point4f(u,v,points_c1[i].x,points_c1[i].y));
                }
                continue;
            }
            if(v==8 || u==8) continue;
            //否则，用中心子孔径的筛选后特征点约束同个光场的其余特征点位置
            std::vector<cv::detail::ImageFeatures> features(3);//0-img_an_0, 1-img_an_1, 2-img_0
            std::vector<cv::detail::MatchesInfo> matches(9);
            cyl::feature_withLFCenterConstraint(img_c0, img_0, img_1, features_c, matches_c, features, matches);
            
            std::vector<cv::Point2f> points0, points1;
            if(cyl::isHomoFromMatches(matches[1],features[0],features[1],points0,points1))
            {
                std::cout<<"两个光场内点数为："<<matches[1].matches.size()<<std::endl;
                for(int i=0;i<(int)points0.size();i++)
                {
                    points0_LF.push_back(cv::Point4f(u,v,points0[i].x,points0[i].y));
                    points1_LF.push_back(cv::Point4f(u,v,points1[i].x,points1[i].y));
                }
            }
        }
        cv::Mat H5x5;
        //H5x5 = cyl::computeHomo5x5(points0_LF,points1_LF);
        cyl::runKernel_LF( points0_LF, points1_LF, H5x5 );
        std::cout<<"优化前"<<std::endl<<H5x5<<std::endl;
        cv::Mat H24(24, 1, CV_64F, H5x5.ptr<double>());
        cv::createLMSolver(cv::makePtr<cyl::HomographyRefineCallback_LF>(points0_LF, points1_LF), 10)->run(H24);
        std::cout<<"优化后"<<std::endl<<H5x5<<std::endl;
        
        
        std::vector<std::vector<std::vector<cv::UMat>>> images_warped_LF(LFrow_number);
        std::vector<std::vector<std::vector<cv::UMat>>> masks_warped_LF(LFrow_number);
        std::vector<std::vector<std::vector<cv::Point>>> corners_LF(LFrow_number);
        for(size_t v=0;v<LFrow_number;v++)
        {
            images_warped_LF[v].resize(LFcol_number);
            masks_warped_LF[v].resize(LFcol_number);
            corners_LF[v].resize(LFcol_number);
        }
        for(size_t v=0;v<LFrow_number;v++)
            for(size_t u=0;u<LFcol_number;u++)
            {
                std::cout<<"warping v="<<v<<", u="<<u<<std::endl;
                //用于图割
                std::vector<cv::UMat> images_warped;
                std::vector<cv::UMat> masks_warped;
                std::vector<cv::Point> corners;
                
                cv::Mat H = (cv::Mat_<double>(3,3) << H5x5.at<double>(0,12),H5x5.at<double>(0,13),H5x5.at<double>(0,10)*u+H5x5.at<double>(0,11)*v+H5x5.at<double>(0,14),H5x5.at<double>(0,17),H5x5.at<double>(0,18),H5x5.at<double>(0,15)*u+H5x5.at<double>(0,16)*v+H5x5.at<double>(0,19),H5x5.at<double>(0,22),H5x5.at<double>(0,23),H5x5.at<double>(0,20)*u+H5x5.at<double>(0,21)*v+H5x5.at<double>(0,24));
                H = H / H.at<double>(2,2);
                std::cout<<H<<std::endl;
                cyl::warpImageAndMask(LF_0[v][u], LF_1[v][u], H, images_warped, masks_warped, corners);
                
                images_warped_LF[v][u]=images_warped;
                masks_warped_LF[v][u]=masks_warped;
                corners_LF[v][u]=corners;
            }
            
        int result_height, result_width;
        std::vector<std::vector<cv::Mat>> LF_result(LFrow_number);
        for(size_t v=0;v<LFrow_number;v++)
            for(size_t u=0;u<LFcol_number;u++)
            {
                std::cout<<"blending v="<<v<<", u="<<u<<std::endl;
    
                cv::Mat result;
                cyl::findSeamAndBlend_LF(images_warped_LF, masks_warped_LF, corners_LF, v, u, result, cv::detail::Blender::MULTI_BAND);
                
                //cyl::findSeamAndBlend(images_warped_LF[v][u], masks_warped_LF[v][u], corners_LF[v][u], result, cv::detail::Blender::MULTI_BAND);
                //cv::imshow("result",result);
                //cv::waitKey();
                LF_result[v].push_back(result);
                if(u==0 && v==0) {
                    result_height=LF_result[v][u].rows; 
                    result_width=LF_result[v][u].cols; 
                }
                else{
                    if(LF_result[v][u].rows < result_height) result_height=LF_result[v][u].rows;
                    if(LF_result[v][u].cols < result_width) result_width=LF_result[v][u].cols;
                }
                
            }
        
        

        for(size_t v=0;v<LFrow_number;v++)
            for(size_t u=0;u<LFcol_number;u++)
            {
                std::string result_name;
                std::stringstream r_uu,r_vv;
                std::string r_uutr,r_vvtr;
                r_uu<<(u+1);
                r_uu>>r_uutr;
                r_vv<<(v+1);
                r_vv>>r_vvtr;
                //result_name = "./result_0328_0390(5)/"+ r_vvtr + "_" + r_uutr + ".bmp";
                result_name = "./result_0430_0490(7)_9x9/"+ r_vvtr + "_" + r_uutr + ".bmp";
                cv::Mat result=cv::Mat::zeros(result_height, result_width, CV_8UC3);
                LF_result[v][u](cv::Rect(0, 0, result_width, result_height)).copyTo(result);
                cv::imwrite(result_name,result);
                
                
                //result_name = "./result_mask390(5)/"+ r_vvtr + "_" + r_uutr + ".bmp";
                //cv::imwrite(result_name,masks_warped_LF[v][u][1]);
            }
        
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        std::cout<<"time cost = "<<time_used.count()<<" seconds. "<<std::endl;
        
        cv::waitKey(0);
        
        return 0;
}



