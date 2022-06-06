#include "../include/visual_odom.hpp"
#include <boost/foreach.hpp> 

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

VisualOdom::VisualOdom(const std::string &folder_):image_folder_{folder_}{

    std::cout << "Inside the VO constructor!" << std::endl;    //build_image_list(image_folder_);

    build_image_list(folder_);



}

void VisualOdom::match_features(const cv::Mat &img_1){
//void VisualOdom::match_features(boost::filesystem::path &path_){

    using namespace cv;

    //Mat image = imread(path_.c_str());
    Mat image;
    cv::cvtColor(img_1, image, COLOR_BGR2GRAY);

    std::vector< cv::Point2f > corners;

    int maxCorners = 2000;

    double qualityLevel = 0.01;

    double minDistance = 1.;

    cv::Mat mask;
    
    int blockSize = 3;

    bool useHarrisDetector = false;

    double k = 0.04;

    cv::goodFeaturesToTrack( image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
    
    cv::cvtColor(image, image, COLOR_GRAY2BGR);


    for( size_t i = 0; i < corners.size(); i++ )
    {  
        
        cv::circle( image, corners[i], 2, cv::Scalar( 0, 255, 0), -1 );
    }



    waitKey(100);

    imshow("img",image);

}


void VisualOdom::build_image_list(const std::string &folder){

    namespace fs = boost::filesystem; 

    fs::path targetDir(folder); 

    fs::directory_iterator it(targetDir), eod;

    BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod))   
    { 
    
        if(fs::is_regular_file(p))
        {

            image_path_list_.push_back(p);

        } 
    }

    //printf("image_path_list.size(): %d\n", image_path_list_.size());
    sort(image_path_list_.begin(), image_path_list_.end());

}






int main(){


    VisualOdom visual_odom_("../data/02/image_0/");

   
    //cv::Mat img_1 = cv::imread("../data/02/image_0/000001.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat img_2 = cv::imread("../data/02/image_0/000002.png", cv::IMREAD_GRAYSCALE);

    int sz_ = (int)visual_odom_.image_path_list_.size(); 

    //int sz_ = 10;

    for(int i = 0 ; i < sz_ - 1; i++) {


        cv::Mat img_1 = cv::imread(visual_odom_.image_path_list_[i].c_str());
        cv::Mat img_2 = cv::imread(visual_odom_.image_path_list_[i + 1].c_str());

        visual_odom_.match_features(img_1);

    }

    //cv::waitKey(0);

    return 0;



}