#include "../include/visual_odom.hpp"
#include <boost/foreach.hpp> 

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

VisualOdom::VisualOdom(const std::string &folder_):image_folder_{folder_}{

    std::cout << "Inside the VO constructor!" << std::endl;    //build_image_list(image_folder_);

    build_image_list(folder_);



}

void VisualOdom::match_features(const cv::Mat &img_1,  const cv::Mat &img_2){

    using namespace cv;

    //Mat out_img_;
    //cvtColor(img_1, img_1, IMREAD_COLOR);
    imshow("image", img_1);


    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    std::vector<DMatch> matches;
    matcher->match ( descriptors_1, descriptors_2, matches );

    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    
    Scalar green_=  Scalar(0, 0, 255);

    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );

            circle(img_1, keypoints_1[matches[i].queryIdx].pt, 10, Scalar(0 , 0, 255));
            circle(img_1, keypoints_1[matches[i].trainIdx].pt, 10, Scalar(0 , 255, 0));
        
        }
    }

    imshow("outimg_1",img_1);


    //Mat img_match;
    //Mat img_goodmatch;
    //drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    //drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    
    //imshow ( "所有匹配点对", img_match );
    //imshow ( "优化后匹配点对", img_goodmatch );
    
    waitKey(100);
    
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

    printf("image_path_list.size(): %d\n", image_path_list_.size());

}






int main(){


    VisualOdom visual_odom_("../data/02/image_0/");

    sort(visual_odom_.image_path_list_.begin(), visual_odom_.image_path_list_.end());

    //cv::Mat img_1 = cv::imread("../data/02/image_0/000001.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat img_2 = cv::imread("../data/02/image_0/000002.png", cv::IMREAD_GRAYSCALE);

    int sz_ = (int)visual_odom_.image_path_list_.size(); 

    for(int i = 0 ; i < sz_ - 1; i++) {


        cv::Mat img_1 = cv::imread(visual_odom_.image_path_list_[i].c_str(), cv::IMREAD_COLOR);
        cv::Mat img_2 = cv::imread(visual_odom_.image_path_list_[i + 1].c_str(), cv::IMREAD_COLOR);

        visual_odom_.match_features(img_1, img_2);

    }

    cv::waitKey(0);

    return 0;



}