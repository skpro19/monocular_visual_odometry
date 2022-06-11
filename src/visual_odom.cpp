#include "../include/visual_odom.hpp"
#include <boost/foreach.hpp> 

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

VisualOdom::VisualOdom(const std::string &folder_):image_folder_{folder_}{

    std::cout << "Inside the VO constructor!" << std::endl;    //build_image_list(image_folder_);

    build_image_list(folder_);



}

void VisualOdom::extract_features(const cv::Mat &img_1, const cv::Mat &img_2){

    cv::Mat image_one, image_two;

    cv::cvtColor(img_1, image_one, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2, image_two, cv::COLOR_BGR2GRAY);

    std::vector< cv::Point2f > corners_one, corners_two;

    
    int maxCorners = 5000;

    double qualityLevel = 0.01;

    double minDistance = 1.0;

    cv::Mat mask = cv::Mat();
    
    int blockSize = 1;

    bool useHarrisDetector = false;

    double k = 0.04;

    
    //*** keypoints extraction
    cv::goodFeaturesToTrack( image_one, corners_one, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
    cv::goodFeaturesToTrack( image_two, corners_two, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );

    
    for (auto point: corners_one) {

        kp_1.push_back(cv::KeyPoint(cv::Point2f(point.x, point.y), 2));

    }

    for (auto point: corners_two) {


        kp_2.push_back(cv::KeyPoint(cv::Point2f(point.x, point.y), 2));
    }
    

}


void VisualOdom::match_features(const cv::Mat &img_1, const cv::Mat &img_2){
    
    
    cv::Mat image_one, image_two;
    cv::cvtColor(img_1, image_one, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2, image_two, cv::COLOR_BGR2GRAY);
    
    cv::Mat mask = cv::Mat();
    cv::Mat des_1, des_2;
    
    cv::Ptr<cv::ORB>orb_ = cv::ORB::create(5000);

    


    //*** extracting descriptors from keypoints
    orb_->detectAndCompute(image_one, mask, kp_1, des_1);
    orb_->detectAndCompute(image_two, mask, kp_2, des_2);
    
   // des_1.convertTo(des_1, 5);
    //des_2.convertTo(des_2, 5);
    
    des_1.convertTo(des_1, 0);
    des_2.convertTo(des_2, 0);
    

    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<cv::DMatch> brute_hamming_matches;
    matcher->match(des_1, des_2, brute_hamming_matches);

    
    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < des_1.rows; i++ )
    {
        double dist = brute_hamming_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    std::vector<cv::DMatch> good_matches;
    
    for ( int i = 0; i < des_1.rows; i++ )
    {
        if ( brute_hamming_matches[i].distance <= std::max( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back (brute_hamming_matches[i] );
        }
    }
    

    for (auto match : good_matches) {

        cv::circle( img_1, kp_1[match.queryIdx].pt, 2, cv::Scalar( 0, 255, 0), -1 );
        cv::circle( img_1, kp_2[match.trainIdx].pt, 2, cv::Scalar( 255, 0, 0), -1 );
        cv::line(img_1, kp_1[match.queryIdx].pt, kp_2[match.trainIdx].pt, cv::Scalar(0, 255,0));
    }

    

    cv::imshow("img",img_1);
    cv::waitKey(10);
  

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

    int sz_ = (int)visual_odom_.image_path_list_.size(); 

    for(int i = 0 ; i < sz_ - 1; i++) {

        cv::Mat img_1 = cv::imread(visual_odom_.image_path_list_[i].c_str());
        cv::Mat img_2 = cv::imread(visual_odom_.image_path_list_[i + 1].c_str());

        visual_odom_.extract_features(img_1, img_2);
        visual_odom_.match_features(img_1, img_2);

    }

    return 0;



}