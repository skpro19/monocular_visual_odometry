#include "../include/visual_odom.hpp"
#include <boost/foreach.hpp> 

#include <fstream>
#include <iostream>


//VisualOdom::VisualOdom(const std::string &folder_):image_folder_{folder_}{
VisualOdom::VisualOdom(const std::string &folder_){


    std::cout << "version: " << CV_VERSION << std::endl;

    base_dir_ = "/home/skpro19/simple_visual_odom/";
    //data_dir_ = base_dir_ + "data/00/";
    
    
    process_data_files();

    run_vo_pipeline();

    //read_projection_matrix();
    //read_ground_truth_poses();
    //load_camera_params();


    //build_image_list(folder_);

    //load_camera_params();


}


void VisualOdom::extract_features(const cv::Mat &img_1, const cv::Mat &img_2){

    cv::Mat image_one, image_two;

    cv::cvtColor(img_1, image_one, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2, image_two, cv::COLOR_BGR2GRAY);

    std::vector< cv::Point2f > corners_one, corners_two;
    
    int maxCorners = 2000;

    double qualityLevel = 0.01;

    double minDistance = 1.0;

    cv::Mat mask = cv::Mat();
    
    int blockSize = 1;

    bool useHarrisDetector = false;

    double k = 0.04;

    
    //*** keypoints extraction
    cv::goodFeaturesToTrack( image_one, corners_one, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );

    cv::goodFeaturesToTrack( image_two, corners_two, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );

    cv::KeyPoint::convert(kp_1, corners_one, std::vector<int>());
    cv::KeyPoint::convert(kp_2, corners_two, std::vector<int>());


}

void VisualOdom::match_features(const cv::Mat &img_1, const cv::Mat &img_2){
    
    kp_1_matched.clear(); 
    kp_2_matched.clear();

    cv::Mat image_one, image_two;
    cv::cvtColor(img_1, image_one, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2, image_two, cv::COLOR_BGR2GRAY);
    
    cv::Mat mask = cv::Mat();
    cv::Mat des_1, des_2;
    
    cv::Ptr<cv::ORB>orb_ = cv::ORB::create(5000);

    //*** extracting descriptors from keypoints
    orb_->detectAndCompute(image_one, mask, kp_1, des_1);
    orb_->detectAndCompute(image_two, mask, kp_2, des_2);
        
    des_1.convertTo(des_1, 0);
    des_2.convertTo(des_2, 0);
    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<cv::DMatch> brute_hamming_matches;
    matcher->match(des_1, des_2, brute_hamming_matches);

    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < des_1. rows; i++ )
    {
        double dist = brute_hamming_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    
    for ( int i = 0; i < des_1.rows; i++ )
    {
        if ( brute_hamming_matches[i].distance <= std::max( 2*min_dist, 20.0 ) )
        {
            good_matches.push_back (brute_hamming_matches[i]);
        }
    }

    for (auto match : good_matches) {
        
        kp_1_matched.push_back(kp_1[match.queryIdx]);
        kp_2_matched.push_back(kp_2[match.trainIdx]);

    }
    
}


double VisualOdom::getScale(int curr_idx_, int prev_idx_) {

    cv::Mat prev_poses_ = gt_poses_[prev_idx_];
    cv::Mat curr_poses_ = gt_poses_[curr_idx_]; 

    cv::Point3d prev_point_ = {prev_poses_.at<double>(0,3), prev_poses_.at<double>(1,3), prev_poses_.at<double>(2,3)};
    cv::Point3d curr_point_ = {curr_poses_.at<double>(0,3), curr_poses_.at<double>(1,3), curr_poses_.at<double>(2,3)};
    cv::Point3d diff_ = (curr_point_ - prev_point_);

    double scale_ = cv::norm(diff_);
    
    return scale_;

}







