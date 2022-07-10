#include "../include/visual_odom.hpp"


 void VisualOdom::draw_trajectory_windows(const cv::Mat &C_k_, int i){

    std::cout << "Inside the draw_trajectory_windows function" << std::endl;

    double gx_ = gt_poses_[i].at<double>(0,3) + 200;
    double gy_ = gt_poses_[i].at<double>(2,3) + 200;


    //** predicted poses
    double px_ =  C_k_.at<double>(0,3) + 200;
    double py_ = C_k_.at<double>(2,3) + 200;

    cv::circle(predictions_mat_, cv::Point(px_, py_) ,1, CV_RGB(255,0,0), 2);
    cv::circle(gt_mat_, cv::Point(gx_, gy_) ,1, CV_RGB(255,255, 255), 2);
    
    cv::Mat combined_mat_ = gt_mat_ + predictions_mat_;
    cv::hconcat(gt_mat_, predictions_mat_, combined_mat_);
    
    cv::imshow("Ground Truth", combined_mat_);

 }

