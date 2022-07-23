#include "../include/visual_odom.hpp"


//TODO-> Change C_k_ to mat_
void VisualOdom::draw_trajectory_windows(const cv::Mat &mat_, int i){

    //std::cout << "Inside the draw_trajectory_windows function" << std::endl;

    double gx_ = gt_poses_[i].at<double>(0,3) + 300;
    double gy_ = gt_poses_[i].at<double>(2,3) + 50;


    //** predicted poses
    double px_ =  mat_.at<double>(0,3) + 200;
    double py_ = mat_.at<double>(2,3) + 200;

    cv::circle(predictions_mat_, cv::Point(px_, py_) ,1, CV_RGB(255,0,0), 2);
    cv::circle(gt_mat_, cv::Point(gx_, gy_) ,1, CV_RGB(255,255, 255), 2);

    cv::Mat combined_mat_ = gt_mat_ + predictions_mat_;
    cv::hconcat(gt_mat_, predictions_mat_, combined_mat_);
    
    cv::imshow("Ground Truth", combined_mat_);
    //cv::imshow("C_K_", predictions_mat_);
    cv::waitKey(10);

}

void VisualOdom::draw_trajectory_windows_mod(const cv::Mat &t_f, int i){

   // std::cout << "Inside the draw_trajectory_windows function" << std::endl;

    double gx_ = gt_poses_[i].at<double>(0,3) + 200;
    double gy_ = gt_poses_[i].at<double>(2,3) + 300;


    //** predicted poses
    double px_ =  t_f.at<double>(0,0) + 200;
    double py_ = t_f.at<double>(2, 0) + 200;

    cv::circle(predictions_mat_, cv::Point(px_, py_) ,1, CV_RGB(255,0,0), 2);
    cv::circle(gt_mat_, cv::Point(gx_, gy_) ,1, CV_RGB(255,255, 255), 2);

    cv::Mat combined_mat_ = gt_mat_ + predictions_mat_;
    cv::hconcat(gt_mat_, predictions_mat_, combined_mat_);
    
    //cv::imshow("Ground Truth", combined_mat_);
    cv::imshow("t_f_", predictions_mat_);
    cv::waitKey(10);

}


