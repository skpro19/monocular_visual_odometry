#include "../include/visual_odom.hpp"


cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{

    //assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);

}

void VisualOdom::run_vo_pipeline(){

    cv::Mat E_, R, t;
    cv::Mat R_f, t_f;

    R_f.convertTo(R_f, CV_64F);
    t_f.convertTo(t_f, CV_64F);

    R.convertTo(R, CV_64F);
    t.convertTo(t, CV_64F);    

    C_k_.convertTo(C_k_, CV_64F);
    C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);
    
    cv::Mat NO_ROT_ = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat NO_T_ = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::Mat TEMP_ = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1); 
    TEMP_.convertTo(TEMP_, CV_64F);
    
    
    //** Initializing C_k_minus_1 with 0 translation and 0 rotation w.r.t. the 'some' initial co-ordinate frame
    cv::hconcat(NO_ROT_, NO_T_, C_k_minus_1_);
    cv::vconcat(C_k_minus_1_, TEMP_, C_k_minus_1_);

    std::cout << "C_k_minus_1_: " << C_k_minus_1_ << std::endl;

    //C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);

    //std::cout << "C_k_minus_1_.size(): " << C_k_minus_1_.size() << std::endl;
    //std::cout << "C_k_minus_1_: " << C_k_minus_1_ << std::endl;



    int sz_ = image_file_names_.size(); 

    for(int i = 400 ; i < sz_  - 1; i++) {

        //std::cout << "i: " << i << std::endl;

        kp_1.resize(0); 
        kp_2.resize(0); 

        kp_1_matched.resize(0); 
        kp_2_matched.resize(0);
    
        cv::Mat img_1 = cv::imread(image_file_names_[i].c_str());
        cv::Mat img_2 = cv::imread(image_file_names_[i + 1 ].c_str());

        extract_features(img_1, img_2);
        match_features(img_1, img_2);

        assert((int)kp_1_matched.size() == (int)kp_2_matched.size());

        if(i == 0) {

            prev_kps_ = kp_1_matched;

        }

        curr_kps_ = kp_2_matched;


        std::vector<cv::Point2f> kp_1f, kp_2f; //array of keypoint co-ordinates

        //converting vector<Keypoints> to vector<Point2f>
        for(int k = 0; k < (int)kp_1_matched.size(); k++) {

            cv::Point2f p1_ = kp_1_matched[k].pt, p2_ = kp_2_matched[k].pt;
            
            kp_1f.push_back(p1_); 
            kp_2f.push_back(p2_);

            cv::circle( img_1, p1_, 2, cv::Scalar( 0, 255, 0), -1 );
            cv::line(img_1, p1_, p2_, cv::Scalar(0, 255,0));
            cv::circle( img_1, p2_, 2, cv::Scalar( 255, 0, 0), -1 );
            
        }
        
        
        cv::Mat E_mask_;
        E_mask_.convertTo(E_mask_, CV_64F);

        E_ = cv::findEssentialMat(kp_2f, kp_1f, K_,cv::RANSAC, 0.999, 1.0, E_mask_);

        std::cout << "E_mask_: " << E_mask_.size() << std::endl;

        int inlier_cnt_ =0 ; 

        cv::imshow("img1", img_1);

        inlier_cnt_ = cv::recoverPose(E_, kp_2f, kp_1f, K_, R, t, E_mask_);

        std::cout << "inlier_cnt_: " << inlier_cnt_ << std::endl;

        std::cout << "E_mask_.size(): " << E_mask_.size() << std::endl;

        /*if(inlier_cnt_ < 200) {

            std::cout << "INLIER CNT < 50 --> " << inlier_cnt_ << "------ IGNORING!" << std::endl;

        }*/

        double scale_ = getAbsoluteScale(i); 
        
        bool flag_ = ((scale_ > 0.1) &&  (t.at<double>(2,0) > t.at<double>(0, 0)) && (t.at<double>(2, 0) > t.at<double>(1,0)))  ; 

        //bool flag_ = ((scale_ > 0.1) &&  (t.at<double>(2,0) > t.at<double>(0, 0)) && (t.at<double>(2, 0) > t.at<double>(1,0)) && t.at<double>(2,0) > 0.1)  ; 

        std::cout <<"scale_: " << scale_ << std::endl;

        std::cout << std::endl;

        if(!flag_) {continue; ;}

        
        cv::Mat temp_ = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1); 
        temp_.convertTo(temp_, CV_64F);

        //T_k_ = cv::Mat(); 
        T_k_.convertTo(T_k_, CV_64F);

        cv::hconcat(R, t, T_k_);
        cv::vconcat(T_k_, temp_, T_k_);

        
        C_k_ =  C_k_minus_1_ * T_k_;
        C_k_minus_1_ = C_k_;

        //std::cout << "Case Two: " << "(" <<C_k_.at<double>(0, 3)/C_k_.at<double>(3, 3) << "," << C_k_.at<double>(2, 3)/C_k_.at<double>(3,3) << ")" << std::endl;
        //std::cout << "Case Two: " << "(" <<C_k_.at<double>(0, 3) << "," << C_k_.at<double>(1, 3) << "," << C_k_.at<double>(2,3) << ")" << " ";
        //std::cout << "(" <<C_k_.at<double>(0, 3) << "," << C_k_.at<double>(1, 3) << "," << C_k_.at<double>(2,3) << ")" << std::endl << std::endl;
        //v_ = rotationMatrixToEulerAngles(R_f);

        //std::cout << "v_: " << v_ << std::endl;

        //std::cout << "Case Three: " << "(" <<gt_poses_[i].at<double>(0, 3)<< "," << gt_poses_[i].at<double>(2, 3)<< ")" << std::endl;

       draw_trajectory_windows(C_k_, i);
        //draw_trajectory_windows_mod(t_f, i);
    
    }

}
