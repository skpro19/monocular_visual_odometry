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


    //std::cout << "R_: " << R_ << std::endl;
    //std::cout << "t_: " << t_ << std::endl; 
   
    //cv::namedWindow( "Predictions", cv::WINDOW_AUTOSIZE );
    //cv::namedWindow( "Ground Truth", cv::WINDOW_AUTOSIZE );

    //cv::Mat predictions_mat_ = cv::Mat::zeros(600, 600, CV_8UC3);
    //cv::Mat gt_mat_ = cv::Mat::zeros(600, 600, CV_8UC3);
    
    //std::cout << "gt_poses[0]: "  << gt_poses_[0] << std::endl;

    cv::Mat E_, R, t;
    cv::Mat R_f, t_f;


    R_f.convertTo(R_f, CV_64F);
    t_f.convertTo(t_f, CV_64F);

    R.convertTo(R, CV_64F);
    t.convertTo(t, CV_64F);    

    C_k_.convertTo(C_k_, CV_64F);
    C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);
    
    ///C_k_minus_1_ = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);  //initial camera pose w.r.t to some frame 
    
    cv::Mat NO_ROT_ = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat NO_T_ = cv::Mat::zeros(3, 1, CV_64F);
    
    //cv::Mat TEMP_ = (cv::Mat_<float>(1, 4) << 0, 0, 0, 1); 
    cv::Mat TEMP_ = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1); 
    TEMP_.convertTo(TEMP_, CV_64F);

    //C_k_minus_1_ = gt_poses_[0];
    //C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);

    /*std::cout << "NO_ROT_: " << NO_ROT_ << std::endl;
    std::cout << "NO_T_: " << NO_T_ << std::endl;
    std::cout << "TEMP_: " << TEMP_ << std::endl;
    */

    //** Initializing C_k_minus_1 with 0 translation and 0 rotation w.r.t. the 'some' initial co-ordinate frame
    cv::hconcat(NO_ROT_, NO_T_, C_k_minus_1_);
    cv::vconcat(C_k_minus_1_, TEMP_, C_k_minus_1_);

    std::cout << "C_k_minus_1_: " << C_k_minus_1_ << std::endl;

    //C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);

    //std::cout << "C_k_minus_1_.size(): " << C_k_minus_1_.size() << std::endl;
    //std::cout << "C_k_minus_1_: " << C_k_minus_1_ << std::endl;


    int sz_ = image_file_names_.size(); 

    for(int i = 0 ; i < sz_  - 1; i++) {

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

        std::vector<cv::Point2f> kp_1f, kp_2f; //array of keypoint co-ordinates

        //converting vector<Keypoints> to vector<Point2f>
        for(int k = 0; k < (int)kp_1_matched.size(); k++) {

            cv::Point2f p1_ = kp_1_matched[k].pt, p2_ = kp_2_matched[k].pt;
            
            kp_1f.push_back(p1_); 
            kp_2f.push_back(p2_);

            cv::circle( img_1, p1_, 2, cv::Scalar( 0, 255, 0), -1 );
            cv::circle( img_1, p2_, 2, cv::Scalar( 255, 0, 0), -1 );
            cv::line(img_1, p1_, p2_, cv::Scalar(0, 255,0));
            
        }
        
        
        cv::Mat E_mask_;
        E_mask_.convertTo(E_mask_, CV_64F);

        E_ = cv::findEssentialMat(kp_2f, kp_1f, K_,cv::RANSAC, 0.999, 1.0, E_mask_);

        int inlier_cnt_ =0 ; 

        //int row_cnt_ = E_mask_.rows;

        //std::cout << "HI " << std::endl;

        /*
        for(int j = 0 ; j < row_cnt_; j++) {

            double val_ = E_mask_.at<double>(j, 0);

            if(val_ > 0) {

                cv::Point2f p1_ = kp_1_matched[j].pt;
                cv::circle( img_1, p1_, 2, cv::Scalar( 0, 255, 255), -1 );
            
            }

            if(val_ > 0) {inlier_cnt_++;}

        } */

        //cv::imshow("img1", img_1);

        inlier_cnt_ = cv::recoverPose(E_, kp_2f, kp_1f, K_, R, t, E_mask_);

        double scale_ = getAbsoluteScale(i); 
        //scale_ = 1.0;
        

        //std::cout << "H2" << std::endl;
        //t = scale_ * t ;

        /*std::cout << "t.size(): " << t.size() << std::endl;

        std::cout << "t.at<double>(0,0): " << t.at<double>(0, 0) << std::endl;
        std::cout << "t.at<double>(1,0): " << t.at<double>(1, 0) << std::endl;
        std::cout << "t.at<double>(2,0): " << t.at<double>(2, 0) << std::endl;
        */


        
       bool flag_ = ((scale_ > 0.1) &&  (t.at<double>(2,0) > t.at<double>(0, 0)) && (t.at<double>(2, 0) > t.at<double>(1,0)))  ; 
       
       //double scale_ = 1.0;

        if(i == 0) {
            
            t_f = t.clone();
            //t_f.convertTo(t_f, CV_64F);
            
            R_f = R.clone();
          //  R_f.convertTo(R_f, CV_64F);

          //  std::cout << "H3" << std::endl;

        }else {

                t_f = t_f + (R_f * t);
                R_f = R * R_f;
                
            
        }

        if(!flag_) {continue; ;}
        //std::cout << "Case One: " << "(" << t_f.at<double>(0,0) << "," << t_f.at<double>(1,0) <<  "," << t_f.at<double>(2,0) << ")";
        std::cout << "(" << t_f.at<double>(0,0) << "," << t_f.at<double>(1,0) <<  "," << t_f.at<double>(2,0) << ")" << std::endl;

        cv::Vec3f v_ = rotationMatrixToEulerAngles(R_f);

        //std::cout << "v_: " << v_ << std::endl;

        //draw_trajectory_windows(t_f, i);
        
        //cv::waitKey(1);

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
        std::cout << "(" <<C_k_.at<double>(0, 3) << "," << C_k_.at<double>(1, 3) << "," << C_k_.at<double>(2,3) << ")" << std::endl << std::endl;
        v_ = rotationMatrixToEulerAngles(R_f);

        //std::cout << "v_: " << v_ << std::endl;

        //std::cout << "Case Three: " << "(" <<gt_poses_[i].at<double>(0, 3)<< "," << gt_poses_[i].at<double>(2, 3)<< ")" << std::endl;

        draw_trajectory_windows(C_k_, i);

    }

}
