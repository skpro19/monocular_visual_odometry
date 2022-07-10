#include "../include/visual_odom.hpp"


void VisualOdom::run_vo_pipeline(){


   
    //cv::namedWindow( "Predictions", cv::WINDOW_AUTOSIZE );
    //cv::namedWindow( "Ground Truth", cv::WINDOW_AUTOSIZE );

    C_k_minus_1_ = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);  //initial camera pose w.r.t to some frame 
    C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);

    int sz_ = image_file_names_.size(); 

    for(int i = 0 ; i < sz_ - 1; i++) {

        kp_1.resize(0); 
        kp_2.resize(0); 

        kp_1_matched.resize(0); 
        kp_2_matched.resize(0);
    
        cv::Mat img_1 = cv::imread(image_file_names_[i].c_str());
        cv::Mat img_2 = cv::imread(image_file_names_[i + 1].c_str());

        extract_features(img_1, img_2);
        match_features(img_1, img_2);

        cv::Mat E_, R, t;

        assert((int)kp_1_matched.size() == (int)kp_2_matched.size());

        std::vector<cv::Point2f> kp_1f, kp_2f; //array of keypoint co-ordinates


        for(int k = 0; k < (int)kp_1_matched.size(); k++) {

            cv::Point2f p1_ = kp_1_matched[k].pt, p2_ = kp_2_matched[k].pt;
            
            kp_1f.push_back(p1_); 
            kp_2f.push_back(p2_);

            cv::circle( img_1, p1_, 2, cv::Scalar( 0, 255, 0), -1 );
            cv::circle( img_1, p2_, 2, cv::Scalar( 255, 0, 0), -1 );
            cv::line(img_1, p1_, p2_, cv::Scalar(0, 255,0));
            
        }
        
        
        cv::Mat E_mask_;

        E_ = cv::findEssentialMat(kp_2f, kp_1f, K_,cv::RANSAC, 0.9, 0.1, E_mask_);

        std::cout << "E_mask_.size(): " << E_mask_.size() << std::endl;

        int inlier_cnt_ =0 ; 

        int row_cnt_ = E_mask_.rows;

        for(int j = 0 ; j < row_cnt_; j++) {

            double val_ = E_mask_.at<double>(j, 0);

            if(val_ > 0) {

                cv::Point2f p1_ = kp_1_matched[j].pt;
                cv::circle( img_1, p1_, 2, cv::Scalar( 0, 255, 255), -1 );
            
            }

            if(val_ > 0) {inlier_cnt_++;}

        } 

        cv::imshow("img1", img_1);

        std::cout << "inlier_cnt_ before recoverPose: " << inlier_cnt_ << std::endl;

        //inlier_cnt_ = cv::recoverPose(E_, kp_2f, kp_1f, R ,t ,focal_, pp_, E_mask_);
        inlier_cnt_ = cv::recoverPose(E_, kp_2f, kp_1f, K_, R, t, E_mask_);

        std::cout << "inlier_cnt_ after recoverPose: " << inlier_cnt_ << std::endl;

        //std::cout << "inlier_cnt for frame " << i << " is: " << inlier_cnt_ << std::endl;
        //std::cout <<"mask: " << mask_  << std::endl;
        
        
        
        //** building T_k_
        T_k_ = cv::Mat();

        cv::Mat temp_ = (cv::Mat_<float>(1, 4) << 0, 0, 0, 1); 
        temp_.convertTo(temp_, CV_64F);

        //std::cout << "R.type(): " << R.type() << std::endl;
        //std::cout << "t.type(): " << t.type() << std::endl;

        R.convertTo(R, CV_64F);
        t.convertTo(t, CV_64F);    

        //std::cout << "R.size(): " << R.size() << std::endl;
        //std::cout << "t.size(): " << t.size() << std::endl;

        double scale_ = getAbsoluteScale(i); 

       // std::cout << "kp_2_matched.size(): " << kp_2_matched.size() << std::endl;

        /*if(scale_ > 0.1 && (t.at<double>(2, 0) > t.at<double>(1, 0)) && (t.at<double>(2, 0) > t.at<double>(0,0))) {

            //std::cout << "t: " << t << std::endl;
            //std::cout << "scale: " << scale_ << std::endl;
            t = scale_ * t ; 

            //std::cout << "t: " << t << std::endl;

            cv::hconcat(R, t, T_k_);
            
            cv::vconcat(T_k_, temp_, T_k_);

            //std::cout << "T_k: " << T_k_ << std::endl;

            C_k_ =  C_k_minus_1_ * T_k_;

            //std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << C_k_.at<double>(0,3) <<"," << C_k_.at<double>(2, 3) <<"]" <<std::endl;
            
            C_k_minus_1_ = C_k_;

        }

        else {

            C_k_ = C_k_minus_1_;

        }*/

        cv::hconcat(R, t, T_k_);            
        cv::vconcat(T_k_, temp_, T_k_);

        C_k_ =  C_k_minus_1_ * T_k_;
        C_k_minus_1_ = C_k_;


        draw_trajectory_windows(C_k_, i);


        cv::waitKey(1);

    }

    //std::cout << "predicted_poses_.size(): " << (int)predicted_poses_.size() << std::endl;

    /*for(int i = 0; i < 100; i++) {
        std::cout << "i: " << i << std::endl;
        std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << predicted_poses_[i].at<double>(0,3) <<"," << predicted_poses_[i].at<double>(2, 3) <<"]" <<std::endl;
        
    }*/


}
