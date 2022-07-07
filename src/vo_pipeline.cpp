#include "../include/visual_odom.hpp"


void VisualOdom::run_vo_pipeline(){


    cv::Mat predictions_mat_ = cv::Mat::zeros(1000, 1000, CV_8UC3);
    cv::Mat gt_mat_ = cv::Mat::zeros(1000, 1000, CV_8UC3);
    
    cv::namedWindow( "Predictions", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "Ground Truth", cv::WINDOW_AUTOSIZE );



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

        cv::Mat E_, mask_, R, t;


        //std::cout << "kp_1_matched.size(): " << kp_1_matched.size() << std::endl;
        //std::cout << "kp_2_matched.size(): " << kp_2_matched.size() << std::endl;

        E_ = cv::findEssentialMat(kp_2_matched, kp_1_matched, K_,cv::RANSAC, 0.99, 1.0, mask_);

        cv::recoverPose(E_, kp_2_matched, kp_1_matched, R ,t ,focal_, pp_, mask_);

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

        //std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << C_k_.at<double>(0,3) <<"," << C_k_.at<double>(2, 3) <<"]" <<std::endl;
        
        cv::hconcat(R, t, T_k_);
            
        cv::vconcat(T_k_, temp_, T_k_);

        //std::cout << "T_k: " << T_k_ << std::endl;

        C_k_ =  C_k_minus_1_ * T_k_;

        //std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << C_k_.at<double>(0,3) <<"," << C_k_.at<double>(2, 3) <<"]" <<std::endl;
        
        C_k_minus_1_ = C_k_;


        //*** visualization code

        
        //** ground truth poses
        double gx_ = gt_poses_[i].at<double>(0,3) + 500;
        double gy_ = gt_poses_[i].at<double>(2,3) + 500;

        //std::cout <<"(" << gx_ << "," << gy_ << ")" << std::endl;
        //** predicted poses
        double px_ =  C_k_.at<double>(0,3) + 500;
        double py_ = C_k_.at<double>(2,3) + 500;

        //cv::namedWindow("gt_poses", cv::WINDOW_AUTOSIZE);
        //cv::rectangle( traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0,0,0), -1);
        cv::circle(predictions_mat_, cv::Point(px_, py_) ,1, CV_RGB(255,0,0), 2);
        cv::circle(gt_mat_, cv::Point(gx_, gy_) ,1, CV_RGB(255,255, 255), 2);
        
        //cv::imshow("Predictions", predictions_mat_);
        //cv::imshow("Ground Truth", gt_mat_);

        //cv::Mat combined_mat_ = gt_mat_ + predictions_mat_;
        cv::Mat combined_mat_ = predictions_mat_ + gt_mat_;
        
        cv::imshow("Ground Truth", combined_mat_);
        
        //cv::imshow("Ground Truth", traj);
        
        cv::waitKey(1);

    }

    //std::cout << "predicted_poses_.size(): " << (int)predicted_poses_.size() << std::endl;

    /*for(int i = 0; i < 100; i++) {
        std::cout << "i: " << i << std::endl;
        std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << predicted_poses_[i].at<double>(0,3) <<"," << predicted_poses_[i].at<double>(2, 3) <<"]" <<std::endl;
        
    }*/


}
