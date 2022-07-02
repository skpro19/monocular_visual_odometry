#include "../include/visual_odom.hpp"
#include <boost/foreach.hpp> 

#include <fstream>
#include <iostream>


//VisualOdom::VisualOdom(const std::string &folder_):image_folder_{folder_}{
VisualOdom::VisualOdom(const std::string &folder_){

    base_dir_ = "/home/skpro19/simple_visual_odom/";
    data_dir_ = base_dir_ + "data/02/";
    
    
    process_data_files();

   run_vo_pipeline();

    //read_projection_matrix();
    //read_ground_truth_poses();
    //load_camera_params();


    //build_image_list(folder_);

    //load_camera_params();


}

void VisualOdom::process_data_files(){

    calib_file_name_ = "calib.txt";
    gt_file_name_ = "02.txt";
    image_dir_ = "image_0/";
    
    read_projection_matrix(calib_file_name_);
    load_camera_params_matrix();
    read_ground_truth_poses(gt_file_name_);
    read_image_files(image_dir_);

}

void VisualOdom::read_ground_truth_poses(const std::string &gt_file_name_){

    //std::cout <<"Inside the read_ground_truth_poses function!" << std::endl;

    std::string gt_file_ = data_dir_ + gt_file_name_;
    
    std::ifstream gt_;

    int cnt = 1; 

    gt_.open(gt_file_);
    
    if(gt_.is_open()) {

        std::string line_;

        while(std::getline(gt_, line_) ){

            //float f_;
            double f_;
            std::stringstream ss_(line_);
            std::vector<float> v_; 

            for(int i = 0; i < 12 ; i++) {

                ss_ >> f_; 
                v_.push_back(f_);

            }
            
            cv::Mat gt_;
            gt_ = cv::Mat(v_).reshape(0, 3);
            gt_.convertTo(gt_, CV_64F);

            gt_poses_.push_back(gt_);

        }

    }



    else {

        std::cerr << "Unable to read calib file!" << std::endl;

    }

}


/**
 * @brief decompose P = K[R|t]
 * 
 */
void VisualOdom::load_camera_params_matrix(){

    cv::decomposeProjectionMatrix(P_, K_, R_, t_);

    /*std::cout << "P_.size(): " << P_.size() << std::endl; 
    std::cout << "K_.size(): " << K_.size() << std::endl; 
    
    std::cout << "K_: " << K_ << std::endl;
    

    std::cout << "R_.size(): " << R_.size() << std::endl;
    std::cout << "R_: " << R_ << std::endl; 
    
    std::cout << "t_.size(): " << t_.size() << std::endl;
    std::cout << "t_: " << t_ << std::endl;
    */

}


/**
 * @brief read projection matrix from calib.txt to P_
 * 
 */

void VisualOdom::read_projection_matrix(const std::string &calib_file_name_){

    std::string calib_file_ = data_dir_ + calib_file_name_;
    
    //std::cout << "calib_file_: " << calib_file_ << std::endl;

    std::ifstream calib_;

    calib_.open(calib_file_);
    if(calib_.is_open()) {

        std::string line_;
        std::getline(calib_, line_);

        //std::cout << "line_: " << line_ << std::endl;

        std::string s_;
        double f_;

        std::stringstream ss_(line_);

        ss_ >> s_; // bypass "P0:""

        //std::cout << "s_: " << s_ << std::endl;

        std::vector<float> v_;

        int cnt_ = 12;
        while(ss_ >> f_ && cnt_ > 0) {

            v_.push_back(f_);
            cnt_--;

        }

        //std::cout << "v_.size(): " << v_.size() << std::endl; 

        //for(auto t: v_) std::cout << t << " " ;
        //std::cout << std::endl;

        //auto *ptr =  v_.data();

        //P_ = cv::Mat(3, 4,  CV_64F, ptr);
        
        P_ = cv::Mat(v_).reshape(0, 3);
        P_.convertTo(P_, CV_64F);

        std::cout << P_.size() << " " << P_ << std::endl;

    }



    else {

        std::cerr << "Unable to read calib file!" << std::endl;

    }

    //cv::FileStorage fs_(calib_file_, cv::FileStorage::READ);

    


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
    
    
    cv::Mat image_one, image_two;
    cv::cvtColor(img_1, image_one, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2, image_two, cv::COLOR_BGR2GRAY);
    
    cv::Mat mask = cv::Mat();
    cv::Mat des_1, des_2;
    
    //cv::Ptr<cv::ORB>orb_ = cv::ORB::create(5000);
    cv::Ptr<cv::ORB>orb_ = cv::ORB::create(2000);

    
    
    //*** extracting descriptors from keypoints
    orb_->detectAndCompute(image_one, mask, kp_1, des_1);
    orb_->detectAndCompute(image_two, mask, kp_2, des_2);
    
    
    des_1.convertTo(des_1, 0);
    des_2.convertTo(des_2, 0);
    
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

    std::vector<cv::DMatch> good_matches;
    
    for ( int i = 0; i < des_1.rows; i++ )
    {
        if ( brute_hamming_matches[i].distance <= std::max( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back (brute_hamming_matches[i]);
        }
    }


    for (auto match : good_matches) {
        
        kp_1_matched.push_back(cv::Point2f{kp_1[match.queryIdx].pt.x, kp_1[match.queryIdx].pt.y}); 
        kp_2_matched.push_back(cv::Point2f{kp_2[match.trainIdx].pt.x, kp_2[match.trainIdx].pt.y}); 
        
    }

}

void VisualOdom::read_image_files(const std::string &img_folder_name_){

    std::string image_folder_ = data_dir_ + img_folder_name_;

    cv::glob(image_folder_, image_file_names_, false);

    
}

double VisualOdom::getAbsoluteScale(int frame_id)	{

    using namespace std;

    string line;
    int i = 0;
    ifstream myfile ("/home/skpro19/simple_visual_odom/data/02/02.txt");
    
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
    z_prev = z;
    x_prev = x;
    y_prev = y;
    std::istringstream in(line);
    //cout << line << '\n';
    for (int j=0; j<12; j++)  {
    in >> z ;
    if (j==7) y=z;
    if (j==3)  x=z;
    }

    i++;
    }
    myfile.close();
    }

    else {
    cout << "Unable to open file";
    return 0;
    }

    //std::cout << "x: " << x  << " y: " << y << " z: " << z << std::endl; 
    //std::cout << "x_prev: " << x_prev << " y_prev: " << y_prev << " z_prev: " << z_prev << std::endl;

    double scale_ =  sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

    //std::cout << "frame: " << frame_id << "scale: " << scale_ << std::endl;

    return scale_;

}


void VisualOdom::run_vo_pipeline(){

    C_k_minus_1_ = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);  //initial camera pose w.r.t to some frame 
    C_k_minus_1_.convertTo(C_k_minus_1_, CV_64F);
        
    //std::cout << "Inside the run_vo_pipeline function!" << std::endl;

    int sz_ = image_file_names_.size(); 

    //std::cout << "sz_: " << sz_ << std::endl;

    sz_ = 2; 

    //std::cout << "K_: " << K_ << std::endl;

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


        std::cout << "kp_1_matched.size(): " << kp_1_matched.size() << std::endl;
        std::cout << "kp_2_matched.size(): " << kp_2_matched.size() << std::endl;

        E_ = cv::findEssentialMat(kp_2_matched, kp_1_matched, K_,cv::RANSAC, 0.99, 1.0, mask_);
        
        //std::cout << "E_: " << E_ << std::endl;

        cv::recoverPose(E_, kp_2_matched, kp_1_matched, R ,t ,focal_, pp_, mask_);

        //std::cout << "R.size(): " << R.size() << std::endl;
        //std::cout << "t.size(): " << t.size() << std::endl;

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

        //if(scale_ > 0.1 && (t.at<double>(2, 0) > t.at<double>(1, 0)) && (t.at<double>(2, 0) > t.at<double>(0,0))) {

            //std::cout << "t: " << t << std::endl;
            //std::cout << "scale: " << scale_ << std::endl;
            t = scale_ * t ; 

            //std::cout << "t: " << t << std::endl;

            cv::hconcat(R, t, T_k_);
            
            cv::vconcat(T_k_, temp_, T_k_);

            std::cout << "T_k: " << T_k_ << std::endl;

            C_k_ =  C_k_minus_1_ * T_k_;

            //std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << C_k_.at<double>(0,3) <<"," << C_k_.at<double>(2, 3) <<"]" <<std::endl;
            
            C_k_minus_1_ = C_k_;

        //}

        //else {

          //  C_k_ = C_k_minus_1_;

        //}

        std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << C_k_.at<double>(0,3) <<"," << C_k_.at<double>(2, 3) <<"]" <<std::endl;
            

    }

    //std::cout << "predicted_poses_.size(): " << (int)predicted_poses_.size() << std::endl;

    /*for(int i = 0; i < 100; i++) {
        std::cout << "i: " << i << std::endl;
        std::cout << "[" << gt_poses_[i].at<double>(0,3) << "," << gt_poses_[i].at<double>(2,3) <<"] --->" << "[" << predicted_poses_[i].at<double>(0,3) <<"," << predicted_poses_[i].at<double>(2, 3) <<"]" <<std::endl;
        
    }*/


}


int main(){

    VisualOdom vo_("abcd");

    return 0;



}