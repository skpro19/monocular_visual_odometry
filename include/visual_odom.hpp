#ifndef VO_H
#define VO_H


#include <iostream>
#include <sstream>

//*** opencv headers
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>


//#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
//#undef BOOST_NO_CXX11_SCOPED_ENUMS




//TODO:
/*

- Rename visual_odom.cpp to feature_processing
- Create proejction matrix 
- cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
- Reduce the number of global variables in VisualOdom class
- Write static class for each of the source files

    

    
    //use glob
    //use cv::decomposeProjectionMatrix to decompose P =  [K R t]
    //import gt in  a gt_matrix

     Mat H = convertToHomogeneousMat(R_trans, T_trans);

    Transformation Matrix is actually called Homogenous Transformation Matrix

    Plot the predicted point and the acutal ground truth and the error in realtime

    GNUPlot

    remove kp_1, kp_1_matched from global variables --- potential source of error

    //add ransac visualization

*/

/*

Implementation Pipeline - 


*/

class VisualOdom{


    private:

        //**visualiztion_helpers.cpp
        void draw_trajectory_windows(const cv::Mat &C_k_, int i);
        void draw_trajectory_windows_mod(const cv::Mat &C_k_, int i);


        //**io.cpp
        void process_data_files();
        void read_projection_matrix(const std::string &file_name_);
        void load_camera_params_matrix(); 
        void read_ground_truth_poses(const std::string &file_name_);


        //** camera movement functions
        void calculate_camera_poses(); //calcula
        


        //** to be refactored
        void run_vo_pipeline();
        void run_pipeline();
        //void play_video();
        

    public:     

        VisualOdom(const std::string &folder_);

        
        void read_image_files(const std::string &folder_);
        void extract_features(const cv::Mat &img_1, const cv::Mat &img_2);
        void match_features(const cv::Mat &img_1, const cv::Mat &img_2);
        cv::Mat get_essential_matrix(const cv::Mat &img_1, const cv::Mat &img_2);        
        
        double getAbsoluteScale(int frame_id);
    
        std::string image_folder_;  
        
        std::vector<boost::filesystem::path> image_path_list_; 
        std::vector<cv::KeyPoint> kp_1, kp_2;
        std::vector<cv::KeyPoint> kp_1_matched, kp_2_matched; 


        //*** keypoints under consideration for feature matching
        std::vector<cv::KeyPoint> curr_kps_, prev_kps_;


        //*** camera params
        cv::Mat P_; //camera projection matrix -> [3 * 4]
        cv::Mat K_; //camera intrinsics matrix [3 * 3]
        cv::Mat R_; //camera axis rotation w.r.t. world 
        cv::Mat t_; //camera axis translation w.r.t. world 


        //** file input vars
        std::string base_dir_; 
        std::string data_dir_;
        std::string image_dir_;
        std::string calib_file_name_;
        std::string gt_file_name_;
        std::vector<cv::String> image_file_names_;

        //** data vars
        std::vector<cv::Mat> gt_poses_;
        std::vector<cv::Mat> predicted_poses_;
        

        //** camera translation params
         //C_k = C_k_minus_1_ * T_k_
        cv::Mat C_k_, C_k_minus_1_; //C_k -> camera pose in the kth frame w.r.t. intial frame
        cv::Mat T_k_; //relates the transform between the camera poses C_k_minus_1_ and C_k_


        //*** tuning params
        int good_matches_size_ = 10;
        //TODO 
        /*
        
        - add ransac parameters here from the findEssentialMat function
        - orb params tuning
        
        */


        //**visualization vars
        cv::Mat predictions_mat_ = cv::Mat::zeros(1000, 1000, CV_8UC3);
        cv::Mat gt_mat_ = cv::Mat::zeros(1000, 1000, CV_8UC3);


        double getScale(int curr_idx_, int prev_idx_);
        double get_z_scale(int curr_idx_, int prev_idx_);

};  


#endif