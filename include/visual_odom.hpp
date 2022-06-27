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

- Create proejction matrix 
cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);

    use glob
    use cv::decomposeProjectionMatrix to decompose P =  [K R t]
    import gt in  a gt_matrix

     Mat H = convertToHomogeneousMat(R_trans, T_trans);

    Transformation Matrix is actually called Homogenous Transformation Matrix

    Plot the predicted point and the acutal ground truth and the error in realtime

    GNUPlot

*/

class VisualOdom{


    private:

        //**input functions
        void read_projection_matrix();
        void load_camera_params_matrix(); 
        void load_camera_params();
        void read_ground_truth_poses();

    public:     

        VisualOdom(const std::string &folder_);

        
        void build_image_list(const std::string &folder_);
        void extract_features(const cv::Mat &img_1, const cv::Mat &img_2);
        void match_features(const cv::Mat &img_1, const cv::Mat &img_2);
        cv::Mat get_essential_matrix(const cv::Mat &img_1, const cv::Mat &img_2);        
        
        double getAbsoluteScale(int frame_id);
    
        std::string image_folder_;  
        
        std::vector<boost::filesystem::path> image_path_list_; 
        std::vector<cv::KeyPoint> kp_1, kp_2; 

        
        std::vector<cv::Point2f> kp_1_matched, kp_2_matched; 



        double focal_ ; 
        cv::Point2d pp_;

        //** data vars
        std::vector<cv::Mat> gt_poses_;
        
        
        //** camera parameters -> extrinsic + intrinsic
        cv::Mat P_; //camera projection matrix -> [3 * 3]
        cv::Mat K_; //camera matrix -> [3 * 3]
        cv::Mat R_; //camera rotation -> [3 * 3]
        cv::Mat t_; //camera translation -> [4 * 1]

        //** file input vars
        std::string base_dir_; 
        std::string data_dir_;
        std::string calib_file_name_;
        std::string gt_file_name_;

};  


#endif