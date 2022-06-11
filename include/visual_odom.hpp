#ifndef VO_H
#define VO_H


#include <iostream>

//*** opencv headers
#include <opencv2/opencv.hpp>

//#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
//#undef BOOST_NO_CXX11_SCOPED_ENUMS



class VisualOdom{

    public:     

        VisualOdom(const std::string &folder_);

        void load_camera_params();
        


        void build_image_list(const std::string &folder_);
        void extract_features(const cv::Mat &img_1, const cv::Mat &img_2);
        void match_features(const cv::Mat &img_1, const cv::Mat &img_2);
        
    //private:

        std::string image_folder_;
        
        std::vector<boost::filesystem::path> image_path_list_; 
        std::vector<cv::KeyPoint> kp_1, kp_2; 
  

};


#endif