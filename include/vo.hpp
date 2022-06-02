#ifndef VO_H
#define VO_H


#include <iostream>

//*** opencv headers
#include <opencv2/opencv.hpp>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS



class VisualOdom{

    public:     

        VisualOdom(const std::string &folder_);

        void build_image_list(const std::string &folder_);



    private:

        std::string image_folder_;
        
        std::vector<boost::filesystem::path> image_path_list_; 


};


#endif