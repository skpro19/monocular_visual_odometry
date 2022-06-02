#include "../include/vo.hpp"
#include <boost/foreach.hpp> 



VisualOdom::VisualOdom(const std::string &folder_):image_folder_{folder_}{

    std::cout << "Inside the VO constructor!" << std::endl;    //build_image_list(image_folder_);



}


void VisualOdom::build_image_list(const std::string &folder){

    namespace fs = boost::filesystem; 

    fs::path targetDir(folder); 

    fs::directory_iterator it(targetDir), eod;

    BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod))   
    { 
    
        if(fs::is_regular_file(p))
        {

            image_path_list_.push_back(p);

        } 
    
    
    }

    printf("image_path_list.size(): %d\n", image_path_list_.size());

}






int main(){


    VisualOdom vo_("../data/02/image_0/");


    cv::Mat image_ = cv::imread("../data/02/image_0/000001.png", cv::IMREAD_GRAYSCALE);

    std::cout << image_.size() << std::endl; 

    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image", image_);

    cv::waitKey(0);

    return 0;



}