#include "../include/visual_odom.hpp"



void VisualOdom::process_data_files(){

    calib_file_name_ = "calib.txt";
    gt_file_name_ = "00.txt";
    image_dir_ = "image_0/";
    
    read_projection_matrix(calib_file_name_);
    load_camera_params_matrix();
    read_ground_truth_poses(gt_file_name_);
    read_image_files(image_dir_);

}

void VisualOdom::read_projection_matrix(const std::string &calib_file_name_){

    std::string calib_file_ = data_dir_ + calib_file_name_;
    
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


/**
 * @brief decompose P = K[R|t]  <--- extracting camera intrinsics and extrincics from the Projection Matrix
 *
 * 
 */
void VisualOdom::load_camera_params_matrix(){

    cv::decomposeProjectionMatrix(P_, K_, R_, t_);

    std::cout << "K_: " << K_ << std::endl;

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



void VisualOdom::read_image_files(const std::string &img_folder_name_){

    std::string image_folder_ = data_dir_ + img_folder_name_;

    cv::glob(image_folder_, image_file_names_, false);

    
}