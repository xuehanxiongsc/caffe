#ifndef CAFFE_OPENCV_UTIL_H_
#define CAFFE_OPENCV_UTIL_H_

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <vector>

namespace caffe {
    void imcrop(const cv::Mat& inputIm, cv::Mat& outputIm,
                const cv::Rect& inputROI, cv::Point& offset, bool fill=false, uchar fillValue=0);
    
    inline std::vector<cv::Point> corners_from_rect(const cv::Rect& rect) {
        std::vector<cv::Point> corners(4);
        corners[0] = cv::Point(rect.x,rect.y);
        corners[1] = cv::Point(rect.x+rect.width-1,rect.y);
        corners[2] = cv::Point(rect.x+rect.width-1,rect.y+rect.height-1);
        corners[3] = cv::Point(rect.x,rect.y+rect.height-1);
        return corners;
    }
    
    inline cv::Rect vec2rect(const cv::Vec4f& vec) {
        return cv::Rect(vec[0],vec[1],vec[2]-vec[0]+1,vec[3]-vec[1]+1);
    }
    
    inline cv::Rect make_square(const cv::Rect& box, float margin) {
        int dim = std::max<int>(box.width,box.height);
        int new_dim = dim*(1.0+margin*2);
        int center_x = box.x + (box.width-1)/2;
        int center_y = box.y + (box.height-1)/2;
        int offset = (new_dim-1)/2;
        return cv::Rect(center_x-offset,center_y-offset,new_dim,new_dim);
    }
}

#endif

#endif