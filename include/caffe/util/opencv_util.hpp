#ifndef CAFFE_OPENCV_UTIL_H_
#define CAFFE_OPENCV_UTIL_H_

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <vector>

namespace caffe {

template<class Iterator>
void Flip(const cv::Mat& image, Iterator begin, Iterator end,
          cv::Mat* flip_image, cv::Mat* landmarks) {
  cv::flip(image, *flip_image, 1);
  cv::Mat landmark_ref = *landmarks;
  landmark_ref.col(0) = image.cols - landmark_ref.col(0) - 1.0f;
  cv::Mat landmark_copy = landmark_ref.clone();
  int count = 0;
  for (Iterator it = begin; it != end; it++,count++) {
    landmark_copy.row(*it).copyTo(landmarks->row(count));
  }
}

template<typename T>
void UpdateHeatmap(T* data, const cv::Point2f& pt, int height, int width,
                   float sigma) {
  int count = 0;
  float inv_sigma_sqr = 0.5/(sigma*sigma);
  for (int i = 0; i < height; i++) {
    float dy2 = (i-pt.y)*(i-pt.y);
    for (int j = 0; j < width; j++) {
      float diff = (dy2 + (j-pt.x)*(j-pt.x))*inv_sigma_sqr;
      data[count] += (diff < 4.6052 ? std::expf(-diff) : 0.f);
      data[count] = data[count] > 1.0 ? 1.0 : data[count];
      ++count;
    }
  }
}

template<typename T>
void BackgroundHeatmap(const T* fgd_ptr, int height, int width,
                       int num_fgd, T* bgd_ptr) {
  
  for (int n = 0; n < num_fgd; n++) {
    int count = 0;
    const int offset = height*width*n;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        bgd_ptr[count] += fgd_ptr[count+offset];
        ++count;
      }
    }
  }
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      bgd_ptr[count] = std::min<T>(1.0,std::max<T>(1.0-bgd_ptr[count],0.0));
      ++count;
    }
  }
}

void imcrop(const cv::Mat& in_image, const cv::Rect& roi, cv::Point& offset,
            cv::Mat* out_image, bool fill);

inline std::vector<cv::Point> corners_from_rect(const cv::Rect& rect) {
  std::vector<cv::Point> corners(4);
  corners[0] = cv::Point(rect.x,rect.y);
  corners[1] = cv::Point(rect.x+rect.width-1,rect.y);
  corners[2] = cv::Point(rect.x+rect.width-1,rect.y+rect.height-1);
  corners[3] = cv::Point(rect.x,rect.y+rect.height-1);
  return corners;
}

template<typename T>
cv::Rect_<T> vec2rect(const cv::Vec4f& vec) {
  return cv::Rect_<T>(vec[0],vec[1],vec[2]-vec[0]+1,vec[3]-vec[1]+1);
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
