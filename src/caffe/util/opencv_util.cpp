#include "caffe/util/opencv_util.hpp"

namespace caffe {
void imcrop(const cv::Mat& in_image, const cv::Rect& roi, cv::Point& offset,
            cv::Mat* out_image, bool fill)
{
  if (roi.x >= in_image.cols || roi.y >= in_image.rows) return;
  int dstx = roi.x + roi.width - 1;
  int dsty = roi.y + roi.height - 1;
  if (dstx < 0 || dsty < 0) return;
  int minx = std::max<int>(0, roi.x);
  int miny = std::max<int>(0, roi.y);
  int maxx = std::min<int>(in_image.cols - 1, dstx);
  int maxy = std::min<int>(in_image.rows - 1, dsty);
  
  if (fill) {
    *out_image = cv::Mat::zeros(roi.height, roi.width, in_image.type());
    cv::Range out_col_range(std::max(0,-roi.x),
                            std::min(roi.width, in_image.cols-roi.x));
    cv::Range out_row_range(std::max(0,-roi.y),
                            std::min(roi.height, in_image.rows-roi.y));
    cv::Mat& out_image_ref = *out_image;
    in_image(cv::Range(miny, maxy + 1), cv::Range(minx, maxx + 1)).copyTo(
        out_image_ref(out_row_range,out_col_range));
    offset.x = roi.x;
    offset.y = roi.y;
    return;
  }
  offset.x = minx;
  offset.y = miny;
  in_image(cv::Range(miny, maxy + 1), cv::Range(minx, maxx + 1)).copyTo(
      *out_image);
}

    
}
