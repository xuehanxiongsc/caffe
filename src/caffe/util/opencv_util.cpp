#include "caffe/util/opencv_util.hpp"

#include <opencv2/imgproc/imgproc.hpp>

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

cv::Mat imrotate(const cv::Mat& in_image, float deg, cv::Mat* out_image) {
  const int width = in_image.cols;
  const int height = in_image.rows;
  cv::Mat border = (cv::Mat_<float>(2,4)
                    << 0.f, width-1, width-1, 0.f,
                    0.f, 0.f, height-1, height-1);
  cv::Mat R = R_from_deg(deg2rad(deg));
  cv::Mat rotated_border = R*border;
  cv::Mat min_border,max_border,dim;
  cv::reduce(rotated_border,min_border,1,CV_REDUCE_MIN);
  cv::reduce(rotated_border,max_border,1,CV_REDUCE_MAX);
  dim = max_border - min_border + 1.f;
  cv::Size dst_size(static_cast<int>(dim.at<float>(0)),
                    static_cast<int>(dim.at<float>(1)));
  cv::Mat M, wim, wpts;
  cv::hconcat(R,-min_border,M);
  cv::warpAffine(in_image, *out_image, M, dst_size);
  return M;
}
}
