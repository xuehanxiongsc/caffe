#include <caffe/util/opencv_util.hpp>

namespace caffe {
    void imcrop(const cv::Mat& inputIm, cv::Mat& outputIm, const cv::Rect& inputROI, cv::Point& offset, bool fill, uchar fillValue)
    {
        if (inputROI.x >= inputIm.cols || inputROI.y >= inputIm.rows) return;
        int dstx = inputROI.x + inputROI.width - 1;
        int dsty = inputROI.y + inputROI.height - 1;
        if (dstx < 0 || dsty < 0) return;
        int minx = std::max<int>(0, inputROI.x);
        int miny = std::max<int>(0, inputROI.y);
        int maxx = std::min<int>(inputIm.cols - 1, dstx);
        int maxy = std::min<int>(inputIm.rows - 1, dsty);
        
        if (fill) {
            cv::Scalar toAdd;
            if (inputIm.channels() == 1) {
                toAdd = cv::Scalar(fillValue);
            } else if (inputIm.channels() == 3) {
                toAdd = cv::Scalar(fillValue,fillValue,fillValue);
            }
            outputIm = cv::Mat::zeros(inputROI.height, inputROI.width, inputIm.type());
            cv::add(outputIm, toAdd, outputIm);
            inputIm(cv::Range(miny, maxy + 1), cv::Range(minx, maxx + 1)).copyTo(outputIm(cv::Range(std::max(0,-inputROI.y), std::min(inputROI.height, inputIm.rows-inputROI.y)), cv::Range(std::max(0,-inputROI.x), std::min(inputROI.width, inputIm.cols-inputROI.x))));
            
            offset.x = inputROI.x;
            offset.y = inputROI.y;
            return;
        }
        offset.x = minx;
        offset.y = miny;
        outputIm = inputIm(cv::Range(miny, maxy + 1), cv::Range(minx, maxx + 1));
    }

    
}