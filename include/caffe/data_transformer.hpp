#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    
    /**
     * @brief Applies common transformations to the input data, such as
     * scaling, mirroring, substracting the image mean...
     */
    template <typename Dtype>
    class DataTransformer {
    public:
        explicit DataTransformer(const TransformationParameter& param, Phase phase);
        virtual ~DataTransformer() {}
        
        /**
         * @brief Initialize the Random number generations if needed by the
         *    transformation.
         */
        void InitRand();
        
        /**
         * @brief Applies the transformation defined in the data layer's
         * transform_param block to the data.
         *
         * @param datum
         *    Datum containing the data to be transformed.
         * @param transformed_blob
         *    This is destination blob. It can be part of top blob's data if
         *    set_cpu_data() is used. See data_layer.cpp for an example.
         */
        void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
        
        /**
         * @brief Applies the transformation defined in the data layer's
         * transform_param block to a vector of Datum.
         *
         * @param datum_vector
         *    A vector of Datum containing the data to be transformed.
         * @param transformed_blob
         *    This is destination blob. It can be part of top blob's data if
         *    set_cpu_data() is used. See memory_layer.cpp for an example.
         */
        void Transform(const vector<Datum> & datum_vector,
                       Blob<Dtype>* transformed_blob);
        
#ifdef USE_OPENCV
        /**
         * @brief Applies the transformation defined in the data layer's
         * transform_param block to a vector of Mat.
         *
         * @param mat_vector
         *    A vector of Mat containing the data to be transformed.
         * @param transformed_blob
         *    This is destination blob. It can be part of top blob's data if
         *    set_cpu_data() is used. See memory_layer.cpp for an example.
         */
        void Transform(const vector<cv::Mat> & mat_vector,
                       Blob<Dtype>* transformed_blob);
        
        /**
         * @brief Applies the transformation defined in the data layer's
         * transform_param block to a cv::Mat
         *
         * @param cv_img
         *    cv::Mat containing the data to be transformed.
         * @param transformed_blob
         *    This is destination blob. It can be part of top blob's data if
         *    set_cpu_data() is used. See image_data_layer.cpp for an example.
         */
        void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
        /**
         *  @brief only applicable for GOT data
         *
         *  @param datum             
         *     A Blob containing the data to be transformed. It applies the same
         *     transformation to all the num images in the blob.
         *  @param transformed_data
         *     This is destination blob, it will contain as many images as the
         *     input blob. It can be part of top blob's data.
         *  @param transformed_label
         *     This is destination blob, it will contain as many labels as the
         *     input blob. It can be part of top blob's data.
         */
        void GOTTransform(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label);
#endif  // USE_OPENCV
        
        /**
         * @brief Applies the same transformation defined in the data layer's
         * transform_param block to all the num images in a input_blob.
         *
         * @param input_blob
         *    A Blob containing the data to be transformed. It applies the same
         *    transformation to all the num images in the blob.
         * @param transformed_blob
         *    This is destination blob, it will contain as many images as the
         *    input blob. It can be part of top blob's data.
         */
        void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);
        
        /**
         * @brief Infers the shape of transformed_blob will have when
         *    the transformation is applied to the data.
         *
         * @param datum
         *    Datum containing the data to be transformed.
         */
        vector<int> InferBlobShape(const Datum& datum);
        /**
         * @brief Infers the shape of transformed_blob will have when
         *    the transformation is applied to the data.
         *    It uses the first element to infer the shape of the blob.
         *
         * @param datum_vector
         *    A vector of Datum containing the data to be transformed.
         */
        vector<int> InferBlobShape(const vector<Datum> & datum_vector);
        /**
         * @brief Infers the shape of transformed_blob will have when
         *    the transformation is applied to the data.
         *    It uses the first element to infer the shape of the blob.
         *
         * @param mat_vector
         *    A vector of Mat containing the data to be transformed.
         */
#ifdef USE_OPENCV
        vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
        /**
         * @brief Infers the shape of transformed_blob will have when
         *    the transformation is applied to the data.
         *
         * @param cv_img
         *    cv::Mat containing the data to be transformed.
         */
        vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV
        
    protected:
        /**
         * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
         *
         * @param n
         *    The upperbound (exclusive) value of the random number.
         * @return
         *    A uniformly random integer value from ({0, 1, ..., n-1}).
         */
        virtual int Rand(int n);
        
        void Transform(const Datum& datum, Dtype* transformed_data);
        // Tranformation parameters
        TransformationParameter param_;
        
        
        shared_ptr<Caffe::RNG> rng_;
        Phase phase_;
        Blob<Dtype> data_mean_;
        vector<Dtype> mean_values_;
        
        template<typename T>
        void DecodeFloats(const string& data, size_t idx, T* pf, size_t len) {
            memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(T));
        }
#ifdef USE_OPENCV
        struct GOTLabelData {
            cv::Size img_size;
            int num_objects;
            std::vector<std::pair<cv::Vec4f,cv::Vec4f> > boxes; // bounding boxes, upper left and bottom right points
        };
        void ReadGOTLabelData(GOTLabelData& label_data, const std::string& data, int offset, int width);
        void GOTAugment(GOTLabelData& label_data);
        cv::Mat grayImageFromDatum(const Datum& datum, int offset);
        void CopyToDatum(Dtype* data, const cv::Mat& mat, Dtype mean=0.0, Dtype div=1.0);
        void GOTTransform(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label);
        
#endif
        
    };
    
}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
