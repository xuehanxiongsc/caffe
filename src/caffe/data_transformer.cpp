#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/opencv_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
    
    template<typename Dtype>
    DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
                                            Phase phase)
    : param_(param), phase_(phase) {
        // check if we want to use mean_file
        if (param_.has_mean_file()) {
            CHECK_EQ(param_.mean_value_size(), 0) <<
            "Cannot specify mean_file and mean_value at the same time";
            const string& mean_file = param.mean_file();
            if (Caffe::root_solver()) {
                LOG(INFO) << "Loading mean file from: " << mean_file;
            }
            BlobProto blob_proto;
            ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
            data_mean_.FromProto(blob_proto);
        }
        // check if we want to use mean_value
        if (param_.mean_value_size() > 0) {
            CHECK(param_.has_mean_file() == false) <<
            "Cannot specify mean_file and mean_value at the same time";
            for (int c = 0; c < param_.mean_value_size(); ++c) {
                mean_values_.push_back(param_.mean_value(c));
            }
        }
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const Datum& datum,
                                           Dtype* transformed_data) {
        const string& data = datum.data();
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        
        const int crop_size = param_.crop_size();
        const Dtype scale = param_.scale();
        const bool do_mirror = param_.mirror() && Rand(2);
        const bool has_mean_file = param_.has_mean_file();
        const bool has_uint8 = data.size() > 0;
        const bool has_mean_values = mean_values_.size() > 0;
        
        CHECK_GT(datum_channels, 0);
        CHECK_GE(datum_height, crop_size);
        CHECK_GE(datum_width, crop_size);
        
        Dtype* mean = NULL;
        if (has_mean_file) {
            CHECK_EQ(datum_channels, data_mean_.channels());
            CHECK_EQ(datum_height, data_mean_.height());
            CHECK_EQ(datum_width, data_mean_.width());
            mean = data_mean_.mutable_cpu_data();
        }
        if (has_mean_values) {
            CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
            "Specify either 1 mean_value or as many as channels: " << datum_channels;
            if (datum_channels > 1 && mean_values_.size() == 1) {
                // Replicate the mean_value for simplicity
                for (int c = 1; c < datum_channels; ++c) {
                    mean_values_.push_back(mean_values_[0]);
                }
            }
        }
        
        int height = datum_height;
        int width = datum_width;
        
        int h_off = 0;
        int w_off = 0;
        if (crop_size) {
            height = crop_size;
            width = crop_size;
            // We only do random crop when we do training.
            if (phase_ == TRAIN) {
                h_off = Rand(datum_height - crop_size + 1);
                w_off = Rand(datum_width - crop_size + 1);
            } else {
                h_off = (datum_height - crop_size) / 2;
                w_off = (datum_width - crop_size) / 2;
            }
        }
        
        Dtype datum_element;
        int top_index, data_index;
        for (int c = 0; c < datum_channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
                    if (do_mirror) {
                        top_index = (c * height + h) * width + (width - 1 - w);
                    } else {
                        top_index = (c * height + h) * width + w;
                    }
                    if (has_uint8) {
                        datum_element =
                        static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                    } else {
                        datum_element = datum.float_data(data_index);
                    }
                    if (has_mean_file) {
                        transformed_data[top_index] =
                        (datum_element - mean[data_index]) * scale;
                    } else {
                        if (has_mean_values) {
                            transformed_data[top_index] =
                            (datum_element - mean_values_[c]) * scale;
                        } else {
                            transformed_data[top_index] = datum_element * scale;
                        }
                    }
                }
            }
        }
    }
    
    
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const Datum& datum,
                                           Blob<Dtype>* transformed_blob) {
        // If datum is encoded, decoded and transform the cv::image.
        if (datum.encoded()) {
#ifdef USE_OPENCV
            CHECK(!(param_.force_color() && param_.force_gray()))
            << "cannot set both force_color and force_gray";
            cv::Mat cv_img;
            if (param_.force_color() || param_.force_gray()) {
                // If force_color then decode in color otherwise decode in gray.
                cv_img = DecodeDatumToCVMat(datum, param_.force_color());
            } else {
                cv_img = DecodeDatumToCVMatNative(datum);
            }
            // Transform the cv::image into blob.
            return Transform(cv_img, transformed_blob);
#else
            LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
        } else {
            if (param_.force_color() || param_.force_gray()) {
                LOG(ERROR) << "force_color and force_gray only for encoded datum";
            }
        }
        
        const int crop_size = param_.crop_size();
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        
        // Check dimensions.
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        const int num = transformed_blob->num();
        
        CHECK_EQ(channels, datum_channels);
        CHECK_LE(height, datum_height);
        CHECK_LE(width, datum_width);
        CHECK_GE(num, 1);
        
        if (crop_size) {
            CHECK_EQ(crop_size, height);
            CHECK_EQ(crop_size, width);
        } else {
            CHECK_EQ(datum_height, height);
            CHECK_EQ(datum_width, width);
        }
        
        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        Transform(datum, transformed_data);
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                           Blob<Dtype>* transformed_blob) {
        const int datum_num = datum_vector.size();
        const int num = transformed_blob->num();
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        
        CHECK_GT(datum_num, 0) << "There is no datum to add";
        CHECK_LE(datum_num, num) <<
        "The size of datum_vector must be no greater than transformed_blob->num()";
        Blob<Dtype> uni_blob(1, channels, height, width);
        for (int item_id = 0; item_id < datum_num; ++item_id) {
            int offset = transformed_blob->offset(item_id);
            uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
            Transform(datum_vector[item_id], &uni_blob);
        }
    }
    
#ifdef USE_OPENCV
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                           Blob<Dtype>* transformed_blob) {
        const int mat_num = mat_vector.size();
        const int num = transformed_blob->num();
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        
        CHECK_GT(mat_num, 0) << "There is no MAT to add";
        CHECK_EQ(mat_num, num) <<
        "The size of mat_vector must be equals to transformed_blob->num()";
        Blob<Dtype> uni_blob(1, channels, height, width);
        for (int item_id = 0; item_id < mat_num; ++item_id) {
            int offset = transformed_blob->offset(item_id);
            uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
            Transform(mat_vector[item_id], &uni_blob);
        }
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                           Blob<Dtype>* transformed_blob) {
        const int crop_size = param_.crop_size();
        const int img_channels = cv_img.channels();
        const int img_height = cv_img.rows;
        const int img_width = cv_img.cols;
        
        // Check dimensions.
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        const int num = transformed_blob->num();
        
        CHECK_EQ(channels, img_channels);
        CHECK_LE(height, img_height);
        CHECK_LE(width, img_width);
        CHECK_GE(num, 1);
        
        CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
        
        const Dtype scale = param_.scale();
        const bool do_mirror = param_.mirror() && Rand(2);
        const bool has_mean_file = param_.has_mean_file();
        const bool has_mean_values = mean_values_.size() > 0;
        
        CHECK_GT(img_channels, 0);
        CHECK_GE(img_height, crop_size);
        CHECK_GE(img_width, crop_size);
        
        Dtype* mean = NULL;
        if (has_mean_file) {
            CHECK_EQ(img_channels, data_mean_.channels());
            CHECK_EQ(img_height, data_mean_.height());
            CHECK_EQ(img_width, data_mean_.width());
            mean = data_mean_.mutable_cpu_data();
        }
        if (has_mean_values) {
            CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
            "Specify either 1 mean_value or as many as channels: " << img_channels;
            if (img_channels > 1 && mean_values_.size() == 1) {
                // Replicate the mean_value for simplicity
                for (int c = 1; c < img_channels; ++c) {
                    mean_values_.push_back(mean_values_[0]);
                }
            }
        }
        
        int h_off = 0;
        int w_off = 0;
        cv::Mat cv_cropped_img = cv_img;
        if (crop_size) {
            CHECK_EQ(crop_size, height);
            CHECK_EQ(crop_size, width);
            // We only do random crop when we do training.
            if (phase_ == TRAIN) {
                h_off = Rand(img_height - crop_size + 1);
                w_off = Rand(img_width - crop_size + 1);
            } else {
                h_off = (img_height - crop_size) / 2;
                w_off = (img_width - crop_size) / 2;
            }
            cv::Rect roi(w_off, h_off, crop_size, crop_size);
            cv_cropped_img = cv_img(roi);
        } else {
            CHECK_EQ(img_height, height);
            CHECK_EQ(img_width, width);
        }
        
        CHECK(cv_cropped_img.data);
        
        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        int top_index;
        for (int h = 0; h < height; ++h) {
            const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < img_channels; ++c) {
                    if (do_mirror) {
                        top_index = (c * height + h) * width + (width - 1 - w);
                    } else {
                        top_index = (c * height + h) * width + w;
                    }
                    // int top_index = (c * height + h) * width + w;
                    Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                    if (has_mean_file) {
                        int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                        transformed_data[top_index] =
                        (pixel - mean[mean_index]) * scale;
                    } else {
                        if (has_mean_values) {
                            transformed_data[top_index] =
                            (pixel - mean_values_[c]) * scale;
                        } else {
                            transformed_data[top_index] = pixel * scale;
                        }
                    }
                }
            }
        }
    }
#endif  // USE_OPENCV
    
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                           Blob<Dtype>* transformed_blob) {
        const int crop_size = param_.crop_size();
        const int input_num = input_blob->num();
        const int input_channels = input_blob->channels();
        const int input_height = input_blob->height();
        const int input_width = input_blob->width();
        
        if (transformed_blob->count() == 0) {
            // Initialize transformed_blob with the right shape.
            if (crop_size) {
                transformed_blob->Reshape(input_num, input_channels,
                                          crop_size, crop_size);
            } else {
                transformed_blob->Reshape(input_num, input_channels,
                                          input_height, input_width);
            }
        }
        
        const int num = transformed_blob->num();
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        const int size = transformed_blob->count();
        
        CHECK_LE(input_num, num);
        CHECK_EQ(input_channels, channels);
        CHECK_GE(input_height, height);
        CHECK_GE(input_width, width);
        
        
        const Dtype scale = param_.scale();
        const bool do_mirror = param_.mirror() && Rand(2);
        const bool has_mean_file = param_.has_mean_file();
        const bool has_mean_values = mean_values_.size() > 0;
        
        int h_off = 0;
        int w_off = 0;
        if (crop_size) {
            CHECK_EQ(crop_size, height);
            CHECK_EQ(crop_size, width);
            // We only do random crop when we do training.
            if (phase_ == TRAIN) {
                h_off = Rand(input_height - crop_size + 1);
                w_off = Rand(input_width - crop_size + 1);
            } else {
                h_off = (input_height - crop_size) / 2;
                w_off = (input_width - crop_size) / 2;
            }
        } else {
            CHECK_EQ(input_height, height);
            CHECK_EQ(input_width, width);
        }
        
        Dtype* input_data = input_blob->mutable_cpu_data();
        if (has_mean_file) {
            CHECK_EQ(input_channels, data_mean_.channels());
            CHECK_EQ(input_height, data_mean_.height());
            CHECK_EQ(input_width, data_mean_.width());
            for (int n = 0; n < input_num; ++n) {
                int offset = input_blob->offset(n);
                caffe_sub(data_mean_.count(), input_data + offset,
                          data_mean_.cpu_data(), input_data + offset);
            }
        }
        
        if (has_mean_values) {
            CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
            "Specify either 1 mean_value or as many as channels: " << input_channels;
            if (mean_values_.size() == 1) {
                caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
            } else {
                for (int n = 0; n < input_num; ++n) {
                    for (int c = 0; c < input_channels; ++c) {
                        int offset = input_blob->offset(n, c);
                        caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
                                         input_data + offset);
                    }
                }
            }
        }
        
        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        
        for (int n = 0; n < input_num; ++n) {
            int top_index_n = n * channels;
            int data_index_n = n * channels;
            for (int c = 0; c < channels; ++c) {
                int top_index_c = (top_index_n + c) * height;
                int data_index_c = (data_index_n + c) * input_height + h_off;
                for (int h = 0; h < height; ++h) {
                    int top_index_h = (top_index_c + h) * width;
                    int data_index_h = (data_index_c + h) * input_width + w_off;
                    if (do_mirror) {
                        int top_index_w = top_index_h + width - 1;
                        for (int w = 0; w < width; ++w) {
                            transformed_data[top_index_w-w] = input_data[data_index_h + w];
                        }
                    } else {
                        for (int w = 0; w < width; ++w) {
                            transformed_data[top_index_h + w] = input_data[data_index_h + w];
                        }
                    }
                }
            }
        }
        if (scale != Dtype(1)) {
            DLOG(INFO) << "Scale: " << scale;
            caffe_scal(size, scale, transformed_data);
        }
    }
    
    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
        if (datum.encoded()) {
#ifdef USE_OPENCV
            CHECK(!(param_.force_color() && param_.force_gray()))
            << "cannot set both force_color and force_gray";
            cv::Mat cv_img;
            if (param_.force_color() || param_.force_gray()) {
                // If force_color then decode in color otherwise decode in gray.
                cv_img = DecodeDatumToCVMat(datum, param_.force_color());
            } else {
                cv_img = DecodeDatumToCVMatNative(datum);
            }
            // InferBlobShape using the cv::image.
            return InferBlobShape(cv_img);
#else
            LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
        }
        const int crop_size = param_.crop_size();
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        // Check dimensions.
        CHECK_GT(datum_channels, 0);
        CHECK_GE(datum_height, crop_size);
        CHECK_GE(datum_width, crop_size);
        // Build BlobShape.
        vector<int> shape(4);
        shape[0] = 1;
        shape[1] = datum_channels;
        shape[2] = (crop_size)? crop_size: datum_height;
        shape[3] = (crop_size)? crop_size: datum_width;
        return shape;
    }
    
    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(
                                                       const vector<Datum> & datum_vector) {
        const int num = datum_vector.size();
        CHECK_GT(num, 0) << "There is no datum to in the vector";
        // Use first datum in the vector to InferBlobShape.
        vector<int> shape = InferBlobShape(datum_vector[0]);
        // Adjust num to the size of the vector.
        shape[0] = num;
        return shape;
    }
    
#ifdef USE_OPENCV
    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
        const int crop_size = param_.crop_size();
        const int img_channels = cv_img.channels();
        const int img_height = cv_img.rows;
        const int img_width = cv_img.cols;
        // Check dimensions.
        CHECK_GT(img_channels, 0);
        CHECK_GE(img_height, crop_size);
        CHECK_GE(img_width, crop_size);
        // Build BlobShape.
        vector<int> shape(4);
        shape[0] = 1;
        shape[1] = img_channels;
        shape[2] = (crop_size)? crop_size: img_height;
        shape[3] = (crop_size)? crop_size: img_width;
        return shape;
    }
    
    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(const vector<cv::Mat> & mat_vector) {
        const int num = mat_vector.size();
        CHECK_GT(num, 0) << "There is no cv_img to in the vector";
        // Use first cv_img in the vector to InferBlobShape.
        vector<int> shape = InferBlobShape(mat_vector[0]);
        // Adjust num to the size of the vector.
        shape[0] = num;
        return shape;
    }
#endif  // USE_OPENCV
    
    template <typename Dtype>
    void DataTransformer<Dtype>::InitRand() {
        const bool needs_rand = param_.mirror() ||
        (phase_ == TRAIN && param_.crop_size());
        if (needs_rand) {
            const unsigned int rng_seed = caffe_rng_rand();
            rng_.reset(new Caffe::RNG(rng_seed));
        } else {
            rng_.reset();
        }
    }
    
    template <typename Dtype>
    int DataTransformer<Dtype>::Rand(int n) {
        CHECK(rng_);
        CHECK_GT(n, 0);
        caffe::rng_t* rng =
        static_cast<caffe::rng_t*>(rng_->generator());
        return ((*rng)() % n);
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::SegTransform(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label) {
        const int label_channels = transformed_label->channels();
        const int data_channels = transformed_data->channels();
        CHECK_EQ(datum.channels(), label_channels + data_channels);
        CHECK_EQ(transformed_data->width(), transformed_label->width());
        CHECK_EQ(transformed_data->height(), transformed_label->height());
        CHECK_LE(transformed_data->height(), datum.height());
        CHECK_LE(transformed_data->width(), datum.width());
        
        Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
        Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
        
        return SegTransform(datum, transformed_data_pointer, transformed_label_pointer);
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::SegTransform(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label) {
        
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        CHECK_GT(datum_channels, 0);
        const int crop_size = param_.crop_size();
        const int src_offset = datum_width * datum_height;
        const int dst_offset = crop_size * crop_size;
        const bool do_mirror = param_.mirror() && Rand(2) && (phase_ == TRAIN);
        int h_off,w_off;
        if (phase_ == TRAIN) {
            h_off = Rand(datum_height - crop_size + 1);
            w_off = Rand(datum_width - crop_size + 1);
        } else {
            h_off = (datum_height - crop_size) / 2;
            w_off = (datum_width - crop_size) / 2;
        }
        cv::Rect box(w_off,h_off,crop_size,crop_size);
        for (int i = 0; i < datum_channels; i++) {
            cv::Mat image = grayImageFromDatum(datum, i*src_offset);
            cv::Point ul_point;
            cv::Mat crop_image;
            cv::Rect square_box(w_off,h_off,crop_size,crop_size);
            imcrop(image, crop_image, square_box, ul_point, true, 127);
            if (do_mirror)
                cv::flip(crop_image, image, 1);
            else
                image = crop_image;
            if (i == datum_channels-1)
                CopyToDatum(transformed_label, image, 0.0f, 1.0f);
            else
                CopyToDatum(transformed_data + i*dst_offset, image, 127.0f, 1.0f/255.0f);
        }
    }
    
// GOT stuff
    
    template<typename Dtype>
    void DataTransformer<Dtype>::ReadGOTLabelData(GOTLabelData& label_data, const string& data, int offset, int image_width) {
        int height, width;
        DecodeFloats(data, offset,   &height, 1);
        DecodeFloats(data, offset+4, &width,  1);
        label_data.img_size = cv::Size(width, height);
        DecodeFloats(data, offset+image_width, &label_data.num_objects, 1);
        label_data.boxes.resize(label_data.num_objects);
        
        for(int i=0; i<label_data.num_objects; i++){
            int offset2 = offset+image_width*(i+2);
            DecodeFloats(data, offset2,    &label_data.boxes[i].first[0], 1);
            DecodeFloats(data, offset2+4,  &label_data.boxes[i].first[1], 1);
            DecodeFloats(data, offset2+8,  &label_data.boxes[i].first[2], 1);
            DecodeFloats(data, offset2+12, &label_data.boxes[i].first[3], 1);
            
            DecodeFloats(data, offset2+16, &label_data.boxes[i].second[0], 1);
            DecodeFloats(data, offset2+20, &label_data.boxes[i].second[1], 1);
            DecodeFloats(data, offset2+24, &label_data.boxes[i].second[2], 1);
            DecodeFloats(data, offset2+28, &label_data.boxes[i].second[3], 1);
        }
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::GOTAugment(GOTLabelData& label_data) {
        if (phase_ == TRAIN) {
            if (10*(1.0f-param_.perturb_frequency()) > Rand(10)) return;
            int rand_num = param_.corner_perturb_ratio()*200;
            int rand_num_half = rand_num/2;
            for (int i = 0; i < label_data.boxes.size(); i++) {
                std::pair<cv::Vec4f,cv::Vec4f>& points = label_data.boxes[i];
                float dim_x = std::abs(points.second[0] - points.second[2]);
                float dim_y = std::abs(points.second[1] - points.second[3]);
                points.second[0] += (Rand(rand_num)-rand_num_half)*0.01f*dim_x;
                points.second[1] += (Rand(rand_num)-rand_num_half)*0.01f*dim_y;
                points.second[2] += (Rand(rand_num)-rand_num_half)*0.01f*dim_x;
                points.second[3] += (Rand(rand_num)-rand_num_half)*0.01f*dim_y;
            }
        }
    }
    
    template<typename Dtype>
    cv::Mat DataTransformer<Dtype>::grayImageFromDatum(const Datum& datum, int offset) {
        const string& data = datum.data();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        
        cv::Mat img = cv::Mat::zeros(datum_height, datum_width, CV_8UC1);
        int count = offset;
        for (int i = 0; i < img.rows; ++i) {
            uchar* row_ptr = img.ptr<uchar>(i);
            for (int j = 0; j < img.cols; ++j) {
                row_ptr[j] = static_cast<uint8_t>(data[count++]);
            }
        }
        return img;
    }
    
    template<typename Dtype>
    float DataTransformer<Dtype>::GOTTransform(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label) {
        
        const int datum_channels = datum.channels();
        const int im_channels = transformed_data->channels();
        const int im_num = transformed_data->num();
        const int lb_num = transformed_label->num();
        CHECK_EQ(datum_channels, 3);
//        CHECK_EQ(im_num, lb_num);
        CHECK_GE(im_num, 1);
        
        Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
        Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
        
        return GOTTransform(datum, transformed_data_pointer, transformed_label_pointer);
    }
    
    template<typename Dtype>
    float DataTransformer<Dtype>::GOTTransform(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label) {
        GOTLabelData label_data;
        
        const string& data = datum.data();
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        CHECK_GT(datum_channels, 0);
        // read current image
        int offset = datum_width * datum_height;
        cv::Mat cur_img = grayImageFromDatum(datum, 0);
        cv::Mat prev_img = grayImageFromDatum(datum, offset);
        //cv::imwrite("cur.jpg", cur_img);
        //cv::imwrite("prev.jpg", prev_img);
        ReadGOTLabelData(label_data, data, 2*offset, datum_width);
        // which objec to track
        int obj_index = 0;;
        if (phase_ == TRAIN)
            obj_index = Rand(label_data.num_objects);
            
        // augment bounding boxes by perturbing it
        GOTAugment(label_data);
        cv::Vec4f minmax_prev = label_data.boxes[obj_index].second;
        cv::Vec4f minmax_cur = label_data.boxes[obj_index].first;
        cv::Rect box = vec2rect<int>(minmax_prev);
        cv::Rect square_box = make_square(box, param_.crop_margin());
        // crop images according to bounding boxes
        cv::Mat cropped_cur_img, cropped_prev_img;
        cv::Point ul_point;
        imcrop(cur_img,  cropped_cur_img,  square_box, ul_point, true, 127);
        imcrop(prev_img, cropped_prev_img, square_box, ul_point, true, 127);
        unsigned int crop_size = param_.crop_size();
        float scale = static_cast<float>(crop_size)/cropped_cur_img.rows;
        minmax_prev[0] -= ul_point.x;
        minmax_prev[1] -= ul_point.y;
        minmax_prev[2] -= ul_point.x;
        minmax_prev[3] -= ul_point.y;
        minmax_prev *= scale;
        
        minmax_cur[0] -= ul_point.x;
        minmax_cur[1] -= ul_point.y;
        minmax_cur[2] -= ul_point.x;
        minmax_cur[3] -= ul_point.y;
        minmax_cur *= scale;
        // resize image
        cv::Mat resized_cur_image, resized_prev_image;
        cv::resize(cropped_cur_img,  resized_cur_image,  cv::Size(crop_size,crop_size));
        cv::resize(cropped_prev_img, resized_prev_image, cv::Size(crop_size,crop_size));
        // create binary mask
//        cv::Mat binary_mask = cv::Mat::zeros(crop_size,crop_size,CV_8UC1);
//        box = vec2rect(minmax_prev);
//        const std::vector<cv::Point>& corners = corners_from_rect(box);
//        cv::fillConvexPoly(binary_mask, &corners[0], 4, cv::Scalar(255));
//        if (param_.blur_mask()) {
//            cv::Mat blurred_mask;
//            cv::GaussianBlur(binary_mask, blurred_mask, cv::Size(param_.blur_winsize(),param_.blur_winsize()), param_.blur_sigma());
//            binary_mask = blurred_mask;
//        }
        // add mirroring
        const bool do_mirror = param_.mirror() && Rand(2) && (phase_ == TRAIN);
        if (do_mirror) {
            cv::Mat flipped_cur_image, flipped_prev_image, flipped_binary_mask;
            cv::flip(resized_cur_image, flipped_cur_image, 1);
            cv::flip(resized_prev_image, flipped_prev_image, 1);
//            cv::flip(binary_mask, flipped_binary_mask, 1);
            resized_cur_image = flipped_cur_image;
            resized_prev_image = flipped_prev_image;
//            binary_mask = flipped_binary_mask;
            float orig_ul_x = minmax_prev[0];
            minmax_prev[0] = crop_size - minmax_prev[2];
            minmax_prev[2] = crop_size - orig_ul_x;
            
            orig_ul_x = minmax_cur[0];
            minmax_cur[0] = crop_size - minmax_cur[2];
            minmax_cur[2] = crop_size - orig_ul_x;
        }
        // copy back to datum
        offset = crop_size*crop_size;
        if (param_.use_gradient()) {
            cv::Mat It = resized_cur_image - resized_prev_image;
            cv::Mat Ix, Iy;
            cv::Sobel(resized_cur_image, Ix, resized_cur_image.depth(), 1, 0);
            cv::Sobel(resized_cur_image, Iy, resized_cur_image.depth(), 0, 1);
            
            CopyToDatum(transformed_data, It, 0.0f, 1.0f/255.0f);
            CopyToDatum(transformed_data + offset, Ix, 0.0f, 1.0f/255.0f);
            CopyToDatum(transformed_data + 2*offset, Iy, 0.f, 1.0/255.f);
//            CopyToDatum(transformed_data + 3*offset, binary_mask, 0.f, 1.0/255.f);
        } else {
            // current frame in grayscale
            CopyToDatum(transformed_data, resized_cur_image, 127.0f, 1.0f/255.0f);
            // previous frame in grayscale
            CopyToDatum(transformed_data + offset, resized_prev_image, 127.0f, 1.0f/255.0f);
            // binary mask
//            CopyToDatum(transformed_data + offset + offset, binary_mask, 0.f, 1.0/255.f);
        }
        // label: two corners of bounding box  + displacement of bounding boxes
        transformed_label[4] = minmax_prev[0];
        transformed_label[5] = minmax_prev[1];
        transformed_label[6] = minmax_prev[2];
        transformed_label[7] = minmax_prev[3];
        //const float inv_crop_size = 1.0f/crop_size;
        const float inv_crop_size = 1.0f;
        transformed_label[0] = (minmax_cur[0]-minmax_prev[0])*inv_crop_size;
        transformed_label[1] = (minmax_cur[1]-minmax_prev[1])*inv_crop_size;
        transformed_label[2] = (minmax_cur[2]-minmax_prev[2])*inv_crop_size;
        transformed_label[3] = (minmax_cur[3]-minmax_prev[3])*inv_crop_size;
        return transformed_label[0]*transformed_label[0] + transformed_label[1]*transformed_label[1] +
            transformed_label[2]*transformed_label[2] +transformed_label[3]*transformed_label[3];
    }
    
    template<typename Dtype>
    void DataTransformer<Dtype>::CopyToDatum(Dtype* data, const cv::Mat& mat, Dtype mean, Dtype div) {
        int count = 0;
        for (int i = 0; i < mat.rows; ++i) {
            const uchar* row_ptr = mat.ptr<uchar>(i);
            for (int j = 0; j < mat.cols; ++j) {
                data[count++] = (row_ptr[j] - mean)*div;
            }
        }
    }
    
// end of GOT stuff
    
    INSTANTIATE_CLASS(DataTransformer);
    
}  // namespace caffe
