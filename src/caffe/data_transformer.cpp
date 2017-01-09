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
void DataTransformer<Dtype>::SegTransform(const Datum& datum,
                                          Blob<Dtype>* transformed_data,
                                          Blob<Dtype>* transformed_label) {
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
void DataTransformer<Dtype>::SegTransform(const Datum& datum,
                                          Dtype* transformed_data,
                                          Dtype* transformed_label) {
  
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
    w_off = (datum_width -crop_size) / 2;
  }
  cv::Rect box(w_off,h_off,crop_size,crop_size);
  for (int i = 0; i < datum_channels; i++) {
    cv::Mat image = SingleChannelFromDatum(datum, i*src_offset);
    cv::Point ul_point;
    cv::Mat crop_image;
    cv::Rect square_box(w_off,h_off,crop_size,crop_size);
    image(square_box).copyTo(crop_image);
    cv::Mat flip_image;
    if (do_mirror) {
      cv::flip(crop_image, flip_image, 1);
    } else {
      flip_image = crop_image;
    }
    if (i == datum_channels-1) {
      CopyToDatum(transformed_label, flip_image, 0.0f, 1.0f);
    } else {
      CopyToDatum(transformed_data + i*dst_offset, flip_image, 127.0f,
                  1.0f/255.0f);
    }
  }
}

// Landmark
template<typename Dtype>
void DataTransformer<Dtype>::ReadLandmark(const string& data, int offset,
                                          int image_width,
                                          LandmarkLabel* label_data) {
  LandmarkLabel& label_ref = *label_data;
  cv::Mat& landmarks_ref = label_ref.landmarks;
  DecodeString<int>(data, offset, &label_ref.num_objects, 1);
  DecodeString<float>(data, offset+image_width, &label_ref.center.x, 1);
  DecodeString<float>(data, offset+image_width+4, &label_ref.center.y, 1);
  int total_landmarks = label_ref.num_objects * param_.num_landmarks();
  int num_floats_per_object = param_.num_landmarks() * 3;
  landmarks_ref.create(total_landmarks,3,CV_32FC1);
  for(int i=0; i<label_ref.num_objects; i++){
    int offset2 = offset+image_width*(i+2);
    DecodeString<float>(data, offset2,
                        landmarks_ref.ptr<float>(i*param_.num_landmarks()),
                        num_floats_per_object);
  }
}

template<typename Dtype>
cv::Mat DataTransformer<Dtype>::SingleChannelFromDatum(const Datum& datum,
                                                       int offset) {
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
void DataTransformer<Dtype>::LandmarkTransform(const Datum& datum,
                                               Blob<Dtype>* transformed_data,
                                               Blob<Dtype>* transformed_label) {
  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
  LandmarkTransform(datum, transformed_data_pointer, transformed_label_pointer);
}

template<typename Dtype>
void DataTransformer<Dtype>::LandmarkTransform(const Datum& datum,
                                               Dtype* transformed_data,
                                               Dtype* transformed_label) {
  LandmarkLabel landmark_data;
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  const int bytes_per_channel = datum_width*datum_height;
  const int crop_size = param_.crop_size();
  const int label_size = crop_size/param_.label_stride();
  const int bytes_per_image_channel = crop_size*crop_size;
  const int bytes_per_label_channel = label_size*label_size;
  const int num_landmarks = param_.num_landmarks();
  const int slice_point = num_landmarks+1;
  CHECK_EQ(datum_channels, 4);
  // read landmark data from last channel
  ReadLandmark(data, (datum_channels-1)*bytes_per_channel, datum_width,
               &landmark_data);
  CHECK(landmark_data.landmarks.rows % landmark_data.num_objects == 0);
  const bool do_mirror = (phase_ == TRAIN) && param_.mirror() && Rand(2);
  float perturb_scale = (phase_ == TRAIN) ? (param_.min_scale() +
      static_cast<float>(Rand(100))/100.f *
      (param_.max_scale()-param_.min_scale())) : 1.0f;
  int perturb_x = (phase_ == TRAIN) ? (-param_.max_shift() +
      Rand(2*param_.max_shift())) : 0;
  int perturb_y = (phase_ == TRAIN) ? (-param_.max_shift() +
      Rand(2*param_.max_shift())) : 0;
  std::vector<cv::Mat> bgr_array(3);
  bgr_array[0] = SingleChannelFromDatum(datum, 0);
  bgr_array[1] = SingleChannelFromDatum(datum, bytes_per_channel);
  bgr_array[2] = SingleChannelFromDatum(datum, bytes_per_channel*2);
  cv::Mat bgr;
  cv::merge(bgr_array, bgr);
  cv::Mat crop_image;
  PerturbLandmarkData(bgr,perturb_scale,perturb_x,perturb_y,do_mirror,
                      &crop_image,&landmark_data);
  // write transformed image to data
  std::vector<cv::Mat> crop_bgr_array(3);
  cv::split(crop_image, crop_bgr_array);
  CopyToDatum(transformed_data, crop_bgr_array[0], 127.0f, 1.0f/255.0f);
  CopyToDatum(transformed_data+bytes_per_image_channel, crop_bgr_array[1],
              127.0f, 1.0f/255.0f);
  CopyToDatum(transformed_data+bytes_per_image_channel*2, crop_bgr_array[2],
              127.0f, 1.0f/255.0f);
  float crop_image_center = 0.5f*(param_.crop_size()-1.0f);
  memset(transformed_data+bytes_per_image_channel*3,0,sizeof(Dtype)*bytes_per_image_channel);
  UpdateHeatmap(transformed_data+bytes_per_image_channel*3,
                cv::Point2f(crop_image_center,crop_image_center),
                crop_size, crop_size, param_.centermap_sigma());
  // write heatmaps to label
  memset(transformed_label,0,sizeof(Dtype)*bytes_per_label_channel*(num_landmarks+1)*2);
  for (int n = 0; n < landmark_data.num_objects; n++) {
    for (int i = 0; i < num_landmarks; i++) {
      const cv::Mat& landmark_i = landmark_data.landmarks.row(
          i+n*num_landmarks);
      // if this landmark is invisible
      if (landmark_i.at<float>(2) == 0.0) continue;
      // the first slice of labels contain heatmaps from all people
      UpdateHeatmap(transformed_label+bytes_per_label_channel*i,
                    cv::Point2f(landmark_i.at<float>(0),
                                landmark_i.at<float>(1)),
                    label_size, label_size, param_.sigma(),
                    param_.label_stride());
      // the second slice of labels only contain heatmaps from the 1st person
      if (n==0) {
        UpdateHeatmap(transformed_label+bytes_per_label_channel*(i+slice_point),
                      cv::Point2f(landmark_i.at<float>(0),
                                  landmark_i.at<float>(1)),
                      label_size, label_size, param_.sigma(),
                      param_.label_stride());
      }
    }
  }
  BackgroundHeatmap(transformed_label, label_size, label_size, num_landmarks,
                    transformed_label+num_landmarks*bytes_per_label_channel);
  BackgroundHeatmap(transformed_label+slice_point*bytes_per_label_channel,
                    label_size, label_size, num_landmarks,
                    transformed_label+
                        (slice_point+num_landmarks)*bytes_per_label_channel);
}

template<typename Dtype>
void DataTransformer<Dtype>::PerturbLandmarkData(
    const cv::Mat& input_image,
    float perturbed_scale,
    int perturbed_x,
    int perturbed_y,
    bool mirror,
    cv::Mat* output_image,
    LandmarkLabel* inout_label) {
  // apply random scale perturbation
  float scale = param_.abs_scale()*perturbed_scale;
  cv::Mat resized_image;
  if (perturbed_scale > 1.0f) {
    cv::resize(input_image,resized_image,cv::Size(),scale,scale,
               CV_INTER_LINEAR);
  } else {
    cv::resize(input_image,resized_image,cv::Size(),scale,scale,
               CV_INTER_AREA);
  }
  inout_label->landmarks.colRange(0,2) *= scale;
  inout_label->center.x *= scale;
  inout_label->center.y *= scale;
  // apply random translation perturbation
  cv::Point offset;
  const int half_crop_size = param_.crop_size()/2;
  int crop_x_start = inout_label->center.x - half_crop_size + perturbed_x;
  int crop_y_start = inout_label->center.y - half_crop_size + perturbed_y;
  cv::Rect roi(crop_x_start,crop_y_start,param_.crop_size(),param_.crop_size());
  cv::Mat crop_image;
  imcrop(resized_image,roi,offset,&crop_image,true);
  inout_label->landmarks.col(0) -= offset.x;
  inout_label->landmarks.col(1) -= offset.y;
  // apply mirroring
  if (mirror) {
    Flip(crop_image, param_.flip_index().begin(), param_.flip_index().end(),
         output_image, &(inout_label->landmarks));
  } else {
    *output_image = crop_image;
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CopyToDatum(Dtype* data, const cv::Mat& mat,
                                         Dtype mean, Dtype div) {
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
