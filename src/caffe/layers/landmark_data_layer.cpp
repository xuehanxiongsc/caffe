#include "caffe/layers/landmark_data_layer.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <string>

#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
    
template <typename Dtype>
LandmarkDataLayer<Dtype>::LandmarkDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param), reader_(param) {
}

template <typename Dtype>
LandmarkDataLayer<Dtype>::~LandmarkDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void LandmarkDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());
  // the last channel stores label info
  const int datum_channel = datum.channels();
  const int data_channel = datum_channel;
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int num_landmarks = this->layer_param_.transform_param().num_landmarks();
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int label_stride = this->layer_param_.transform_param().label_stride();
  if (crop_size > 0) {
    // top[0] stores image
    top[0]->Reshape(batch_size, data_channel, crop_size, crop_size);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, data_channel,
                                       crop_size, crop_size);
    }
    this->transformed_data_.Reshape(batch_size, data_channel, crop_size,
                                    crop_size);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  // top[1] stores label info
  if (this->output_labels_) {
    int label_width = crop_size/label_stride;
    int label_height = crop_size/label_stride;
    int label_channel = (num_landmarks+1)*2;
    top[1]->Reshape(batch_size, label_channel, label_height, label_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, label_channel, label_height,
                                        label_width);
    }
    this->transformed_label_.Reshape(batch_size, label_channel, label_height,
                                     label_width);
    LOG(INFO) << "output label size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void LandmarkDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    
  CPUTimer batch_timer;
  batch_timer.Start();
  double deque_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  
  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  
  if (this->output_labels_) {
      top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a blob
      timer.Start();
      Datum& datum = *(reader_.full().pop("Waiting for data"));
      deque_time += timer.MicroSeconds();
      
      // Apply data transformations (mirror, scale, crop...)
      timer.Start();
      const int offset_data = batch->data_.offset(item_id);
      const int offset_label = batch->label_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset_data);
      this->transformed_label_.set_cpu_data(top_label + offset_label);
      this->data_transformer_->LandmarkTransform(
          datum,
          &(this->transformed_data_),
          &(this->transformed_label_));
      trans_time += timer.MicroSeconds();
      reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
    
#ifdef BENCHMARK_DATA
  LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms.";
  LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
#endif
}

INSTANTIATE_CLASS(LandmarkDataLayer);
REGISTER_LAYER_CLASS(LandmarkData);
  
}  // namespace caffe
