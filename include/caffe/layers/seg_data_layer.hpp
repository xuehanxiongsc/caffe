#ifndef CAFFE_SEG_DATA_LAYER_HPP
#define CAFFE_SEG_DATA_LAYER_HPP

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    
    template <typename Dtype>
    class SegDataLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit SegDataLayer(const LayerParameter& param);
        virtual ~SegDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
        virtual inline bool ShareInParallel() const { return false; }
        virtual inline const char* type() const { return "SegData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }
        
    protected:
        virtual void load_batch(Batch<Dtype>* batch);
        DataReader reader_;
        Blob<Dtype> transformed_label_; // add another blob
    };
    
}

#endif