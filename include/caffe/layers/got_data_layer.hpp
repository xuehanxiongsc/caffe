#ifndef CAFFE_GOT_DATA_LAYER_HPP
#define CAFFE_GOT_DATA_LAYER_HPP

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    
    template <typename Dtype>
    class GOTDataLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit GOTDataLayer(const LayerParameter& param);
        virtual ~GOTDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
        virtual inline bool ShareInParallel() const { return false; }
        virtual inline const char* type() const { return "GOTData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }
        
    protected:
        virtual void load_batch(Batch<Dtype>* batch);
        //void generateLabelMap();
        
        DataReader reader_;
        Blob<Dtype> transformed_label_; // add another blob
        //Blob<Dtype> transformed_label_all_; // all peaks, including others
    };
    
}

#endif