#ifndef CAFFE_TEMPORAL_DUPLICATE_LAYER_HPP_
#define CAFFE_TEMPORAL_DUPLICATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Duplicate the input blobs along temporal axis, which will be added
 * in front of all other axises.
 */
template <typename Dtype>
class TemporalDuplicateLayer : public Layer<Dtype> {
	public:
		explicit TemporalDuplicateLayer(const LayerParameter& param)
			  : Layer<Dtype>(param) {}
	  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "TemporalDuplicate"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int count_;
		int num_duplicate_;
};

} // namespace caffe

#endif // CAFFE_TEMPORAL_DUPLICATE_LAYER_HPP
