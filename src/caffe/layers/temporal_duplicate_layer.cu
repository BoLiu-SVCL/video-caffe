#include <vector>

#include "caffe/layers/temporal_duplicate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TemporalDuplicateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < num_duplicate_; i++) {
		caffe_copy(count_, bottom[0]->gpu_data(),
				top[0]->mutable_gpu_data() + i * count_);
	}
}

template <typename Dtype>
void TemporalDuplicateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	vector<int> top_shape = top[0]->shape();
	if (top_shape[0] == 1) {
		caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		return;
	}
	caffe_gpu_add(count_, top[0]->gpu_diff(), top[0]->gpu_diff() + count_,
			bottom[0]->mutable_gpu_diff());
	for (int i = 2; i < top_shape[0]; i++) {
		caffe_gpu_axpy(count_, Dtype(1.), top[0]->gpu_diff() + i * count_,
				bottom[0]->mutable_gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(TemporalDuplicateLayer);

}  // namespace caffe
