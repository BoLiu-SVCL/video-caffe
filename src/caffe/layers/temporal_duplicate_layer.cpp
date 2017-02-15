#include <vector>

#include "caffe/layers/temporal_duplicate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TemporalDuplicateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1) << "TemporalDuplicate Layer only accepts 1 bottom blob";
	CHECK_EQ(top.size(), 1) << "TemporalDuplicate Layer only accepts 1 top blob";
	const DuplicateParameter& duplicate_param = this->layer_param_.duplicate_param();
	count_ = bottom[0]->count();
	num_duplicate_ = duplicate_param.num();
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape;
	top_shape.clear();
	top_shape.push_back(num_duplicate_);
	for (int i = 0; i < bottom_shape.size(); i++) {
		top_shape.push_back(bottom_shape[i]);
	}
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void TemporalDuplicateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < num_duplicate_; i++) {
		caffe_copy(count_, bottom[0]->cpu_data(),
				top[0]->mutable_cpu_data() + i * count_);
	}
}

template <typename Dtype>
void TemporalDuplicateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	vector<int> top_shape = top[0]->shape();
	if (top_shape[0] == 1) {
		caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
		return;
	}
	caffe_add(count_, top[0]->cpu_diff(), top[0]->cpu_diff() + count_,
			bottom[0]->mutable_cpu_diff());
	for (int i = 2; i < top_shape[0]; i++) {
		caffe_axpy(count_, Dtype(1.), top[0]->cpu_diff() + i * count_,
				bottom[0]->mutable_cpu_diff());
	}
}


#ifdef CPU_ONLY
STUB_GPU(TemporalDuplicateLayer);
#endif

INSTANTIATE_CLASS(TemporalDuplicateLayer);
REGISTER_LAYER_CLASS(TemporalDuplicate);

} // namespace caffe
