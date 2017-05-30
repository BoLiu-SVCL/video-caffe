#include <cfloat>

#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/video_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void VideoPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  VideoPoolingParameter video_pool_param = this->layer_param_.video_pooling_param();
  CHECK_GT(video_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(video_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = video_pool_param.pooled_h();
  pooled_width_ = video_pool_param.pooled_w();
  spatial_scale_ = video_pool_param.spatial_scale();
  pad_ratio_ = video_pool_param.pad_ratio();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void VideoPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(bottom[0]->shape(0) == bottom[1]->shape(0))
		<< "Batch size must be consistant";
	batch_size_ = bottom[0]->shape(0);
	frames_ = bottom[0]->shape(1);
	pooled_parts_ = bottom[1]->shape(1);
  channels_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
	vector<int> top_shape;
	top_shape.push_back(batch_size_);
	top_shape.push_back(pooled_parts_);
	top_shape.push_back(frames_);
	top_shape.push_back(channels_);
	top_shape.push_back(pooled_height_);
	top_shape.push_back(pooled_width_);
  top[0]->Reshape(top_shape);
  max_idx_.Reshape(top_shape);
}

template <typename Dtype>
void VideoPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
	int top_count = top[0]->count();

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

	for (int b = 0; b < batch_size_; b++) {
		for (int pp = 0; pp < pooled_parts_; pp++) {

			// padding
			Dtype pad_w, pad_h;
			pad_w = (bottom_rois[2] - bottom_rois[0] + 1) * pad_ratio_;
			pad_h = (bottom_rois[3] - bottom_rois[1] + 1) * pad_ratio_;
			int roi_start_w = round((bottom_rois[0] - pad_w) * spatial_scale_);
			int roi_start_h = round((bottom_rois[1] - pad_h) * spatial_scale_);
			int roi_end_w = round((bottom_rois[2] + pad_w) * spatial_scale_);
			int roi_end_h = round((bottom_rois[3] + pad_h) * spatial_scale_);

			int roi_height = max(roi_end_h - roi_start_h + 1, 1);
			int roi_width = max(roi_end_w - roi_start_w + 1, 1);
			const Dtype bin_size_h = static_cast<Dtype>(roi_height)
				                       / static_cast<Dtype>(pooled_height_);
			const Dtype bin_size_w = static_cast<Dtype>(roi_width)
				                       / static_cast<Dtype>(pooled_width_);

 	    const Dtype* batch_data = bottom_data + bottom[0]->offset(b);

			for (int f = 0; f < frames_; f++) {
				for (int c = 0; c < channels_; c++) {
					for (int ph = 0; ph < pooled_height_; ph++) {
						for (int pw = 0; pw < pooled_width_; pw++) {
							// Compute pooling region for this output unit:
          		//  start (included) = floor(ph * roi_height / pooled_height_)
          		//  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          		int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
          		                                    * bin_size_h));
          		int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
          		                                    * bin_size_w));
          		int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
          		                                 * bin_size_h));
          		int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
          		                                 * bin_size_w));

          		hstart = min(max(hstart + roi_start_h, 0), height_);
          		hend = min(max(hend + roi_start_h, 0), height_);
          		wstart = min(max(wstart + roi_start_w, 0), width_);
          		wend = min(max(wend + roi_start_w, 0), width_);

          		bool is_empty = (hend <= hstart) || (wend <= wstart);

          		const int pool_index = ph * pooled_width_ + pw;
          		if (is_empty) {
          		  top_data[pool_index] = 0;
          		  argmax_data[pool_index] = -1;
          		}

          		for (int h = hstart; h < hend; ++h) {
          		  for (int w = wstart; w < wend; ++w) {
          		    const int index = h * width_ + w;
          		    if (batch_data[index] > top_data[pool_index]) {
          		      top_data[pool_index] = batch_data[index];
          		      argmax_data[pool_index] = index;
          		    }
          		  }
          		}
						}
					}
					batch_data += bottom[0]->offset(0, 0, 1);
					top_data += top[0]->offset(0, 0, 0, 1);
		      argmax_data += max_idx_.offset(0, 0, 0, 1);
				}
			}

			bottom_rois += bottom[1]->offset(0, 1);
		}
	}
}

template <typename Dtype>
void VideoPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(VideoPoolingLayer);
#endif

INSTANTIATE_CLASS(VideoPoolingLayer);
REGISTER_LAYER_CLASS(VideoPooling);

}  // namespace caffe
