#include <string>
#include <fstream>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;

void outputNet(std::string outfile_name, NetParameter& net_param, 
		std::string layer_name1, std::string layer_name2, std::string layer_name3);
void outputLayer(std::ofstream& outfile, LayerParameter* layer_param,
		std::string layer_name1, std::string layer_name2, std::string layer_name3);
void reshape1(BlobProto* blob);
void reshape2(BlobProto* blob);
void reshape3(BlobProto* blob);

int main(int argc, char* argv[]) {
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;
	if (argc != 7) {
		LOG(INFO) << "Need 6 params.";
		return 1;
	}

	std::string proto_name(argv[1]);
	std::string out_name(argv[2]);
	std::string out_net_name(argv[3]);
	std::string layer_name1(argv[4]);
	std::string layer_name2(argv[5]);
	std::string layer_name3(argv[6]);

	NetParameter net_param;
	CHECK(ReadProtoFromBinaryFile(proto_name, &net_param))
		<< "Failed to parse NetParameter file: " << proto_name;
	outputNet(out_name, net_param, layer_name1, layer_name2, layer_name3);

	WriteProtoToBinaryFile(net_param, out_net_name);

	return 0;
}

void outputNet(std::string outfile_name, NetParameter& net_param,
		std::string layer_name1, std::string layer_name2, std::string layer_name3) {
	std::ofstream outfile(outfile_name.c_str());
	if (net_param.has_name()) {
		outfile << "name: \"" << net_param.name() << "\"" << std::endl;
	}
	int num_layer = net_param.layer_size();
	for (int i = 0; i < num_layer; i++) {
		outputLayer(outfile, net_param.mutable_layer(i),
				layer_name1, layer_name2, layer_name3);
	}
	outfile.close();
}

void outputLayer(std::ofstream& outfile, LayerParameter* layer_param,
		std::string layer_name1, std::string layer_name2, std::string layer_name3) {
	int num_blobs = layer_param->blobs_size();
	if (num_blobs == 0) {
		outfile << "layer: \"" << layer_param->name() << std::endl;
	}
	else {
		outfile << "layer {" << std::endl;
		outfile << "  name: \"" << layer_param->name() << std::endl;
		outfile << "  type: \"" << layer_param->type() << std::endl;
		if (layer_param->name() == layer_name1) {
  		reshape1(layer_param->mutable_blobs(0));
	  }
		if (layer_param->name() == layer_name2) {
  		reshape2(layer_param->mutable_blobs(0));
	  }
		if (layer_param->name() == layer_name3) {
  		reshape3(layer_param->mutable_blobs(0));
	  }
		for (int i = 0; i < num_blobs; i++) {
			int shape_length = layer_param->blobs(i).shape().dim_size();
			outfile << "  shape: ";
			for (int j = 0; j < shape_length; j++) {
				outfile << layer_param->blobs(i).shape().dim(j) << ", ";
			}
			outfile << std::endl;
		}
		outfile << "}" << std::endl;
	}
}

void reshape1(BlobProto* blob) {
	blob->clear_shape();
	blob->mutable_shape()->add_dim(4096);
	blob->mutable_shape()->add_dim(512);
	blob->mutable_shape()->add_dim(4);
	blob->mutable_shape()->add_dim(4);
}

void reshape2(BlobProto* blob) {
	blob->mutable_shape()->add_dim(1);
	blob->mutable_shape()->add_dim(1);
}

void reshape3(BlobProto* blob) {
	blob->mutable_shape()->add_dim(1);
	blob->mutable_shape()->add_dim(1);
}
