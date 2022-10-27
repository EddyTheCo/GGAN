#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>
#ifdef USE_YAML
#include<yaml-cpp/yaml.h>
#endif


namespace custom_models{
	class GGANImpl : public torch::nn::Module {
		public:
			GGANImpl(const std::vector<size_t> &layers,double leaky_relu_m=0.2):input_size(layers.front()),output_size(layers.back()),leaky_relu_(leaky_relu_m)
		{

			for(auto i=0;i<layers.size()-1;i++)
			{
				module_cont.push_back(register_module(("fc"+std::to_string(i)).c_str(),
							torch::nn::Linear(layers[i],layers[i+1])));
			}
		}
#ifdef USE_YAML
			GGANImpl(YAML::Node config):GGANImpl(config["layers"].as<std::vector<size_t>>(),config["leaky relu"].as<double>()){std::cout<<config<<std::endl;};
#endif
			void update(void)const{};
			torch::Tensor forward(torch::Tensor x) {
				if(x.sizes().size()>2)x=x.view({x.size(0),x.numel()/x.size(0)});
				for(auto i=0;i<module_cont.size();i++)
				{
					if(i<module_cont.size()-1)
					{
						x = torch::leaky_relu(module_cont[i](x),leaky_relu_);
					}
					else
					{
						x = torch::tanh(module_cont[i](x));
					}
				}

				return x;
			}
			const int64_t input_size,output_size;
			const double leaky_relu_;
		private:
			std::vector<torch::nn::Linear> module_cont;

	};
	TORCH_MODULE(GGAN);


};
