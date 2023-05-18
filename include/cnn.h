#pragma once
#include "tensor.h"
#include "gradient.h"
#include "Fc.h"
#include "Pool.h"
#include "Relu.h"
#include "Conv.h"
#include "DropOut.h"
using namespace std;

class Model
{
public:
	Model(){}
	
	void add_conv( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size )
	{
		conv_layer_t * layer = new conv_layer_t( stride, extend_filter, number_filters, in_size);
		layers.push_back( (layer_t*)layer );
	}

	void add_relu( tdsize in_size )
	{
		relu_layer_t * layer = new relu_layer_t( in_size );
		layers.push_back( (layer_t*)layer );
	}
	
	void add_pool( uint16_t stride, uint16_t extend_filter, tdsize in_size )
	{
		pool_layer_t * layer = new pool_layer_t( stride, extend_filter, in_size );
		layers.push_back( (layer_t*)layer );
	}

	void add_fc( tdsize in_size, int out_size )
	{
		fc_layer_t * layer = new fc_layer_t( in_size, 10 );
		layers.push_back( (layer_t*)layer );
	}

	tdsize& output_size()
	{
		return layers.back()->out.size;
	}

	int predict()
	{
		int ans = 0;
		for( int i = 0; i < 10; i++ )
			if( layers.back()->out( i, 0, 0 ) > layers.back()->out( ans, 0, 0 ) )
				ans = i ;
		return ans;
	}

	tensor_t<float>& predict_info()
	{
		return layers.back()->out;
	}

	void forward( tensor_t<float>& data )
	{
		for ( int i = 0; i < layers.size(); i++ )
			layers[i]->activate( i ? layers[i - 1]->out : data );
	}

	float train( tensor_t<float>& data, tensor_t<float>& label )
	{
		forward( data );
		auto res_info = layers.back()->out - label;

		for ( int i = layers.size() - 1; i >= 0; i-- )
			layers[i]->calc_grads( i < layers.size() - 1 ? layers[i + 1]->grads_in : res_info );

		for ( int i = 0; i < layers.size(); i++ ) 
			layers[i]->fix_weights();

		float err = 0;
		for ( int i = 0; i < 10; i++ )
		{	
			float x = label(i, 0, 0) - res_info(i, 0, 0);
			err += x*x ;
		}
		return sqrt(err) * 100;
	}


private:
	vector<layer_t*> layers;
};
