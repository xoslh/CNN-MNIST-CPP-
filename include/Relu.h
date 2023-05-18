#pragma once
#include "LayerBase.h"

class relu_layer_t: public layer_t
{
public:

	relu_layer_t( tdsize in_size )
		:
		layer_t( layer_type::relu, in_size, in_size ) {}

	void activate( tensor_t<float>& in ) override
	{
		this->in = in;
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
					out( i, j, z ) = in( i, j, z ) < 0 ? 0 : in( i, j, z );
	}

	void fix_weights() override {}

	void calc_grads( tensor_t<float>& grad_next_layer ) override
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
					grads_in( i, j, z ) = in( i, j, z ) < 0 ? 0 : 1 * grad_next_layer( i, j, z ) ;
	}
};