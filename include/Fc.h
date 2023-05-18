#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "LayerBase.h"


class fc_layer_t: public layer_t
{
public:
	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_t( tdsize in_size, int out_size )
		:
		layer_t( layer_type::fc, in_size, {out_size, 1, 1} ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		input = std::vector<float>( out_size );
		gradients = std::vector<gradient_t>( out_size );

		int N = in_size.x * in_size.y * in_size.z;

		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
				weights( h, i, 0 ) = 2.19722f / N * rand() / float( RAND_MAX );
	}

	//铺平后的id
	int id( int x, int y, int z ) 
	{
		return z * (in.size.x * in.size.y) + y * (in.size.x) + x;
	}

	//激活函数： return tanhf(x)
	float activator_function( float x ) 
	{
		return (float) 1.0f / (1.0f + exp( -x ));
	}

	//导回来
	float activator_derivative( float x )
	{
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	void activate( tensor_t<float>& in ) override
	{
		this->in = in;
		float now = 0;
		for ( int cnt = 0; cnt < out.size.x; cnt++, now=0 )
		{
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						now += in( i, j, z ) * weights( id( i, j, z ) , cnt, 0 );
			input[cnt] = now;
			out( cnt, 0, 0 ) = activator_function( now );
		}
	}

	void fix_weights() override
	{
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int k = 0; k < in.size.z; k++ )
					{
						float& w = weights( id( i, j, k ), n, 0 );
						w = update_weight( w, grad, in( i, j, k ) );
					}

			update_gradient( grad );
		}
	}

	void calc_grads( tensor_t<float>& grad_next_layer ) override
	{
		for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int k = 0; k < in.size.z; k++ ) grads_in( i, j, k ) = 0; 
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( input[n] );
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int k = 0; k < in.size.z; k++ )
						grads_in( i, j, k ) += grad.grad * weights( id( i, j, k ), n, 0 );
		}
	}

};
