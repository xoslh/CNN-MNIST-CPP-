#pragma once
#include "LayerBase.h"

class conv_layer_t: public layer_t
{
public:
	std::vector<tensor_t<float>> filters;
	std::vector<tensor_t<gradient_t>> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;
	
	conv_layer_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size )
		:
		layer_t( layer_type::conv, in_size, {(in_size.x - extend_filter) / stride + 1, (in_size.y - extend_filter) / stride + 1, number_filters} )
	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		for ( int n = 0; n < number_filters; n++ )
		{
			tensor_t<float> neww( extend_filter, extend_filter, in_size.z );
			int N = extend_filter * extend_filter * in_size.z;
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in_size.z; z++ )
						neww( i, j, z ) = 1.0f / N * rand() / 2147483647.0;
			filters.push_back( neww );
		}
		for ( int i = 0; i < number_filters; i++ )
		{
			tensor_t<gradient_t> t( extend_filter, extend_filter, in_size.z );
			filter_grads.push_back( t );
		}
	}

	point_t map_to_input( point_t out, int z )
	{
		return {out.x * stride, out.y * stride, z};
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int GET_R( float f, int max, bool lim_min )
	{
		if ( f <= 0 ) return 0;
		max -= 1;
		if ( f >= max ) return max;
		if ( lim_min ) return ceil( f );
		else return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x, b = y;
		return { GET_R( (a - extend_filter + 1) / stride, out.size.x, true ), GET_R( (b - extend_filter + 1) / stride, out.size.y, true ), 0, GET_R( a / stride, out.size.x, false ), GET_R( b / stride, out.size.y, false ), (int)out.size.z - 1, };
	}

	void activate( tensor_t<float>& in ) override
	{
		this->in = in;
		for ( int n = 0; n < filters.size(); n++ )
		{
			tensor_t<float>& filter = filters[n];
			for ( int x = 0; x < out.size.x; x++ )
				for ( int y = 0; y < out.size.y; y++ )
				{
					point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float sum = 0;
					for ( int i = 0; i < extend_filter; i++ )
						for ( int j = 0; j < extend_filter; j++ )
							for ( int z = 0; z < in.size.z; z++ )
								sum += filter( i, j, z ) * in( mapped.x + i, mapped.y + j, z );
					out( x, y, n ) = sum;
				}
		}
	}

	void fix_weights() override
	{
		for ( int a = 0; a < filters.size(); a++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						float& w = filters[a].get( i, j, z );
						gradient_t& grad = filter_grads[a]( i, j, z );
						w = update_weight( w, grad );
						update_gradient( grad );
					}
	}

	void calc_grads( tensor_t<float>& grad_next_layer ) override
	{

		for ( int k = 0; k < filter_grads.size(); k++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads[k].get( i, j, z ).grad = 0;
		for ( int x = 0; x < in.size.x; x++ )
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					//out[i, j, k] -> in[x, y, z] 有贡献的位置
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;
							for ( int k = rn.min_z; k <= rn.max_z; k++ )
							{
								//贡献的系数 -> 第k个核作用 out[ i, j, k] 对应的in区域，in[x, y, z] 的系数
								int K = filters[k]( x - minx, y - miny, z );
								//系数 * 偏导
								sum_error += K * grad_next_layer( i, j, k );
								//卷积核 grad 同理 
								filter_grads[k]( x - minx, y - miny, z ).grad += in( x, y, z ) * grad_next_layer( i, j, k );
							}
						}
					}
					//更新
					grads_in( x, y, z ) = sum_error;
				}
			}
	}
};