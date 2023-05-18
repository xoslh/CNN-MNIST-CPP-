#pragma once
#include "LayerBase.h"

class pool_layer_t: public layer_t
{
public:
	uint16_t stride;
	uint16_t extend_filter;

	pool_layer_t( uint16_t stride_, uint16_t extend_filter_, tdsize in_size )
		:
		stride(stride_),
		extend_filter(extend_filter_),
		layer_t( layer_type::pool, in_size, {(in_size.x - extend_filter_) / stride_ + 1, (in_size.y - extend_filter_) / stride_ + 1, in_size.z} )
	{
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
		for ( int x = 0; x < out.size.x; x++ )
			for ( int y = 0; y < out.size.y; y++ )
				for ( int z = 0; z < out.size.z; z++ )
				{
					point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float maxx = -FLT_MAX;
					for ( int i = 0; i < extend_filter; i++ )
						for ( int j = 0; j < extend_filter; j++ )
						{
							float v = in( mapped.x + i, mapped.y + j, z );
							if ( v > maxx ) maxx = v;
						}
					out( x, y, z ) = maxx;
				}
	}

	void fix_weights() override {}

	void calc_grads( tensor_t<float>& grad_next_layer ) override
	{
		for ( int x = 0; x < in.size.x; x++ )
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					//out[i, j, z] 是 in[x, y, z] 可能有贡献的位置，贡献的系数是 1 或者 0 
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;
							int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
							//偏导 * 系数
							sum_error += is_max * grad_next_layer( i, j, z );
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
	}
};