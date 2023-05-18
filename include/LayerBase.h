#pragma once
#include "tensor.h"
#include "gradient.h"

enum class layer_type
{
	conv,
	fc,
	relu,
	pool,
	dropout_layer
};

class layer_t
{
public:
	layer_type type;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	layer_t( layer_type type_, tdsize in_size, tdsize out_size ):
		type( type_ ),
		in( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z ),
		out( out_size.x, out_size.y, out_size.z )
	{
	}
	virtual ~layer_t(){}
	virtual void activate( tensor_t<float>& in )=0;
	virtual void fix_weights()=0;
	virtual void calc_grads( tensor_t<float>& grad_next_layer )=0;
};