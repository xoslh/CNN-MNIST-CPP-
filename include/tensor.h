#pragma once
#include <cassert>
#include <vector>
#include <string.h>

struct point_t
{
	int x, y, z;
};
using tdsize = point_t;

template<typename T>
struct tensor_t
{
	T * data;

	tdsize size;

	tensor_t( int _x, int _y, int _z )
	{
		data = new T[_x * _y * _z];
		size.x = _x, size.y = _y, size.z = _z;
	}

	tensor_t( const tensor_t& other )
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy( this->data, other.data, other.size.x *other.size.y *other.size.z * sizeof( T ) );
		this->size = other.size;
	}

	tensor_t<T> operator+( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}

	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z )
	{
		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	~tensor_t()
	{
		delete[] data;
	}
};