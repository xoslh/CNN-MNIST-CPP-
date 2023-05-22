#pragma once

struct gradient_t
{
	float grad,oldgrad;
	gradient_t(): grad(0), oldgrad(0) {}
};

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

static float update_weight( float w, gradient_t& grad, float multp = 1 )
{
	w -= LEARNING_RATE  * (grad.grad + grad.oldgrad * MOMENTUM) * multp + LEARNING_RATE * WEIGHT_DECAY * w;
	return w;
}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}