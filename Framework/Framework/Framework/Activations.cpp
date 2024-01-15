#include "Activations.h"
#include <Math.h>

float Activations::sigmoidf(const float x)
{
	return (float)1.0f / (1.0f + expf(-x));
}
