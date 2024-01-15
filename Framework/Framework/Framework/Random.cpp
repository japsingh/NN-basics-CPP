#include "Random.h"
#include "stdlib.h"

float rand_f() {
	return (float)rand() / (float)RAND_MAX;
}
