#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "coord.h"
#include "memorypool.h"
#include <algorithm>
#include <numeric>
#include <iostream>

#define LargeInteger 1000000
#define Infinity 1e+10
#define Tiny 1e-10

#ifdef DEBUG
#define safe_cast dynamic_cast
#else
#define safe_cast static_cast
#endif

namespace UTILS
{

	template < class T >
	std::ostream& operator << (std::ostream& os, const std::vector<T>& v) 
	{
		os << "[";
		for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
		{
			os << *ii << (ii != v.end()-1 ? " " : "");
		}
		os << "]";
		return os;
	}

	inline double GGF(std::vector<double> utility)
	{
		int n = utility.size();
		assert(n > 0);
		std::vector<double> w(n);
		for (int i = 0; i < n; i++) {
			w[i] = 1.0 / pow(2, i);
		}
		std::sort(utility.begin(), utility.end());
		double score = 0.0;
		for (int i = 0; i < n; i++){
			score += w[i] * utility[i];
		}
		return score;
	}

	inline double WS(const std::vector<double> utility)
	{
		int n = utility.size();
		assert(n > 0);
		std::vector<double> w(n, 1.0/n);
		double score = 0.0;
		for (int i = 0; i < n; i++){
			score += w[i] * utility[i];
		}
		return score;
	}

	inline double CV(const std::vector<double> arr)
	{
		assert(arr.size() > 0);
		double sum = std::accumulate(arr.begin(), arr.end(), 0.0);
		double variance = 0.0, mean = 0.0, stdDeviation = 0.0;
		mean = sum / arr.size();
        if (mean == 0) {
            return 0.0;
        }
		for (auto& elem: arr) {
			variance += pow(elem - mean, 2);
		}
		variance /= arr.size();
		stdDeviation = sqrt(variance);
		double cv = stdDeviation / mean;
		return cv;
	}

	inline int Sign(int x)
	{
		return (x > 0) - (x < 0);
	}

	inline int Random(int max)
	{
		return rand() % max;
	}

	inline int Random(int min, int max)
	{
		return rand() % (max - min) + min;
	}

	inline double RandomDouble(double min, double max)
	{
		return (double)rand() / RAND_MAX * (max - min) + min;
	}

	inline void RandomSeed(int seed)
	{
		srand(seed);
	}

	inline bool Bernoulli(double p)
	{
		return rand() < p * RAND_MAX;
	}

	inline int sample(const std::vector<double> prob)
	{
		int n = prob.size();
		std::vector<double> pre_sum(n+1, 0.0);
		for (int i = 1; i <= n; i++) {
			pre_sum[i] = pre_sum[i-1] + prob[i-1];
		}
		double p = RandomDouble(0.0, 1.0);
		int idx = 0;
		for (; idx < n; idx++) {
			if (pre_sum[idx] <= p && p <= pre_sum[idx+1]) break;
		}
		if (idx >= n) {
			std::cout << p << ", " << prob << std::endl;
			idx = Random(n);
		}
		assert (idx >= 0 && idx < n);
		return idx;
	}

	inline bool Near(double x, double y, double tol)
	{
		return fabs(x - y) <= tol;
	}

	inline bool CheckFlag(int flags, int bit) { return (flags & (1 << bit)) != 0; }

	inline void SetFlag(int& flags, int bit) { flags = (flags | (1 << bit)); }

	template<class T>
	inline bool Contains(std::vector<T>& vec, const T& item)
	{
		return std::find(vec.begin(), vec.end(), item) != vec.end();
	}

	void UnitTest();

}

#endif // UTILS_H
