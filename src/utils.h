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
#include <cfloat>
#include <unordered_set>

#define LargeInteger 1000000
#define Infinity 1e+5
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

	inline int findNearestNeighbourIndex(const double value, const std::vector< double > &x )
	{
		double dist = DBL_MAX;
		int idx = -1;
		for ( int i = 0; i < x.size(); ++i ) {
			double newDist = value - x[i];
			if ( newDist > 0 && newDist < dist ) {
				dist = newDist;
				idx = i;
			}
		}

		return idx;
	}

	inline std::vector<double> interp1(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& x_new)
	{
		std::vector< double > y_new;
		y_new.reserve( x_new.size() );

		std::vector< double > dx, dy, slope, intercept;
		dx.reserve( x.size() );
		dy.reserve( x.size() );
		slope.reserve( x.size() );
		intercept.reserve( x.size() );
		for( int i = 0; i < x.size(); ++i ){
			if( i < x.size()-1 )
			{
				dx.push_back( x[i+1] - x[i] );
				dy.push_back( y[i+1] - y[i] );
				slope.push_back( dy[i] / dx[i] );
				intercept.push_back( y[i] - x[i] * slope[i] );
			}
			else
			{
				dx.push_back( dx[i-1] );
				dy.push_back( dy[i-1] );
				slope.push_back( slope[i-1] );
				intercept.push_back( intercept[i-1] );
			}
		}

		for ( int i = 0; i < x_new.size(); ++i ) 
		{
			int idx = findNearestNeighbourIndex( x_new[i], x );
			y_new.push_back( slope[idx] * x_new[i] + intercept[idx] );
		}
		return y_new;

	}

	inline std::vector<int> nonzero(const std::vector<double>& x, const double val)
	{
		// find index such that x[index] == val
		std::vector<int> idx;
		for (int i = 0; i < x.size(); i++) {
			if (x[i] == val)
				idx.push_back(i);
		}
		return idx;
	}

	inline double partialSum(const std::vector<double>& x, const std::vector<int>& idx, int left, int right) 
	{
		// idx[left, right)
		double sum = 0.0;
		for (int i = left; i < right; i++) {
			sum += x[idx[i]];
		}
		return sum;
	}

	inline double G3F(const std::vector<double>& x, const std::vector<double>& p) 
	{
		std::vector<double> sorted_x = x;
		std::sort(sorted_x.begin(), sorted_x.end());
		std::unordered_set<double> cache;
		std::vector<int> sigma;
		for (double val: sorted_x) {
			if (cache.count(val)) 
				continue;
			std::vector<int> idx = nonzero(x, val);
			cache.insert(val);
			for (int id: idx) {
				sigma.push_back(id);
			}
		}

		int N = p.size();
		std::vector<double> w(N, 0.0);
		for (int i = 0; i < N; i++) {
			w[i] = 1.0 / pow(2, i);
		}

		std::vector<double> w_suffix_sum(N+1, 0.0);
		for (int i = N-1; i >= 0; i--) {
			w_suffix_sum[i] = w[i] + w_suffix_sum[i+1];
		}

		std::vector<double> x_data, y_data;
		x_data.push_back(0.0);
		y_data.push_back(0.0);
		for (int i = 0; i < N; i++) {
			x_data.push_back((double)(i+1) / N);
			y_data.push_back(w_suffix_sum[N-i-1]);
		}

		std::vector<double> omega(N, 0.0), x1_list(N, 0.0), x2_list(N, 0.0);
		for (int i = 0; i < N; i++) {
			x1_list[i] = std::min(partialSum(p, sigma, i, N), 1.0);
			x2_list[i] = std::min(partialSum(p, sigma, i+1, N), 1.0);
		}
		std::vector<double> y1_list = interp1(x_data, y_data, x1_list);
		std::vector<double> y2_list = interp1(x_data, y_data, x2_list);

		for (int i = 0; i < N; i++) {
			omega[i] = y1_list[i] - y2_list[i];
		}

		double score = 0.0;
		for (int i = 0; i < N; i++) {
			score += omega[i] * sorted_x[i];
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
		double p = RandomDouble(0.0, 0.9999);
		int idx = 0;
		for (; idx < n; idx++) {
			if (pre_sum[idx] <= p && p <= pre_sum[idx+1]) break;
		}
		if (idx >= n) {
			std::cout << p << ", " << prob << std::endl;
			idx = n-1;
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
