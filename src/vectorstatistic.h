#ifndef VECTORSTATISTIC_H
#define VECTORSTATISTIC_H

#include <math.h>
#include <string>
#include <vector>
#include <assert.h>

class VECTORSTATISTIC
{
public:
	VECTORSTATISTIC();
	VECTORSTATISTIC(int dim);
	VECTORSTATISTIC(int dim, double val, int count);

	void Add(std::vector<double> val);
	void Clear();
	int GetCount() const;
	void Initialise(int dim, double val, int count);
	std::vector<double> GetTotal() const;
	std::vector<double> GetMean() const;
	std::vector<double> GetVariance() const;
	std::vector<double> GetStdDev() const;
	std::vector<double> GetStdErr() const;
	std::vector<double> GetMax() const;
	std::vector<double> GetMin() const;
	void Print(const std::string& name, std::ostream& ostr) const;

private:

	int Count;
	int Dim;
	std::vector<double> Mean;
	std::vector<double> Variance;
	std::vector<double> Min, Max;
};

inline VECTORSTATISTIC::VECTORSTATISTIC()
{
	Dim = 0;
	Clear();
}

inline VECTORSTATISTIC::VECTORSTATISTIC(int dim)
{
	Dim = dim;
	Clear();
}

inline VECTORSTATISTIC::VECTORSTATISTIC(int dim, double val, int count)
{
	Initialise(dim, val, count);
}

inline void VECTORSTATISTIC::Add(std::vector<double> val)
{
	assert(val.size() == Dim);
	std::vector<double> meanOld = Mean;
	int countOld = Count;
	++Count;
	assert(Count > 0); // overflow
	for (int i = 0; i < Dim; i++){
		Mean[i] += (val[i] - Mean[i]) / Count;
		Variance[i] = (countOld * (Variance[i] + meanOld[i] * meanOld[i]) + val[i] * val[i]) / Count - Mean[i] * Mean[i];
		if (val[i] > Max[i]){
			Max[i] = val[i];
		}
		if (val[i] < Min[i]){
			Min[i] = val[i];
		}
	}

	// Mean += (val - Mean) / Count;
	// Variance = (countOld * (Variance + meanOld * meanOld)
	// 	+ val * val) / Count - Mean * Mean;
	// if (val > Max)
	// 	Max = val;
	// if (val < Min)
	// 	Min = val;
}

inline void VECTORSTATISTIC::Clear()
{
	Count = 0;

	Mean.assign(Dim, 0.0);
	Variance.assign(Dim, 0.0);
	Min.assign(Dim, +Infinity);
	Max.assign(Dim, -Infinity);
}

inline int VECTORSTATISTIC::GetCount() const
{
	return Count;
}

inline void VECTORSTATISTIC::Initialise(int dim, double val, int count)
{	
	Dim = dim;
	Count = count;
	Mean.assign(Dim, val);
}

inline std::vector<double> VECTORSTATISTIC::GetTotal() const
{
	std::vector<double> total;
	for (int i = 0; i < Dim; i++){
		total.push_back(Mean[i] * Count);
	}
	return total;
	// return Mean * Count;
}

inline std::vector<double> VECTORSTATISTIC::GetMean() const
{
	return Mean;
}

inline std::vector<double> VECTORSTATISTIC::GetStdDev() const
{	
	std::vector<double> StdDev;
	for (int i = 0; i < Dim; i++){
		StdDev.push_back(sqrt(Variance[i]));
	}
	return StdDev;
	// return sqrt(Variance);
}

inline std::vector<double> VECTORSTATISTIC::GetStdErr() const
{
	std::vector<double> StdErr;
	for (int i = 0; i < Dim; i++){
		StdErr.push_back(sqrt(Variance[i] / Count));
	}
	return StdErr;
	// return sqrt(Variance / Count);
}

inline std::vector<double> VECTORSTATISTIC::GetMax() const
{
	return Max;
}

inline std::vector<double> VECTORSTATISTIC::GetMin() const
{
	return Min;
}

inline void VECTORSTATISTIC::Print(const std::string& name, std::ostream& ostr) const
{	
	ostr << name << ": " << std::endl;
	ostr << "Mean: ";
	for (int i = 0; i < Dim; i++){
		ostr << Mean[i] << ", ";
	}
	ostr << std::endl;

	ostr << "Min: ";
	for (int i = 0; i < Dim; i++){
		ostr << Min[i] << ", ";
	}
	ostr << std::endl;

	ostr << "Max: ";
	for (int i = 0; i < Dim; i++){
		ostr << Max[i] << ", ";
	}
	ostr << std::endl;
	// ostr << name << ": " << Mean << " [" << Min << ", " << Max << "]" << std::endl;
}

#endif // VECTORSTATISTIC_H