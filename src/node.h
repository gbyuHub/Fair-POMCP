#ifndef NODE_H
#define NODE_H

#include "beliefstate.h"
#include "utils.h"
#include <iostream>

class HISTORY;
class SIMULATOR;
class QNODE;
class VNODE;

//-----------------------------------------------------------------------------
// Efficient computation of value from alpha vectors
// Only used for explicit POMDPs
struct ALPHA
{
	std::vector<double> AlphaSum;
	double MaxValue;
};

//-----------------------------------------------------------------------------

template<class COUNT>
class VALUE
{
public:

	void Set(double count, double value)
	{
		Count = count;
		// Total = value * count;
		// SquaredTotal = value*value*count;
		Total.resize(2);
		realCumulatedReward.resize(2);
		simCumulatedReward.resize(2);
		for (int i = 0; i < 2; i++){
			realCumulatedReward[i] = 0.0;
			simCumulatedReward[i] = 0.0;
			Total[i] = value * count;
		}
	}

	void Add(const std::vector<double>& totalReward)
	{
		Count += 1.0;
		assert(totalReward.size() == 2);
		for (int i = 0; i < 2; i++){
			Total[i] += totalReward[i];
		}
		// Total += totalReward;
		// SquaredTotal += totalReward*totalReward;
	}

	void Add(const std::vector<double>& totalReward, COUNT weight)
	{
		Count += weight;
		assert(totalReward.size() == 2);
		for (int i = 0; i < 2; i++){
			Total[i] += totalReward[i] * weight;
		}
		// Total += totalReward * weight;
	}

	std::vector<double> GetValue() const
	{
		// return Count == 0 ? Total : Total / Count;
		if (Count == 0){
			return Total;
		}
		else {
			std::vector<double> ret(2, 0.0);
			for (int i = 0; i < 2; i++){
				ret[i] = Total[i] / Count;
			}
			return ret;
		}
	}

	COUNT GetCount() const
	{
		return Count;
	}

	double GetSquaredValue() const
	{
		return SquaredTotal;
	}

	void AddRealCumulatedReward(const std::vector<double>& reward)
	{
		for (int i = 0; i < 2; i++){
			realCumulatedReward[i] += reward[i];
		}
	}

	void AddSimCumulatedReward(const std::vector<double>& reward)
	{
		for (int i = 0; i < 2; i++){
			simCumulatedReward[i] += reward[i];
		}
	}

	std::vector<double> GetRealCumulatedReward() const
	{
		return realCumulatedReward;
	}

	std::vector<double> GetSimCumulatedReward() const
	{
		return simCumulatedReward;
	}

	void clearSimCumulatedReward() 
	{
		simCumulatedReward = {0.0, 0.0};
	} 

	void clearRealCumulatedReward() 
	{
		realCumulatedReward = {0.0, 0.0};
	} 

private:

	COUNT Count;
	std::vector<double> Total;
	double SquaredTotal;
	std::vector<double> realCumulatedReward; // cumulated reward during simulation in the simulator
	std::vector<double> simCumulatedReward; // cumulated reward obtained in the real environment
};

//-----------------------------------------------------------------------------

class QNODE
{
public:

	VALUE<int> Value;
	VALUE<double> AMAF;

	void Initialise();

	VNODE*& Child(int c) { return Children[c]; }
	VNODE* Child(int c) const { return Children[c]; }
	ALPHA& Alpha() { return AlphaData; }
	const ALPHA& Alpha() const { return AlphaData; }

	void DisplayValue(HISTORY& history, int maxDepth, std::ostream& ostr) const;
	void DisplayPolicy(HISTORY& history, int maxDepth, std::ostream& ostr) const;

	static int NumChildren;
private:

	std::vector<VNODE*> Children;
	ALPHA AlphaData;
	friend class VNODE;
};

//-----------------------------------------------------------------------------

class VNODE : public MEMORY_OBJECT
{
public:
	VALUE<int> Value;
	void Initialise();
	static VNODE* Create();
	static void Free(VNODE* vnode, const SIMULATOR& simulator);
	static void FreeAll();

	QNODE& Child(int c) { return Children[c]; }
	const QNODE& Child(int c) const { return Children[c]; }
	BELIEF_STATE& Beliefs() { return BeliefState; }
	const BELIEF_STATE& Beliefs() const { return BeliefState; }
	void setBeliefs(BELIEF_STATE& newBelief)
	{
		BeliefState = newBelief;
	}

	void SetChildren(int count, double value);

	void DisplayValue(HISTORY& history, int maxDepth, std::ostream& ostr) const;
	void DisplayPolicy(HISTORY& history, int maxDepth, std::ostream& ostr) const;

	static int NumChildren;
private:
	std::vector<QNODE> Children;
	BELIEF_STATE BeliefState;
	static MEMORY_POOL<VNODE> VNodePool;
};

#endif // NODE_H
