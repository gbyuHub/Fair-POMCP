#ifndef RANDOM_DOMAIN_H
#define RANDOM_DOMAIN_H

#include "simulator.h"
#include "utils.h"

using namespace std;

class NUMBERED_STATE: public STATE 
{
public:
    int index;
};

class RANDOMENV: public SIMULATOR
{
public:
	RANDOMENV(int numStates = 2, int numActions = 2, int numObs = 2, int numObjectives = 2);

	virtual std::string GetClassName() const;
	virtual STATE* Copy(const STATE& state) const;
	virtual void Validate(const STATE& state) const;
	virtual STATE* CreateStartState() const;
	virtual void FreeState(STATE* state) const;
	virtual bool Step(STATE& state, int action,
		int& observation, std::vector<double>& reward) const;

	void GenerateLegal(const STATE& state, const HISTORY& history,
		std::vector<int>& legal, const STATUS& status) const;
	// void GeneratePreferred(const STATE& state, const HISTORY& history,
	// 	std::vector<int>& legal, const STATUS& status) const;
	virtual bool LocalMove(STATE& state, const HISTORY& history,
		int stepObservation, const STATUS& status) const;

	virtual void DisplayBeliefs(const BELIEF_STATE& beliefState,
		std::ostream& ostr) const;
	virtual void DisplayState(const STATE& state, std::ostream& ostr) const;
	virtual void DisplayObservation(const STATE& state, int observation, std::ostream& ostr) const;
	virtual void DisplayAction(int action, std::ostream& ostr) const;

protected:
	int NumStates;
	// transition func
	vector<vector<vector<double>>> trans_func;
	vector<vector<vector<double>>> obs_func;
	vector<vector<vector<double>>> rew_func;
	int GetObservation(const NUMBERED_STATE& discstate, const int action) const;
	void GenerateModel();
	void GenerateTransFunc();
	void GenerateObsFunc();
	void GenerateRewFunc();

private:

	mutable MEMORY_POOL<NUMBERED_STATE> MemoryPool;

};

#endif // RANDOM_DOMAIN_H