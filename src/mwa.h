#ifndef MULTIPRODUCT_WEB_ADVERTISE_H
#define MULTIPRODUCT_WEB_ADVERTISE_H

#include "simulator.h"
#include "utils.h"

using namespace std;

class MWA_STATE: public STATE 
{
public:
    int index;
};

class MWA: public SIMULATOR
{
public:
	// MWA(int numStates = 11, int numActions = 6, int numObs = 11, int numObjectives = 5);
	MWA(int numObjectives = 5);

	virtual std::string GetDomainName() const;
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
    int NumVisitStates, NumBuyStates, NumLeaveStates;
	// transition func
	vector<vector<vector<double>>> trans_func;
	vector<vector<double>> obs_func;
	vector<vector<vector<vector<double>>>> rew_func;
	int GetObservation(const MWA_STATE& discstate) const;
	void GenerateModel();
	void GenerateTransFunc();
	void GenerateObsFunc();
	void GenerateRewFunc();

private:

	mutable MEMORY_POOL<MWA_STATE> MemoryPool;

};

#endif // MULTIPRODUCT_WEB_ADVERTISE_H