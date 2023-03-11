#include "randomdomain.h"
#include <vector>

using namespace std;
using namespace UTILS;

RANDOMENV::RANDOMENV(int numStates, int numActions, int numObs, int numObjectives) 
    : NumStates(numStates)
{
    NumActions = numActions;
    NumObservations = numObs;
    NumObjectives = numObjectives;

    GenerateModel();
}

void RANDOMENV::GenerateModel()
{
    GenerateTransFunc();
    GenerateObsFunc();
    GenerateRewFunc();
}

void RANDOMENV::GenerateTransFunc()
{
    trans_func = vector<vector<vector<double>>> (NumStates, vector<vector<double>>(NumActions, vector<double>(NumStates, 0.0)));
    trans_func[0][0][0] = 0.9;
    trans_func[0][0][1] = 0.1;
    trans_func[0][1][0] = 0.4;
    trans_func[0][1][1] = 0.6;
    trans_func[1][0][0] = 0.35;
    trans_func[1][0][1] = 0.65;
    trans_func[1][1][0] = 0.8;
    trans_func[1][1][1] = 0.2;
}

void RANDOMENV::GenerateObsFunc()
{
    obs_func = vector<vector<vector<double>>> (NumStates, vector<vector<double>>(NumActions, vector<double>(NumObservations, 0.0)));
    obs_func[0][0][0] = 0.8;
    obs_func[0][0][1] = 0.2;
    obs_func[1][0][0] = 0.3;
    obs_func[1][0][1] = 0.7;
    obs_func[0][1][0] = 0.4;
    obs_func[0][1][1] = 0.6;
    obs_func[1][1][0] = 0.5;
    obs_func[1][1][1] = 0.5;
}

void RANDOMENV::GenerateRewFunc()
{
    rew_func = vector<vector<vector<double>>> (NumStates, vector<vector<double>>(NumActions, vector<double>(NumObjectives, 0.0)));
    rew_func[0][0] = {3, 7};
    rew_func[0][1] = {4, 4};
    rew_func[1][0] = {5, 5};
    rew_func[1][1] = {8, 2};
}

std::string RANDOMENV::GetClassName() const
{
    return "RANDOM_DOMAIN";
}

STATE* RANDOMENV::Copy(const STATE& state) const
{
	const NUMBERED_STATE& discstate = safe_cast<const NUMBERED_STATE&>(state);
	NUMBERED_STATE* newstate = MemoryPool.Allocate();
	*newstate = discstate;
	return newstate;
}

void RANDOMENV::Validate(const STATE& state) const
{
    const NUMBERED_STATE& discstate = safe_cast<const NUMBERED_STATE&>(state);
    assert(discstate.index >= 0 && discstate.index < 2);
}

STATE* RANDOMENV::CreateStartState() const
{
    int state_idx = Random(NumStates);
    // discrete state
    NUMBERED_STATE* discstate = MemoryPool.Allocate();
    discstate->index = state_idx;
    return discstate;
}

void RANDOMENV::FreeState(STATE* state) const
{
    NUMBERED_STATE* discstate = safe_cast<NUMBERED_STATE*>(state);
    MemoryPool.Free(discstate);
}

bool RANDOMENV::Step(STATE& state, int action, 
        int& observation, std::vector<double>& reward) const
{
    NUMBERED_STATE& discstate = safe_cast<NUMBERED_STATE&>(state);
    int state_idx = discstate.index;
    reward = rew_func[state_idx][action];
    vector<double> next_s_p = trans_func[state_idx][action];
    if (Bernoulli(next_s_p[0])) {
        discstate.index = 0;
    }
    else {
        discstate.index = 1;
    }
    observation = GetObservation(discstate, action);
    return false;
}

void RANDOMENV::GenerateLegal(const STATE& state, const HISTORY& history,
    std::vector<int>& legal, const STATUS& status) const
{
    for (int action = 0; action < NumActions; action++) {
        legal.push_back(action);
    }
    return;
}

bool RANDOMENV::LocalMove(STATE& state, const HISTORY& history,
    int stepObservation, const STATUS& status) const
{
    NUMBERED_STATE& discstate = safe_cast<NUMBERED_STATE&>(state);
    discstate.index = 1 - discstate.index;
    int realObs = history.Back().Observation;
    int action = history.Back().Action;
    int newObs = GetObservation(discstate, action);
    if (newObs != realObs) {
        return false;
    }
    return true;
}

int RANDOMENV::GetObservation(const NUMBERED_STATE& discstate, const int action) const 
{
    int state_idx = discstate.index;
    vector<double> obs_p = obs_func[state_idx][action];
    if (Bernoulli(obs_p[0])) {
        return 0;
    }
    return 1;
}

void RANDOMENV::DisplayBeliefs(const BELIEF_STATE& beliefState,
    std::ostream& ostr) const
{

}

void RANDOMENV::DisplayState(const STATE& state, std::ostream& ostr) const
{

}

void RANDOMENV::DisplayObservation(const STATE& state, int observation, std::ostream& ostr) const
{

}

void RANDOMENV::DisplayAction(int action, std::ostream& ostr) const
{

}