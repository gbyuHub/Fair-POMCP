#include "mwa.h"
#include <vector>

using namespace std;
using namespace UTILS;

MWA::MWA(int numObjectives) 
{
    // n products, n visitor states, n buy states, plus 1 leave state, 2*n+1 states in total. 
    NumObjectives = numObjectives;
    NumStates = 2 * numObjectives + 1;
    // n + 1 actions in total, action i (0 - n-1): show ad for product i, action n: show general ad
    NumActions = numObjectives + 1;
    NumObservations = NumStates;

    NumLeaveStates = 1;
    NumVisitStates = NumBuyStates = (NumStates - NumLeaveStates) / 2;

    GenerateModel();
}

void MWA::GenerateModel()
{
    GenerateTransFunc();
    GenerateObsFunc();
    GenerateRewFunc();
}

void MWA::GenerateTransFunc()
{
    trans_func = vector<vector<vector<double>>> (NumStates, vector<vector<double>>(NumActions, vector<double>(NumStates, 0.0)));

    // vistor states
    for (int s = 0; s < NumVisitStates; s++) {
        for (int a = 0; a < NumActions; a++) {
            if (a == s) {
                trans_func[s][a][s] = 0.8; // stay
                trans_func[s][a][s+NumVisitStates] = 0.05; //buy
                trans_func[s][a][NumStates-1] = 0.15; // leave
            }
            else if (a == NumActions - 1) {
                trans_func[s][a][s] = 2.0/3; // stay
                trans_func[s][a][NumStates-1] = 1.0/3; // leave
            }
            else {
                trans_func[s][a][s] = 0.5; // stay
                trans_func[s][a][NumStates-1] = 0.5; // leave
            }
        }
    }
    // buy states
    for (int s = NumVisitStates; s < NumStates - 1; s++) {
        for (int a = 0; a < NumActions; a++) {
            for (int next_s = 0; next_s < NumVisitStates; next_s++) {
                trans_func[s][a][next_s] = 1.0 / NumVisitStates;
            }
        }
    }
    // for disconnect states
    for (int a = 0; a < NumActions; a++) {
        for (int next_s = 0; next_s < NumVisitStates; next_s++) {
            trans_func[NumStates-1][a][next_s] = 1.0 / NumVisitStates;
        }
    }
}

void MWA::GenerateObsFunc()
{
    obs_func = vector<vector<double>> (NumStates, vector<double>(NumObservations, 0.0));
    for (int s = 0; s < NumVisitStates; s++) {
        double sum = 0;
        for (int obs = 0; obs < NumVisitStates; obs++) {
            obs_func[s][obs] = exp(-1 * abs(s - obs));
            sum += obs_func[s][obs];
        }
        for (int obs = 0; obs < NumVisitStates; obs++) obs_func[s][obs] /= sum;
    }
    for (int s = NumVisitStates; s < NumStates - 1; s++) obs_func[s][s] = 1.0;
    obs_func[NumStates-1][NumStates-1] = 1.0;
}

void MWA::GenerateRewFunc()
{
    // rew_func[s][a][s']: vector reward
    rew_func = vector<vector<vector<vector<double>>>> (NumStates, vector<vector<vector<double>>>(NumActions, vector<vector<double>>(NumStates, vector<double>(NumObjectives, 0.0))));
    for (int r = 0; r < NumObjectives; r++) {
        for (int s = 0; s < NumVisitStates; s++) {
            for (int a = 0; a < NumActions - 1; a++) {
                if (trans_func[s][a][r+NumVisitStates] > 0) rew_func[s][a][r+NumVisitStates][r] = 5.0;
            }
        }
    }
}

std::string MWA::GetDomainName() const
{
    return "MWA";
}

STATE* MWA::Copy(const STATE& state) const
{
	const MWA_STATE& discstate = safe_cast<const MWA_STATE&>(state);
	MWA_STATE* newstate = MemoryPool.Allocate();
	*newstate = discstate;
	return newstate;
}

void MWA::Validate(const STATE& state) const
{
    const MWA_STATE& discstate = safe_cast<const MWA_STATE&>(state);
    assert(discstate.index >= 0 && discstate.index < NumStates);
}

STATE* MWA::CreateStartState() const
{
    int state_idx = Random(NumStates);
    // discrete state
    MWA_STATE* discstate = MemoryPool.Allocate();
    discstate->index = state_idx;
    return discstate;
}

void MWA::FreeState(STATE* state) const
{
    MWA_STATE* discstate = safe_cast<MWA_STATE*>(state);
    MemoryPool.Free(discstate);
}

bool MWA::Step(STATE& state, int action, 
        int& observation, std::vector<double>& reward) const
{
    MWA_STATE& discstate = safe_cast<MWA_STATE&>(state);
    int state_idx = discstate.index;
    int next_s_idx = sample(trans_func[state_idx][action]);
    reward = rew_func[state_idx][action][next_s_idx];
    discstate.index = next_s_idx;
    observation = GetObservation(discstate);
    return false;
}

void MWA::GenerateLegal(const STATE& state, const HISTORY& history,
    std::vector<int>& legal, const STATUS& status) const
{
    const MWA_STATE& mwastate = safe_cast<const MWA_STATE&>(state);
    int state_idx = mwastate.index;

    for (int action = 0; action < NumActions; action++) {
        legal.push_back(action);
    }
    return;
}

bool MWA::LocalMove(STATE& state, const HISTORY& history,
    int stepObservation, const STATUS& status) const
{
    MWA_STATE& discstate = safe_cast<MWA_STATE&>(state);
    int idx = Random(NumStates);
    discstate.index = idx;
    int realObs = history.Back().Observation;
    int action = history.Back().Action;
    int newObs = GetObservation(discstate);
    if (newObs != realObs) {
        return false;
    }
    return true;
}

int MWA::GetObservation(const MWA_STATE& discstate) const 
{
    int state_idx = discstate.index;
    int obs = sample(obs_func[state_idx]);
    return obs;
}

void MWA::DisplayBeliefs(const BELIEF_STATE& beliefState,
    std::ostream& ostr) const
{

}

void MWA::DisplayState(const STATE& state, std::ostream& ostr) const
{

}

void MWA::DisplayObservation(const STATE& state, int observation, std::ostream& ostr) const
{

}

void MWA::DisplayAction(int action, std::ostream& ostr) const
{

}