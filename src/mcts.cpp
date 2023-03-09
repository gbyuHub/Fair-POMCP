#include "mcts.h"
#include "testsimulator.h"
#include <math.h>

#include <algorithm>

using namespace std;
using namespace UTILS;

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

//-----------------------------------------------------------------------------

MCTS::PARAMS::PARAMS()
	: Verbose(0),
	MaxDepth(100),
	NumSimulations(1000),
	NumStartStates(1000),
	UseTransforms(true),
	NumTransforms(0),
	MaxAttempts(0),
	ExpandCount(1),
	EnsembleSize(4),
    BanditArmCapacity(10),
    BanditConvergenceEpsilon(0.01),
	ExplorationConstant(1),
	UseRave(false),
	RaveDiscount(1.0),
	RaveConstant(0.01),
	DisableTree(false),
	Strategy("GGF"),
	ConsiderPast(true)
{
}

MCTS::MCTS(const SIMULATOR& simulator, const PARAMS& params)
	: Simulator(simulator),
	Params(params),
	TreeDepth(0)
{
	VNODE::NumChildren = Simulator.GetNumActions();
	QNODE::NumChildren = Simulator.GetNumObservations();

	Root = ExpandNode(Simulator.CreateStartState());

	for (int i = 0; i < Params.NumStartStates; i++)
		Root->Beliefs().AddSample(Simulator.CreateStartState());
}

MCTS::~MCTS()
{
	VNODE::Free(Root, Simulator);
	VNODE::FreeAll();
}

bool MCTS::Update(int action, int observation, vector<double>& reward)
{
	History.Add(action, observation);
	BELIEF_STATE beliefs;

	// Find matching vnode from the rest of the tree
	QNODE& qnode = Root->Child(action);
	VNODE* vnode = qnode.Child(observation);
	if (vnode)
	{
		if (Params.Verbose >= 1)
			cout << "Matched " << vnode->Beliefs().GetNumSamples() << " states" << endl;
		beliefs.Copy(vnode->Beliefs(), Simulator);
	}
	else
	{
		if (Params.Verbose >= 1)
			cout << "No matching node found" << endl;
	}

	// Generate transformed states to avoid particle deprivation
	if (Params.UseTransforms)
		AddTransforms(Root, beliefs);

	// If we still have no particles, fail
	if (beliefs.Empty() && (!vnode || vnode->Beliefs().Empty()))
		return false;

	if (Params.Verbose >= 2)
		Simulator.DisplayBeliefs(beliefs, cout);

	// Find a state to initialise prior (only requires fully observed state)
	const STATE* state = 0;
	if (vnode && !vnode->Beliefs().Empty())
		state = vnode->Beliefs().GetSample(0);
	else
		state = beliefs.GetSample(0);

	// Delete old tree and create new root
	VNODE::Free(Root, Simulator);
	VNODE* newRoot = ExpandNode(state);
	newRoot->Beliefs() = beliefs;
	Root = newRoot;
	return true;
}

int MCTS::SelectAction(const std::vector<double>& cumulative_past_rew)
{
	if (Params.DisableTree)
		RolloutSearch();
	else
		UCTSearch(cumulative_past_rew);
	return GreedyUCB(Root, false);
}

void MCTS::RolloutSearch()
{
	std::vector<double> totals(Simulator.GetNumActions(), 0.0);
	int historyDepth = History.Size();
	std::vector<int> legal;
	assert(BeliefState().GetNumSamples() > 0);
	Simulator.GenerateLegal(*BeliefState().GetSample(0), GetHistory(), legal, GetStatus());
	random_shuffle(legal.begin(), legal.end());
	for (int i = 0; i < Params.NumSimulations; i++)
	{
		int action = legal[i % legal.size()];
		STATE* state = Root->Beliefs().CreateSample(Simulator);
		Simulator.Validate(*state);

		int observation;
		std::vector<double> immediateReward(2, 0.0), delayedReward(2, 0.0), totalReward(2, 0.0);
		bool terminal = Simulator.Step(*state, action, observation, immediateReward);

		VNODE*& vnode = Root->Child(action).Child(observation);
		if (!vnode && !terminal)
		{
			vnode = ExpandNode(state);
			AddSample(vnode, *state);
		}
		History.Add(action, observation);

		delayedReward = Rollout(*state);

		// totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
		for (int i = 0; i < 2; i++){
			totalReward[i] = immediateReward[i] + Simulator.GetDiscount() * delayedReward[i];
		}
		Root->Child(action).Value.Add(totalReward);

		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}
}

void MCTS::UCTSearch(std::vector<double> cumulative_past_rew)
{
	ClearStatistics();
	int historyDepth = History.Size();

	for (int n = 0; n < Params.NumSimulations; n++)
	{
		STATE* state = Root->Beliefs().CreateSample(Simulator);
		Simulator.Validate(*state);
		Status.Phase = SIMULATOR::STATUS::TREE;
		if (Params.Verbose >= 1)
		{
			cout << "Starting simulation " << n << endl;
			Simulator.DisplayState(*state, cout);
		}

		TreeDepth = 0;
		PeakTreeDepth = 0;
		std::vector<double> totalReward = SimulateV(*state, Root, cumulative_past_rew, false);
		StatTotalReward.Add(totalReward);	
		StatTreeDepth.Add(PeakTreeDepth);
		// cout << "Total reward = " << "[" << totalReward[0] << ", " <<totalReward[1] << "]" << endl;

		if (Params.Verbose >= 2)
			cout << "Total reward = " << "[" << totalReward[0] << ", " <<totalReward[1] << "]" << endl;
		if (Params.Verbose >= 3)
			DisplayValue(4, cout);

		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}

	DisplayStatistics(cout);
}

std::vector<double> MCTS::SimulateV(STATE& state, VNODE* vnode, std::vector<double> cumulative_past_rew, bool stop_search)
{
	PeakTreeDepth = TreeDepth;
	if (TreeDepth >= Params.MaxDepth) // search horizon reached
	{
		return {0.0, 0.0};
	}
	if (TreeDepth == 1)
		AddSample(vnode, state);
	if (stop_search)
		return {0.0, 0.0};

	int action = GreedyUCB(vnode, true);
	if (Params.Verbose >= 1) cout << "[TREE SEARCH]: select action " << action << endl;

	QNODE& qnode = vnode->Child(action);
	// double totalReward = SimulateQ(state, qnode, action);
	vector<double> totalReward = SimulateQ(state, qnode, action, cumulative_past_rew);
	vnode->Value.Add(totalReward);
	// AddRave(vnode, totalReward);
	return totalReward;
}

std::vector<double> MCTS::SimulateQ(STATE& state, QNODE& qnode, int action, std::vector<double> cumulative_past_rew)
{
	int observation;
	// double immediateReward, delayedReward = 0;
	vector<double> immediateReward(2, 0.0), delayedReward(2, 0.0);
	vector<double> cumulative_past_rew_old = cumulative_past_rew;

	if (Simulator.HasAlpha())
		Simulator.UpdateAlpha(qnode, state);
	bool terminal = Simulator.Step(state, action, observation, immediateReward);
	bool stop_search = (accumulate(immediateReward.begin(), immediateReward.end(), 0.0) > 0);
	if (Params.Verbose >= 1) {
		if (stop_search)
			cout << "[TREE] Sample a rock with reward " << immediateReward << endl;
	}
	for (int i = 0; i < cumulative_past_rew.size(); i++) {
		cumulative_past_rew[i] += Simulator.GetDiscount() * immediateReward[i];
	}
	assert(observation >= 0 && observation < Simulator.GetNumObservations());
	History.Add(action, observation);

	if (Params.Verbose >= 3)
	{
		Simulator.DisplayAction(action, cout);
		Simulator.DisplayObservation(state, observation, cout);
		Simulator.DisplayVectorReward(immediateReward, cout);
		Simulator.DisplayState(state, cout);
	}

	VNODE*& vnode = qnode.Child(observation);
	if (!vnode && !terminal && qnode.Value.GetCount() >= Params.ExpandCount)
	{
		vnode = ExpandNode(&state);
	}

	if (!terminal)
	{
		TreeDepth++;
		if (vnode) {
			delayedReward = SimulateV(state, vnode, cumulative_past_rew, stop_search);
		}
		else {
			delayedReward = Rollout(state);
		}
		TreeDepth--;
	}

	// double totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
	std::vector<double> totalReward(2, 0.0);
	for (int i = 0; i < 2; i++){
		totalReward[i] = immediateReward[i] + Simulator.GetDiscount() * delayedReward[i];
	}
	// qnode.Value.AddCumulatedReward(cumulatedReward);
	if (Params.ConsiderPast) {
		for (int i = 0; i < 2; i++) {
			totalReward[i] = cumulative_past_rew_old[i] + Simulator.GetDiscount() * totalReward[i];
		}
	}
	qnode.Value.Add(totalReward);
	return totalReward;
}

/*
void MCTS::AddRave(VNODE* vnode, double totalReward)
{
	double totalDiscount = 1.0;
	for (int t = TreeDepth; t < History.Size(); ++t)
	{
		QNODE& qnode = vnode->Child(History[t].Action);
		qnode.AMAF.Add(totalReward, totalDiscount);
		totalDiscount *= Params.RaveDiscount;
	}
}
*/

VNODE* MCTS::ExpandNode(const STATE* state)
{
	VNODE* vnode = VNODE::Create();
	vnode->Value.Set(0, 0);
	Simulator.Prior(state, History, vnode, Status);

	if (Params.Verbose >= 2)
	{
		cout << "Expanding node: ";
		History.Display(cout);
		cout << endl;
	}

	return vnode;
}

void MCTS::AddSample(VNODE* node, const STATE& state)
{
	STATE* sample = Simulator.Copy(state);
	node->Beliefs().AddSample(sample);
	if (Params.Verbose >= 2)
	{
		cout << "Adding sample:" << endl;
		Simulator.DisplayState(*sample, cout);
	}
}

int MCTS::GreedyUCB(VNODE* vnode, bool ucb) const
{
	static vector<int> besta;
	besta.clear();
	double bestq = -Infinity;
	int N = vnode->Value.GetCount();
	double logN = log(N + 1);
	bool hasalpha = Simulator.HasAlpha();
	if (Params.Verbose >= 1) cout << (ucb ? "----- PLANING -----" : "----- EXECUTION -----") << endl;

	for (int action = 0; action < Simulator.GetNumActions(); action++)
	{
		std::vector<double> q;
		double a;
		double alphaq;
		int n, alphan;

		QNODE& qnode = vnode->Child(action);
		q = qnode.Value.GetValue();
		n = qnode.Value.GetCount();
        if (Params.Verbose >= 1) {
            cout << "Action = " << action << ", q = " << q << ", n = " << n << ", UCB term = " << FastUCB(N, n, logN) << endl;
        }

		if (Params.Strategy == "GGF") {
			a = GGF(q); // GGF criteria
            // cout << "GGF score = " << a << endl;
		}
		else {
			a = WS(q); // weighted sum
            // cout << "WS score = " << a << endl;
		}
        // cout << "exploration term: " << FastUCB(N, n, logN) << endl;

		if (ucb) a += FastUCB(N, n, logN);
		// cout << "a = " << a << endl;

		if (a >= bestq)
		{
			if (a > bestq)
				besta.clear();
			bestq = a;
			besta.push_back(action);
		}
	}
	assert(!besta.empty());
	return besta[Random(besta.size())];
}

std::vector<double> MCTS::Rollout(STATE& state)
{
	Status.Phase = SIMULATOR::STATUS::ROLLOUT;
	if (Params.Verbose >= 3)
		cout << "Starting rollout" << endl;

	// double totalReward = 0.0;
	std::vector<double> totalReward(2, 0.0);
	double discount = 1.0;
	bool terminal = false;
	int numSteps;
	for (numSteps = 0; numSteps + TreeDepth < Params.MaxDepth && !terminal; ++numSteps)
	{
		int observation;
		std::vector<double> reward(2, 0.0);

		int action = Simulator.SelectRandom(state, History, Status);
		// cout << "[ROLLOUT]: select action " << action << endl;
		terminal = Simulator.Step(state, action, observation, reward);
		bool stop_search = (accumulate(reward.begin(), reward.end(), 0.0) > 0);
		if (Params.Verbose >= 1) {
			if (stop_search)
				cout << "[ROLLOUT] Sample a rock with reward " << reward << endl;
		}
		History.Add(action, observation);

		if (Params.Verbose >= 4)
		{
			Simulator.DisplayAction(action, cout);
			Simulator.DisplayObservation(state, observation, cout);
			// Simulator.DisplayReward(reward, cout);
			Simulator.DisplayVectorReward(reward, cout);
			Simulator.DisplayState(state, cout);
		}

		// totalReward += reward * discount;
		for (int i = 0; i < 2; i++){
			totalReward[i] += reward[i] * discount;
		}
		if (stop_search) 
			break;
		discount *= Simulator.GetDiscount();
	}
	StatRolloutDepth.Add(numSteps);
	if (Params.Verbose >= 3)
		cout << "Ending rollout after " << numSteps
		<< " steps, with total reward " << "[" << totalReward[0] << ", " << totalReward[1] << "]" << endl;
	return totalReward;
}

void MCTS::AddTransforms(VNODE* root, BELIEF_STATE& beliefs)
{
	int attempts = 0, added = 0;

	// Local transformations of state that are consistent with history
	while (added < Params.NumTransforms && attempts < Params.MaxAttempts)
	{
		STATE* transform = CreateTransform();
		if (transform)
		{
			beliefs.AddSample(transform);
			added++;
		}
		attempts++;
	}

	if (Params.Verbose >= 1)
	{
		cout << "Created " << added << " local transformations out of "
			<< attempts << " attempts" << endl;
	}
}

STATE* MCTS::CreateTransform() const
{
	int stepObs;
	std::vector<double> stepReward;

	STATE* state = Root->Beliefs().CreateSample(Simulator);
	Simulator.Step(*state, History.Back().Action, stepObs, stepReward);
	if (Simulator.LocalMove(*state, History, stepObs, Status))
		return state;
	Simulator.FreeState(state);
	return 0;
}

double MCTS::UCB[UCB_N][UCB_n];
bool MCTS::InitialisedFastUCB = true;

void MCTS::InitFastUCB(double exploration)
{
	cout << "Initialising fast UCB table... ";
	for (int N = 0; N < UCB_N; ++N)
		for (int n = 0; n < UCB_n; ++n)
			if (n == 0)
				UCB[N][n] = Infinity;
			else
				UCB[N][n] = exploration * sqrt(log(N + 1) / n);
	cout << "done" << endl;
	InitialisedFastUCB = true;
}

inline double MCTS::FastUCB(int N, int n, double logN) const
{
	if (InitialisedFastUCB && N < UCB_N && n < UCB_n)
		return UCB[N][n];

	if (n == 0)
		return Infinity;
	else
		return Params.ExplorationConstant * sqrt(logN / n);
}

void MCTS::ClearStatistics()
{
	StatTreeDepth.Clear();
	StatRolloutDepth.Clear();
	StatTotalReward.Clear();
}

void MCTS::DisplayStatistics(ostream& ostr) const
{
	if (Params.Verbose >= 1)
	{
		StatTreeDepth.Print("Tree depth", ostr);
		StatRolloutDepth.Print("Rollout depth", ostr);
		StatTotalReward.Print("Total reward", ostr);
	}

	if (Params.Verbose >= 2)
	{
		ostr << "Policy after " << Params.NumSimulations << " simulations" << endl;
		DisplayPolicy(6, ostr);
		ostr << "Values after " << Params.NumSimulations << " simulations" << endl;
		DisplayValue(6, ostr);
	}
}

void MCTS::DisplayValue(int depth, ostream& ostr) const
{
	HISTORY history;
	ostr << "MCTS Values:" << endl;
	Root->DisplayValue(history, depth, ostr);
}

void MCTS::DisplayPolicy(int depth, ostream& ostr) const
{
	HISTORY history;
	ostr << "MCTS Policy:" << endl;
	Root->DisplayPolicy(history, depth, ostr);
}

//-----------------------------------------------------------------------------
/*
void MCTS::UnitTest()
{
	UnitTestGreedy();
	UnitTestUCB();
	UnitTestRollout();
	for (int depth = 1; depth <= 3; ++depth)
		UnitTestSearch(depth);
}

void MCTS::UnitTestGreedy()
{
	TEST_SIMULATOR testSimulator(5, 5, 0);
	PARAMS params;
	MCTS mcts(testSimulator, params);
	int numAct = testSimulator.GetNumActions();
	int numObs = testSimulator.GetNumObservations();

	VNODE* vnode = mcts.ExpandNode(testSimulator.CreateStartState());
	vnode->Value.Set(1, 0);
	vnode->Child(0).Value.Set(0, 1);
	for (int action = 1; action < numAct; action++)
		vnode->Child(action).Value.Set(0, 0);
	assert(mcts.GreedyUCB(vnode, false) == 0);
}

void MCTS::UnitTestUCB()
{
	TEST_SIMULATOR testSimulator(5, 5, 0);
	PARAMS params;
	MCTS mcts(testSimulator, params);
	int numAct = testSimulator.GetNumActions();
	int numObs = testSimulator.GetNumObservations();

	// With equal value, action with lowest count is selected
	VNODE* vnode1 = mcts.ExpandNode(testSimulator.CreateStartState());
	vnode1->Value.Set(1, 0);
	for (int action = 0; action < numAct; action++)
		if (action == 3)
			vnode1->Child(action).Value.Set(99, 0);
		else
			vnode1->Child(action).Value.Set(100 + action, 0);
	assert(mcts.GreedyUCB(vnode1, true) == 3);

	// With high counts, action with highest value is selected
	VNODE* vnode2 = mcts.ExpandNode(testSimulator.CreateStartState());
	vnode2->Value.Set(1, 0);
	for (int action = 0; action < numAct; action++)
		if (action == 3)
			vnode2->Child(action).Value.Set(99 + numObs, 1);
		else
			vnode2->Child(action).Value.Set(100 + numAct - action, 0);
	assert(mcts.GreedyUCB(vnode2, true) == 3);

	// Action with low value and low count beats actions with high counts
	VNODE* vnode3 = mcts.ExpandNode(testSimulator.CreateStartState());
	vnode3->Value.Set(1, 0);
	for (int action = 0; action < numAct; action++)
		if (action == 3)
			vnode3->Child(action).Value.Set(1, 1);
		else
			vnode3->Child(action).Value.Set(100 + action, 1);
	assert(mcts.GreedyUCB(vnode3, true) == 3);

	// Actions with zero count is always selected
	VNODE* vnode4 = mcts.ExpandNode(testSimulator.CreateStartState());
	vnode4->Value.Set(1, 0);
	for (int action = 0; action < numAct; action++)
		if (action == 3)
			vnode4->Child(action).Value.Set(0, 0);
		else
			vnode4->Child(action).Value.Set(1, 1);
	assert(mcts.GreedyUCB(vnode4, true) == 3);
}

void MCTS::UnitTestRollout()
{
	TEST_SIMULATOR testSimulator(2, 2, 0);
	PARAMS params;
	params.NumSimulations = 1000;
	params.MaxDepth = 10;
	MCTS mcts(testSimulator, params);
	double totalReward = 0;
	for (int n = 0; n < mcts.Params.NumSimulations; ++n)
	{
		STATE* state = testSimulator.CreateStartState();
		mcts.TreeDepth = 0;
		totalReward += mcts.Rollout(*state);
	}
	double rootValue = totalReward / mcts.Params.NumSimulations;
	double meanValue = testSimulator.MeanValue();
	assert(fabs(meanValue - rootValue) < 0.1);
}

void MCTS::UnitTestSearch(int depth)
{
	TEST_SIMULATOR testSimulator(3, 2, depth);
	PARAMS params;
	params.MaxDepth = depth + 1;
	params.NumSimulations = pow(10, depth + 1);
	MCTS mcts(testSimulator, params);
	mcts.UCTSearch();
	double rootValue = mcts.Root->Value.GetValue();
	double optimalValue = testSimulator.OptimalValue();
	assert(fabs(optimalValue - rootValue) < 0.1);
}
*/
//-----------------------------------------------------------------------------
