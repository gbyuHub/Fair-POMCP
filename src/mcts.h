#ifndef MCTS_H
#define MCTS_H

#include "simulator.h"
#include "node.h"
#include "statistic.h"
#include "vectorstatistic.h"
#include <numeric>

class MCTS
{
public:

	struct PARAMS
	{
		PARAMS();

		int Verbose;
		int MaxDepth;
		int NumSimulations;
		int NumStartStates;
		bool UseTransforms;
		int NumTransforms;
		int MaxAttempts;
		int ExpandCount;
		int EnsembleSize;
        int BanditArmCapacity;
        double BanditConvergenceEpsilon;
		int BanditBetaPrior;
		double ExplorationConstant;
		bool UseRave;
		double RaveDiscount;
		double RaveConstant;
		bool DisableTree;
		std::string Strategy;
		std::vector<double> ImportanceWeight;
		bool ConsiderPast; // consider past cumulated reward or not
		int NumObjectives;
	};

	MCTS(const SIMULATOR& simulator, const PARAMS& params);
	virtual ~MCTS();

	virtual int SelectAction(const std::vector<double>& cumulative_past_rew);
	bool Update(int action, int observation, std::vector<double>& reward);

	void UCTSearch(std::vector<double> cumulative_past_rew);
	void RolloutSearch();

	std::vector<double> Rollout(STATE& state);

	const BELIEF_STATE& BeliefState() const { return Root->Beliefs(); }
	const HISTORY& GetHistory() const { return History; }
	const SIMULATOR::STATUS& GetStatus() const { return Status; }
	void ClearStatistics();
	void DisplayStatistics(std::ostream& ostr) const;
	void DisplayValue(int depth, std::ostream& ostr) const;
	void DisplayPolicy(int depth, std::ostream& ostr) const;

	// static void UnitTest();
	static void InitFastUCB(double exploration);

	int GreedyUCB(VNODE* vnode, bool ucb) const;
	int SelectRandom() const;
	std::vector<double> SimulateV(STATE& state, VNODE* vnode, std::vector<double> cumulative_past_rew, bool stop_search);
	std::vector<double> SimulateQ(STATE& state, QNODE& qnode, int action, std::vector<double> cumulative_past_rew);
	void AddRave(VNODE* vnode, double totalReward);
	VNODE* ExpandNode(const STATE* state);
	void AddSample(VNODE* node, const STATE& state);
	void AddTransforms(VNODE* root, BELIEF_STATE& beliefs);
	STATE* CreateTransform() const;
	void Resample(BELIEF_STATE& beliefs);

	// Fast lookup table for UCB
	static const int UCB_N = 10000, UCB_n = 100;
	static double UCB[UCB_N][UCB_n];
	static bool InitialisedFastUCB;

	double FastUCB(int N, int n, double logN) const;
	const SIMULATOR& Simulator;
	int TreeDepth, PeakTreeDepth;
	PARAMS Params;
	VNODE* Root;
	HISTORY History;
	SIMULATOR::STATUS Status;
	STATISTIC StatTreeDepth;
	STATISTIC StatRolloutDepth;
	VECTORSTATISTIC StatTotalReward;
private:
	static void UnitTestGreedy();
	static void UnitTestUCB();
	static void UnitTestRollout();
	static void UnitTestSearch(int depth);
};

#endif // MCTS_H
