#include "experiment.h"
#include "boost/timer.hpp"

using namespace std;
using namespace UTILS;

template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) 
{
    os << "[";
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    {
        os << *ii << (ii != v.end() - 1 ? " " : "");
    }
    os << "]";
    return os;
}

EXPERIMENT::PARAMS::PARAMS()
	: NumRuns(1),
	NumSteps(100000),
	SimSteps(1000),
	TimeOut(12*3600),
	MinDoubles(1),
	MaxDoubles(16),
	TransformDoubles(-4),
	TransformAttempts(1000),
	Accuracy(0.01),
	UndiscountedHorizon(1000),
	AutoExploration(true),
	usePOSTS(false)
{
}

EXPERIMENT::EXPERIMENT(const SIMULATOR& real,
	const SIMULATOR& simulator, const string& outputFile,
	EXPERIMENT::PARAMS& expParams, MCTS::PARAMS& searchParams)
	: Real(real),
	Simulator(simulator),
	OutputFile(outputFile.c_str()),
	ExpParams(expParams),
	SearchParams(searchParams)
{
	if (ExpParams.AutoExploration)
	{
		if (SearchParams.UseRave)
			SearchParams.ExplorationConstant = 0;
		else
			SearchParams.ExplorationConstant = simulator.GetRewardRange();
	}
	MCTS::InitFastUCB(SearchParams.ExplorationConstant);
}

void EXPERIMENT::Run()
{
	boost::timer timer;

	MCTS* mcts = NULL;
	// use MCTS anyway
	mcts = new MCTS(Simulator, SearchParams);

	// double undiscountedReturn = 0.0;
	// double discountedReturn = 0.0;
	std::vector<double> undiscountedReturn(2, 0.0);
	std::vector<double> discountedReturn(2, 0.0);
	std::vector<double> cumulativeReward(2, 0.0);
	double discount = 1.0;
	bool terminal = false;
	bool outOfParticles = false;
	int t = 0;
    // variables only for rocksample problem
    int collected_type1_rocks_num = 0;
    int collected_type2_rocks_num = 0;
    int num_check_action = 0;
	int collected_rock_num = 0;

	STATE* state = Real.CreateStartState();
	if (SearchParams.Verbose >= 1)
		Real.DisplayState(*state, cout);

	// for (t = 0; t < ExpParams.NumSteps; t++)
	for (collected_rock_num = 0; collected_rock_num < 2; )
	{
		int observation;
		vector<double> reward;
		// SearchParams.MaxDepth = ExpParams.NumSteps - t;
        int action = mcts->SelectAction(cumulativeReward);
		if (action > 4) num_check_action++; 
		terminal = Real.Step(*state, action, observation, reward);
		t++;

		if (!terminal && accumulate(reward.begin(), reward.end(), 0) > 0) 
		{
			collected_rock_num++;
			if (reward[0] < reward[1]) collected_type1_rocks_num++;
			else collected_type2_rocks_num++;
		}

		Results.Reward.Add(reward);
		for (int i =0; i < 2; i++){
			undiscountedReturn[i] += reward[i];
			discountedReturn[i] += reward[i] * discount;
			cumulativeReward[i] += reward[i];
        }
		discount *= Real.GetDiscount();

		if (SearchParams.Verbose >= 1)
		{
			Real.DisplayAction(action, cout);
			Real.DisplayState(*state, cout);
			Real.DisplayObservation(*state, observation, cout);
			Real.DisplayVectorReward(reward, cout);
		}

		if (terminal)
		{
			cout << "Terminated" << endl;
			break;
		}
		outOfParticles = !mcts->Update(action, observation, reward);
        if (outOfParticles) cout << "No particles!" << endl;
		if (outOfParticles)
			break;

		if (timer.elapsed() > ExpParams.TimeOut)
		{
			cout << "Timed out after " << collected_rock_num << " rocks collected in "
				<< Results.Time.GetTotal() << "seconds" << endl;
			break;
		}
	}

	if (outOfParticles)
	{
		cout << "Out of particles, finishing episode with SelectRandom" << endl;
		HISTORY history = mcts->GetHistory();
		while (++t < ExpParams.NumSteps)
		{
			int observation;
			vector<double> reward;

			// This passes real state into simulator!
			// SelectRandom must only use fully observable state
			// to avoid "cheating"
			int action = Simulator.SelectRandom(*state, history, mcts->GetStatus());
			if (action > 4) num_check_action++; 
			terminal = Real.Step(*state, action, observation, reward);

			if (!terminal && accumulate(reward.begin(), reward.end(), 0) > 0) 
			{
				collected_rock_num++;
				if (reward[0] < reward[1]) collected_type1_rocks_num++;
				else collected_type2_rocks_num++;
			}

			Results.Reward.Add(reward);
			for (int i =0; i < 2; i++){
				undiscountedReturn[i] += reward[i];
				discountedReturn[i] += reward[i] * discount;
			}
			discount *= Real.GetDiscount();

			if (SearchParams.Verbose >= 1)
			{
				Real.DisplayAction(action, cout);
				Real.DisplayState(*state, cout);
				Real.DisplayObservation(*state, observation, cout);
				Real.DisplayVectorReward(reward, cout);
			}

			if (terminal)
			{
				cout << "Terminated" << endl;
				break;
			}

			history.Add(action, observation);
		}
	}

	Results.Time.Add(timer.elapsed());
	Results.Timestep.Add(t);
	Results.GGFScore.Add(GGF(undiscountedReturn));
	Results.UndiscountedRewCV.Add(CV(undiscountedReturn));
	Results.DiscountedRewCV.Add(CV(discountedReturn));
	Results.UndiscountedReturn.Add(undiscountedReturn);
	Results.DiscountedReturn.Add(discountedReturn);
    Results.CollectedType1Rocks.Add(collected_type1_rocks_num);
    Results.CollectedType2Rocks.Add(collected_type2_rocks_num);
    Results.NumCheckAction.Add(num_check_action);
	cout << "num steps = " << t << endl;
	cout << "GGF score = " << GGF(undiscountedReturn) << endl;
	cout << "CV = " << CV(undiscountedReturn) << endl;
	cout << "Discounted return = " << discountedReturn
		<< ", average = " << Results.DiscountedReturn.GetMean() << endl;
	cout << "Undiscounted return = " << undiscountedReturn
		<< ", average = " << Results.UndiscountedReturn.GetMean() << endl;
    cout << "Number of type1 rocks collected = " << collected_type1_rocks_num << endl;
    cout << "Number of type2 rocks collected = " << collected_type2_rocks_num << endl;
    cout << "Number of check actions = " << num_check_action << endl;
    cout << "Number of steps = " << t << endl;
	delete mcts;
}

void EXPERIMENT::MultiRun()
{
	int numberOfRuns = ExpParams.NumRuns;
	for (int n = 0; n < numberOfRuns; n++)
	{
		cout << "Starting run " << n + 1 << " with "
			<< SearchParams.NumSimulations << " simulations... " << endl;
		Run();
		if (Results.Time.GetTotal() > ExpParams.TimeOut)
		{
			cout << "Timed out after " << n << " runs in "
				<< Results.Time.GetTotal() << "seconds" << endl;
			break;
		}
	}
}

void EXPERIMENT::DiscountedReturn()
{
	cout << "Main runs" << endl;
	OutputFile << "Simulations\tRuns\tUndiscounted return\tUndiscounted error\tDiscounted return\tDiscounted error\tTime\tUndiscounted CV\tUndiscounted CV error\tDiscounted CV\tDiscounted CV error\tTimesteps\tTimesteps error\tGGF score\tGGF score error\n";

    SearchParams.MaxDepth = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);
    ExpParams.SimSteps = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);
    ExpParams.NumSteps = Real.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);

	for (int i = ExpParams.MinDoubles; i <= ExpParams.MaxDoubles; i++)
	{
		SearchParams.NumSimulations = 1 << i;
		SearchParams.NumStartStates = 1 << i;
		if (i + ExpParams.TransformDoubles >= 0)
			SearchParams.NumTransforms = 1 << (i + ExpParams.TransformDoubles);
		else
			SearchParams.NumTransforms = 1;
		SearchParams.MaxAttempts = SearchParams.NumTransforms * ExpParams.TransformAttempts;

		Results.Clear();
		MultiRun();

		cout << "Simulations = " << SearchParams.NumSimulations << endl
			<< "Runs = " << Results.Time.GetCount() << endl
			<< "Undiscounted return = " << Results.UndiscountedReturn.GetMean()
			<< " +- " << Results.UndiscountedReturn.GetStdErr() << endl
			<< "Discounted return = " << Results.DiscountedReturn.GetMean()
			<< " +- " << Results.DiscountedReturn.GetStdErr() << endl
			<< "Time = " << Results.Time.GetMean() << endl
			<< "CV of Undiscounted return = " << Results.UndiscountedRewCV.GetMean()
			<< " +- " << Results.UndiscountedRewCV.GetStdErr() << endl
			<< "CV of discounted return = " << Results.DiscountedRewCV.GetMean()
			<< " +- " << Results.DiscountedRewCV.GetStdErr() << endl
			<< "Timesteps = " << Results.Timestep.GetMean()
			<< " +- " << Results.Timestep.GetStdErr() << endl
			<< "GGF score = " << Results.GGFScore.GetMean()
			<< " +- " << Results.GGFScore.GetStdErr() << endl
            << "Collected type1 rocks = " << Results.CollectedType1Rocks.GetMean()
            << " +- " << Results.CollectedType1Rocks.GetStdErr() << endl
            << "Collected type2 rocks = " << Results.CollectedType2Rocks.GetMean()
            << " +- " << Results.CollectedType2Rocks.GetStdErr() << endl
            << "Apply check actions = " << Results.NumCheckAction.GetMean()
            << " +- " << Results.NumCheckAction.GetStdErr() << endl;
		OutputFile << SearchParams.NumSimulations << "\t"
			<< Results.Time.GetCount() << "\t"
			<< Results.UndiscountedReturn.GetMean() << "\t"
			<< Results.UndiscountedReturn.GetStdErr() << "\t"
			<< Results.DiscountedReturn.GetMean() << "\t"
			<< Results.DiscountedReturn.GetStdErr() << "\t"
			<< Results.Time.GetMean() << "\t"
			<< Results.UndiscountedRewCV.GetMean() << "\t"
			<< Results.UndiscountedRewCV.GetStdErr() << "\t"
			<< Results.DiscountedRewCV.GetMean() << "\t"
			<< Results.DiscountedRewCV.GetStdErr() << "\t"
			<< Results.Timestep.GetMean() << "\t"
			<< Results.Timestep.GetStdErr() << "\t"
			<< Results.GGFScore.GetMean() << "\t"
			<< Results.GGFScore.GetStdErr() << endl;
	}
}

void EXPERIMENT::AverageReward()
{
	cout << "Main runs" << endl;
	OutputFile << "Simulations\tSteps\tAverage reward\tAverage time\n";

	ExpParams.SimSteps = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);

	for (int i = ExpParams.MinDoubles; i <= ExpParams.MaxDoubles; i++)
	{
		SearchParams.NumSimulations = 1 << i;
		SearchParams.NumStartStates = 1 << i;
		if (i + ExpParams.TransformDoubles >= 0)
			SearchParams.NumTransforms = 1 << (i + ExpParams.TransformDoubles);
		else
			SearchParams.NumTransforms = 1;
		SearchParams.MaxAttempts = SearchParams.NumTransforms * ExpParams.TransformAttempts;

		Results.Clear();
		Run();

		cout << "Simulations = " << SearchParams.NumSimulations << endl
			<< "Steps = " << Results.Reward.GetCount() << endl
			<< "Average reward = " << Results.Reward.GetMean()
			<< " +- " << Results.Reward.GetStdErr() << endl
			<< "Average time = " << Results.Time.GetMean() / Results.Reward.GetCount() << endl;
		OutputFile << SearchParams.NumSimulations << "\t"
			<< Results.Reward.GetCount() << "\t"
			<< Results.Reward.GetMean() << "\t"
			<< Results.Reward.GetStdErr() << "\t"
			<< Results.Time.GetMean() / Results.Reward.GetCount() << endl;
		OutputFile.flush();
	}
}

//----------------------------------------------------------------------------
