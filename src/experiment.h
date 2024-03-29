#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "mcts.h"
#include "simulator.h"
#include "statistic.h"
#include "vectorstatistic.h"
#include "rocksample.h"
#include <fstream>
#include <numeric>
// #include "planner.h"

//----------------------------------------------------------------------------

struct RESULTS
{
	void Clear();

	STATISTIC Time;
	STATISTIC UndiscountedRewCV;
	STATISTIC DiscountedRewCV;
	STATISTIC GGFScore;
	STATISTIC Timestep;
	VECTORSTATISTIC Reward = VECTORSTATISTIC(2);
	VECTORSTATISTIC DiscountedReturn = VECTORSTATISTIC(2);
	VECTORSTATISTIC UndiscountedReturn = VECTORSTATISTIC(2);
    STATISTIC MaxNumberOfBandits;
};

inline void RESULTS::Clear()
{
	Time.Clear();
	UndiscountedRewCV.Clear();
	DiscountedRewCV.Clear();
	GGFScore.Clear();
	Timestep.Clear();
	Reward.Clear();
	DiscountedReturn.Clear();
	UndiscountedReturn.Clear();
}

//----------------------------------------------------------------------------

class EXPERIMENT
{
public:

	struct PARAMS
	{
		PARAMS();

		int NumRuns;
		int NumSteps;
		int SimSteps;
		double TimeOut;
		int MinDoubles, MaxDoubles;
		int TransformDoubles;
		int TransformAttempts;
		double Accuracy;
		int UndiscountedHorizon;
		bool AutoExploration;
		bool usePOSTS;
		bool ggi;
		bool ws; // weighted sum
	};

	EXPERIMENT(const SIMULATOR& real, const SIMULATOR& simulator,
		const std::string& outputFile,
		EXPERIMENT::PARAMS& expParams, MCTS::PARAMS& searchParams);

	void Run();
	void MultiRun();
	void DiscountedReturn();
	void AverageReward();

private:

	const SIMULATOR& Real;
	const SIMULATOR& Simulator;
	EXPERIMENT::PARAMS& ExpParams;
	MCTS::PARAMS& SearchParams;
	RESULTS Results;

	std::ofstream OutputFile;
};

//----------------------------------------------------------------------------

#endif // EXPERIMENT_H
