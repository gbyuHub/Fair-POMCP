#include "mcts.h"
#include "rocksample.h"
#include "mlu.h"
#include "randomdomain.h"
#include "mwa.h"
#include "experiment.h"
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;

void disableBufferedIO(void)
{
    setbuf(stdout, NULL);
    setbuf(stdin, NULL);
    setbuf(stderr, NULL);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
}

int main(int argc, char* argv[])
{
    MCTS::PARAMS searchParams;
    EXPERIMENT::PARAMS expParams;
    SIMULATOR::KNOWLEDGE knowledge;
    string problem, outputfile, policy;
    int size, number, treeknowledge = 1, rolloutknowledge = 1, smarttreecount = 10;
    int numProducts = 5;
    double smarttreevalue = 1.0;

    options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("test", "run unit tests")
        ("problem", value<string>(&problem), "problem to run")
        ("outputfile", value<string>(&outputfile)->default_value("output.txt"), "summary output file")
		("strategy", value<string>(&searchParams.Strategy)->default_value("GGF"), "action selection strategy")
        ("considerpast", value<bool>(&searchParams.ConsiderPast), "Consider the past returns or not")
        ("policy", value<string>(&policy), "policy file (explicit POMDPs only)")
        ("size", value<int>(&size), "size of problem (problem specific)")
        ("number", value<int>(&number), "number of elements in problem (problem specific)")
        ("numproducts", value<int>(&numProducts)->default_value(5), "number of products in MWA problem")
        ("timeout", value<double>(&expParams.TimeOut), "timeout (seconds)")
        ("mindoubles", value<int>(&expParams.MinDoubles), "minimum power of two simulations")
        ("maxdoubles", value<int>(&expParams.MaxDoubles), "maximum power of two simulations")
        ("runs", value<int>(&expParams.NumRuns), "number of runs")
        ("accuracy", value<double>(&expParams.Accuracy), "accuracy level used to determine horizon")
        ("horizon", value<int>(&expParams.UndiscountedHorizon), "horizon to use when not discounting")
        ("maxdepth", value<int>(&searchParams.MaxDepth), "Max depth of planning tree")
        ("num steps", value<int>(&expParams.NumSteps), "number of steps to run when using average reward")
        ("verbose", value<int>(&searchParams.Verbose), "verbosity level")
        ("autoexploration", value<bool>(&expParams.AutoExploration), "Automatically assign UCB exploration constant")
        ("exploration", value<double>(&searchParams.ExplorationConstant), "Manual value for UCB exploration constant")
        ("usetransforms", value<bool>(&searchParams.UseTransforms), "Use transforms")
        ("transformdoubles", value<int>(&expParams.TransformDoubles), "Relative power of two for transforms compared to simulations")
        ("transformattempts", value<int>(&expParams.TransformAttempts), "Number of attempts for each transform")
        ("userave", value<bool>(&searchParams.UseRave), "RAVE")
        ("ravediscount", value<double>(&searchParams.RaveDiscount), "RAVE discount factor")
        ("raveconstant", value<double>(&searchParams.RaveConstant), "RAVE bias constant")
        ("treeknowledge", value<int>(&knowledge.TreeLevel), "Knowledge level in tree (0=Pure, 1=Legal, 2=Smart)")
        ("rolloutknowledge", value<int>(&knowledge.RolloutLevel), "Knowledge level in rollouts (0=Pure, 1=Legal, 2=Smart)")
        ("smarttreecount", value<int>(&knowledge.SmartTreeCount), "Prior count for preferred actions during smart tree search")
        ("smarttreevalue", value<double>(&knowledge.SmartTreeValue), "Prior value for preferred actions during smart tree search")
        ("disabletree", value<bool>(&searchParams.DisableTree), "Use 1-ply rollout action selection")
        ;

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help"))
    {
        cout << desc << "\n";
        return 1;
    }

    if (vm.count("problem") == 0)
    {
        cout << "No problem specified" << endl;
        return 1;
    }

    SIMULATOR* real = 0;
    SIMULATOR* simulator = 0;
    if (problem == "rocksample")
    {
        real = new ROCKSAMPLE(size, number, 2);
        simulator = new ROCKSAMPLE(size, number, 2);
        expParams.NumObjectives = 2;
        searchParams.NumObjectives = 2;
    }
    else if (problem == "mlu") {
        real = new MLU(5);
        simulator = new MLU(5);
        expParams.NumObjectives = 5;
        searchParams.NumObjectives = 5;
    }
    else if (problem == "mwa") {
        real = new MWA(numProducts);
        simulator = new MWA(numProducts);
        expParams.NumObjectives = numProducts;
        searchParams.NumObjectives = numProducts;
    }
    else if (problem == "random") {
        real = new RANDOMENV();
        simulator = new RANDOMENV();
        expParams.NumObjectives = 2;
        searchParams.NumObjectives = 2;
    }
    else 
    {
        cout << "Unknown problem" << endl;
        exit(1);
    }


    simulator->SetKnowledge(knowledge);
    EXPERIMENT experiment(*real, *simulator, outputfile, expParams, searchParams);
    experiment.DiscountedReturn();

    delete real;
    delete simulator;
    return 0;
}
