#include "mlu.h"
#include "utils.h"

using namespace std;
using namespace UTILS;

MLU::MLU(int numUnloadPos, int xsize /*=8*/, int ysize /*=3*/)
    : Maze(xsize, ysize),
    xSize(xsize),
    ySize(ysize),
    NumUnloadPos(numUnloadPos)
{
	NumActions = 6;
	NumObservations = 7;
	RewardRange = 100;
	Discount = 0.95;

    InitStandardMap();
}

void MLU::InitStandardMap()
{
    cout << "Using standard layout with 5 upload points." << endl;
    int maze[3][8] = 
    {
        {-1, 3, -1, -1, 3, -1, -1, -1},
        {3, 0, 0, 0, 0, 0, 4, 2},
        {-1, 3, -1, -1, 3, -1, -1, -1}
    };
    for (int x = 0; x < 3; x++) {
        Maze.SetRow(x, maze[3-x]);
    }
    StartPos = COORD(4, 1);

}   


STATE* MLU::CreateStartState() const
{
    MLU_STATE* mlustate = MemoryPool.Allocate();
    int x, y;
    while (true) {
        x = Random(xSize);
        y = Random(ySize);
        if (Maze(x, y) != -1) {
            mlustate->AgentPos = COORD(x, y);
            break;
        }
    }
    mlustate->IsLoaded = false;
    return mlustate;
}

STATE* MLU::Copy(const STATE& state) const
{

}

void MLU::Validate(const STATE& state) const
{

}

void MLU::FreeState(STATE* state) const
{

}

bool MLU::Step(STATE& state, int action,
		int& observation, std::vector<double>& reward) const
{

}

void MLU::GenerateLegal(const STATE& state, const HISTORY& history,
		std::vector<int>& legal, const STATUS& status) const
{

}

bool MLU::LocalMove(STATE& state, const HISTORY& history,
		int stepObservation, const STATUS& status) const
{

}

int MLU::GetObservation(const MLU_STATE& mlustate) const
{

}

void MLU::DisplayBeliefs(const BELIEF_STATE& beliefState,
		std::ostream& ostr) const
{

}

void MLU::DisplayState(const STATE& state, std::ostream& ostr) const
{

}

void MLU::DisplayObservation(const STATE& state, int observation, std::ostream& ostr) const
{

}

void MLU::DisplayAction(int action, std::ostream& ostr) const
{

}

