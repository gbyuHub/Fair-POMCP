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
	NumObservations = 1 << 4;
    NumObjectives = numUnloadPos;
	RewardRange = 100;
	Discount = 1.0;

    InitStandardMap();
}

void MLU::InitStandardMap()
{
    cout << "Using standard layout with 5 unload points." << endl;
    int maze[3][8] = 
    {
        {-1, 3, -1, -1, 3, -1, -1, -1},
        {3, 0, 0, 0, 0, 0, 4, 2},
        {-1, 3, -1, -1, 3, -1, -1, -1}
    };
    for (int x = 0; x < 3; x++) {
        Maze.SetRow(x, maze[2-x]);
    }
    LoadPos = COORD(7, 1);
    UnloadPos = {COORD(0, 1), COORD(1, 2), COORD(1, 0), COORD(4, 2), COORD(4, 0)};
    StartPos = COORD(5, 1);
}   


STATE* MLU::CreateStartState() const
{
    MLU_STATE* mlustate = MemoryPool.Allocate();
    // int x, y;
    // while (true) {
    //     x = Random(xSize);
    //     y = Random(ySize);
    //     if (Maze(x, y) != -1) {
    //         mlustate->AgentPos = COORD(x, y);
    //         break;
    //     }
    // }
    mlustate->AgentPos = StartPos;
    mlustate->IsLoaded = false;
    return mlustate;
}

std::string MLU::GetClassName() const
{
    return "MLU";
}

STATE* MLU::Copy(const STATE& state) const
{
	const MLU_STATE& mlustate = safe_cast<const MLU_STATE&>(state);
	MLU_STATE* newstate = MemoryPool.Allocate();
	*newstate = mlustate;
	return newstate;
}

void MLU::Validate(const STATE& state) const
{
    const MLU_STATE& mlustate = safe_cast<const MLU_STATE&>(state);
    assert(Maze(mlustate.AgentPos) != -1);
}

void MLU::FreeState(STATE* state) const
{
    MLU_STATE* mlustate = safe_cast<MLU_STATE*>(state);
    MemoryPool.Free(mlustate);
}

bool MLU::Step(STATE& state, int action,
		int& observation, std::vector<double>& reward) const
{
    MLU_STATE& mlustate = safe_cast<MLU_STATE&>(state);
    int x = mlustate.AgentPos.X, y = mlustate.AgentPos.Y;
    std::fill(reward.begin(), reward.end(), 0.0);
    
    // move actions
    if (action < 4) {
        switch (action)
        {
        case COORD::E_NORTH:
            if (y + 1 < ySize && Maze(x, y+1) != -1) {
                mlustate.AgentPos.Y++;
            }
            else {
                assert(false && "Illegal action: NORTH!");
            }
            break;
        
        case COORD::E_EAST:
            if (x + 1 < xSize && Maze(x+1, y) != -1) {
                mlustate.AgentPos.X++;
            }
            else {
                assert(false && "Illegal action: EAST!");
            }
            break;
        case COORD::E_SOUTH:
            if (y - 1 >= 0 && Maze(x, y-1) != -1) {
                mlustate.AgentPos.Y--;
            }
            else {
                assert(false && "Illegal action: SOUTH!");
            }
            break;
        case COORD::E_WEST:
            if (x - 1 >= 0 && Maze(x-1, y) != -1) {
                mlustate.AgentPos.X--;
            }
            else {
                assert(false && "Illegal action: WEST!");
            }
            break;
        }
    }

    if (action == E_LOAD) {
        if (Maze(x, y) == 2 && !mlustate.IsLoaded) {
            mlustate.IsLoaded = true;
        }
        else {
            assert(false && "Illegal action: LOAD!");
        }
    }

    if (action == E_UNLOAD) {
        if (Maze(x, y) == 3 && mlustate.IsLoaded) {
            for (int i = 0; i < NumUnloadPos; i++) {
                if (UnloadPos[i] == mlustate.AgentPos) {
                    mlustate.IsLoaded = false;
                    reward[i] += 10;
                    break;
                }
            }
        }
        else {
            assert(false && "Illegal action: UNLOAD!");
        }
    }

    observation = GetObservation(mlustate);
    // assume no exit state
    return false;
}

void MLU::GenerateLegal(const STATE& state, const HISTORY& history,
		std::vector<int>& legal, const STATUS& status) const
{
    const MLU_STATE& mlustate = safe_cast<const MLU_STATE&>(state);
    COORD agent_pos = mlustate.AgentPos;
    int x = agent_pos.X, y = agent_pos.Y;
    if (y + 1 < ySize && Maze(x, y+1) != -1) {
        legal.push_back(COORD::E_NORTH);
    }
    if (x + 1 < xSize && Maze(x+1, y) != -1) {
        legal.push_back(COORD::E_EAST);
    }
    if (x - 1 >= 0 && Maze(x-1, y) != -1) {
        legal.push_back(COORD::E_WEST);
    }
    if (y - 1 >= 0 && Maze(x, y-1) != -1) {
        legal.push_back(COORD::E_SOUTH);
    }
    if (Maze(agent_pos) == 2 && !mlustate.IsLoaded) {
        legal.push_back(E_LOAD);
    }
    if (Maze(agent_pos) == 3 && mlustate.IsLoaded) {
        legal.push_back(E_UNLOAD);
    }
}

bool MLU::LocalMove(STATE& state, const HISTORY& history,
		int stepObservation, const STATUS& status) const
{
    MLU_STATE& mlustate = safe_cast<MLU_STATE&>(state);
    // randomly move the agent to a new position
    int x, y;
    while (true) {
        x = Random(xSize);
        y = Random(ySize);
        if (Maze(x, y) != -1 && (x != mlustate.AgentPos.X || y != mlustate.AgentPos.Y)) {
            mlustate.AgentPos = COORD(x, y);
            break;
        }
    }
    if (history.Back().Action < 4) {
        int realObs = history.Back().Observation;
        int newObs = GetObservation(mlustate);
        if (newObs != realObs) {
            return false;
        }
    }
    return true;
}

int MLU::GetObservation(const MLU_STATE& mlustate) const
{
    int x = mlustate.AgentPos.X, y = mlustate.AgentPos.Y;
    // 4-bits obs, from left to right, north -> east -> south -> west, 0 if there is a wall otherwise 1
    int obs = 0;
    if (y + 1 < ySize && Maze(x, y+1) != -1) {
        SetFlag(obs, 0);
    }
    if (x + 1 < xSize && Maze(x+1, y) != -1) {
        SetFlag(obs, 1);
    }
    if (y - 1 >= 0 && Maze(x, y-1) != -1) {
        SetFlag(obs, 2);
    }
    if (x - 1 >= 0 && Maze(x-1, y) != -1) {
        SetFlag(obs, 3);
    }
    return obs;
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

