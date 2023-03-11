/*
Multi-load-unload problem, where the goal of the agent is to load items from a given location, 
then deliver them to different locations and unload.
As a multi-objective problem, each unload points will give a separate reward of 100.
It's also defined to be partially obervable, i.e., the agent can only sense the surrounding walls, 
but cannot perceive load or unload points.
*/

#ifndef MLU_H
#define MLU_H

#include "simulator.h"
#include "coord.h"
#include "grid.h"

class MLU_STATE: public STATE 
{
public:
    COORD AgentPos;
    bool IsLoaded;
};

class MLU: public SIMULATOR
{
public:
    MLU(int numUnloadPos, int xsize = 8, int ysize = 3);

	virtual std::string GetClassName() const;
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
    // observations
	enum
	{
		E_OBS1,
		E_OBS2,
		E_OBS3,
        E_OBS4,
        E_OBS5,
        E_OBS6,
        E_OBS7
	};

	enum
	{
		// two actuator actions
		E_LOAD = 4,
        E_UNLOAD
	};

	void InitStandardMap();
	int GetObservation(const MLU_STATE& mlustate) const;

	GRID<int> Maze;
	std::vector<COORD> UnloadPos;
    COORD LoadPos;
	int xSize, ySize;
	int NumUnloadPos;
	COORD StartPos;

private:

	mutable MEMORY_POOL<MLU_STATE> MemoryPool;

};



#endif // MLU_H