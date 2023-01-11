from QLearner import QLearner
from types import SimpleNamespace
import numpy as np
import SimplePythonClient.CarState as CarState

class QAccelerator(QLearner):

    EXPORT_PATH: str = "./accelerator.export"
    QTABLE_EXPORT_PATH: str = "./accelerator.qtable"
    
    substates = [
        SimpleNamespace(name="track_laser", count=20, min=0, max=100),
        SimpleNamespace(name="speed", count=20, min=0, max=200)
    ]

    controls = [
        "Hold",
        "Increase",
        "Reduce",
        "Brake",
    ]

    def __init__(self, epsilon, alpha, gamma, epsilon_change, epsilon_min) -> None:
        num_states = 1
        for substate in self.substates:
            num_states *= substate.count
            substate.bins = np.linspace(substate.min, substate.max, substate.count + 1)
        
        super().__init__(num_states, len(self.controls), epsilon, alpha, gamma, epsilon_change, epsilon_min)
    
    # TODO PFUSCH!!!!
    def get_substate(self, value, state_count, state_length):
        substate_index = state_count

        substate = np.digitize(value, self.substates[0].bins) - 1
        substate = 0 if substate < 0 else substate
        substate = self.substates[substate_index].count - 1 if substate > self.substates[substate_index].count - 1 else substate

        assert substate >= 0 and substate <= self.substates[substate_index].count - 1

        return substate

    def get_state(self, observation: CarState.CarState):
        
        track_sensor = observation.getTrack(9)
        track_sensor_state = self.get_substate(track_sensor, 0, None)

        speedX = observation.getSpeedX()
        speedX_state = self.get_substate(speedX, 1, None)

        return self.combine_substates(
            [track_sensor_state, self.substates[0].count],
            [speedX_state, self.substates[1].count],
        )
    
    def getRewardScore(self, obs: CarState.CarState):
        track_reward = 10 * obs.getSpeedX()
        return track_reward, 1 
    
    def learn(self, cs: CarState.CarState):

        if (abs(cs.getTrackPos()) >= 1.1):
            super().learn(cs, -10000, 1)
            return

        if (self.last_action != None and self.last_state != None):
            reward, score = self.getRewardScore(cs)
            super().learn(cs, reward, score)
