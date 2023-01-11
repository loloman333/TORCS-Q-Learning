from QLearner import QLearner
from types import SimpleNamespace
import numpy as np
import SimplePythonClient.CarControl as CarControl
import SimplePythonClient.CarState as CarState

class QSteerer(QLearner):

    EXPORT_PATH: str = "./steerer.export"
    QTABLE_EXPORT_PATH: str = "./steerer.qtable"
    
    substates = [
        SimpleNamespace(name="track_posistion", count=21, min=-1.3, max=1.3),
        SimpleNamespace(name="angle", count=21, min=-np.pi/2, max=np.pi/2)
    ]

    # accel, brake, gear, steer, clutch, focus, meta
    controls = [
        "Straight", #CarControl.CarControl(0.25, 0, 1, 0, 0, 0, 0), # Straight
        "Left",     #CarControl.CarControl(0.25, 0, 1, 0.35, 0, 0, 0), # Left
        "Right",    #CarControl.CarControl(0.25, 0, 1, -0.35, 0, 0, 0), # Right
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
        
        track_pos = observation.getTrackPos()
        #track_pos_state = self.get_substate(track_pos, self.track_pos_state_count, 2 * 2 / self.track_pos_state_count)
        track_pos_state = self.get_substate(track_pos, 0, None)

        angle = observation.getAngle()
        #angle_state = self.get_substate(angle, self.angle_state_count, 3.1415 * 2 / self.angle_state_count)
        angle_state = self.get_substate(angle, 1, None)

        return self.combine_substates(
            [track_pos_state, self.substates[0].count],
            [angle_state, self.substates[1].count],
        )
    
    def getRewardScore(self, obs: CarState.CarState):
        track_reward = 10 * (1 - abs(obs.getTrackPos()))
        return track_reward, 1 
    
    def learn(self, cs: CarState.CarState):

        if (abs(cs.getTrackPos()) >= 1.2):
            super().learn(cs, -1000, 1)
            return

        if (self.last_action != None and self.last_state != None):
            reward, score = self.getRewardScore(cs)
            super().learn(cs, reward, score)
