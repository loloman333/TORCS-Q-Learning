from QLearner import QLearner
from types import SimpleNamespace
import numpy as np
import SimplePythonClient.CarControl as CarControl
import SimplePythonClient.CarState as CarState

class QSteerer(QLearner):

    EXPORT_PATH: str = "./steerer.export"
    QTABLE_EXPORT_PATH: str = "./steerer.qtable"
    
    substates: dict[str, SimpleNamespace] = {
        "track_position" : SimpleNamespace(count=21, min=-1.3, max=1.3),
        "angle" : SimpleNamespace(count=21, min=-np.pi/2, max=np.pi/2)
    }

    # accel, brake, gear, steer, clutch, focus, meta
    steering_angles = [
        0,    # Straight
        0.35, # Left
        -0.35 # Right
    ]

    def __init__(self, epsilon, alpha, gamma, epsilon_change, epsilon_min) -> None:
        num_states = 1
        for substate in self.substates.values():
            num_states *= substate.count
        
        super().__init__(num_states, len(self.steering_angles), epsilon, alpha, gamma, epsilon_change, epsilon_min)

    def get_state(self, observation: CarState.CarState):
        
        track_pos = observation.getTrackPos()
        track_pos_state = self.get_substate(track_pos, "track_position")

        angle = observation.getAngle()
        angle_state = self.get_substate(angle, "angle")

        return self.combine_substates(
            [track_pos_state, angle_state]
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
    
    def getSteer(self, cs: CarState.CarState):
        return self.steering_angles[self.policy(cs)]

