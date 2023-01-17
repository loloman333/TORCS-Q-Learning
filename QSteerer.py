from QLearner import QLearner
from types import SimpleNamespace
import numpy as np
import SimplePythonClient.CarState as CarState

class QSteerer(QLearner):

    EXPORT_PATH: str = "./steerer.export"
    QTABLE_EXPORT_PATH: str = "./steerer.qtable"
    
    substates: dict[str, SimpleNamespace] = {
        "track_position" : SimpleNamespace(count=21, min=-1.3, max=1.3, power=1),
        "angle" : SimpleNamespace(count=21, min=-np.pi/2, max=np.pi/2, power=1)
    }

    # Old
    steering_angles = [
        0,    # Straight
        0.35, # Left
        -0.35 # Right
    ]

    # Alternative Idea:
    # steering_changes = [
    #    0,     # Hold,
    #    0.01,  # To the left
    #    -0.01  # To the right
    #]

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
        #if (abs(obs.getTrackPos()) >= 1.2):
        #    return -1000, 1

        track_reward = 10 * (1 - abs(obs.getTrackPos()))
        track_reward = np.sign(track_reward) * track_reward ** 2 
        
        return track_reward, 1 
    
    def learn(self, cs: CarState.CarState):
        if (self.last_action != None and self.last_state != None):
            reward, score = self.getRewardScore(cs)
            super().learn(cs, reward, score)
    
    # current_steer = 0
    def getSteer(self, cs: CarState.CarState):
        # self.current_steer += self.steering_changes[self.policy(cs)]
        # self.current_steer = np.pi if self.current_steer > np.pi else -np.pi if self.current_steer < -np.pi else self.current_steer
        return self.steering_angles[self.policy(cs)]

