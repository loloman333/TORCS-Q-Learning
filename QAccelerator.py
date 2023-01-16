from QLearner import QLearner
from types import SimpleNamespace
import numpy as np
import SimplePythonClient.CarState as CarState

class QAccelerator(QLearner):

    EXPORT_PATH: str = "./accelerator.export"
    QTABLE_EXPORT_PATH: str = "./accelerator.qtable"
    
    substates: dict = {
        "track_laser" : SimpleNamespace(count=20, min=0, max=100),
        "speed" : SimpleNamespace(count=20, min=0, max=200)
    }

    current_gas = 1
    current_brake = 0
    accel_values = [
        (0, -1)     # Hold
        (0.01, -1)  # Increase
        (-0.01, -1) # Reduce
        (0, 0.01)   # Brake
    ]

    def __init__(self, epsilon, alpha, gamma, epsilon_change, epsilon_min) -> None:
        num_states = 1
        for substate in self.substates.values():
            num_states *= substate.count
        super().__init__(num_states, len(self.controls), epsilon, alpha, gamma, epsilon_change, epsilon_min)

    def get_state(self, observation: CarState.CarState):
        
        track_sensor = observation.getTrack(9)
        track_sensor_state = self.get_substate(track_sensor, "track_laser")

        speedX = observation.getSpeedX()
        speedX_state = self.get_substate(speedX, "speed")

        return self.combine_substates(
            [track_sensor_state, speedX_state]
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

    def getAccel(self, cs: CarState.CarState):
        gas_change, brake_change = self.accel_values[self.policy(cs)]
        self.current_gas += gas_change
        self.current_brake += brake_change
        return self.current_gas, self.current_brake
