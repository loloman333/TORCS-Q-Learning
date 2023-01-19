from QLearner import QLearner
from types import SimpleNamespace
import SimplePythonClient.CarState as CarState

class StaticAccelerator:

    def learn(self, cs):
        pass

    def end_episode(self):
        pass

    def getAccel(self, cs):
        return 0.25, 0

class QAccelerator(QLearner):

    EXPORT_PATH: str = "./accelerator.export"
    QTABLE_EXPORT_PATH: str = "./accelerator.qtable"
    
    substates: dict = {
        "track_laser" : SimpleNamespace(count=10, min=0, max=100, power=1),
        "speed" : SimpleNamespace(count=10, min=0, max=200, power=1)
    }

    # current_gas = 1
    # current_brake = 0
    accel_values = [
        (0, 0),      # Roll
        (0.25, 0),   # Low
        (1, 0),      # High
        (0, 0.2)     # Brake
    ]

    def __init__(self, epsilon, alpha, gamma, epsilon_change, epsilon_min) -> None:
        num_states = 1
        for substate in self.substates.values():
            num_states *= substate.count
        super().__init__(num_states, len(self.accel_values), epsilon, alpha, gamma, epsilon_change, epsilon_min)

    def get_state(self, observation: CarState.CarState):
        
        track_sensor = observation.getTrack(9)
        track_sensor_state = self.get_substate(track_sensor, "track_laser")

        speedX = observation.getSpeedX()
        speedX_state = self.get_substate(speedX, "speed")

        return self.combine_substates(
            [track_sensor_state, speedX_state]
        )
    
    def getRewardScore(self, obs: CarState.CarState):
        if (abs(obs.getTrackPos()) >= 1.1):
            return -10000, 1

        speed_reward = 10 * (obs.getSpeedX() / 200)
        track_reward = 10 * (1 - abs(obs.getTrackPos()))

        return speed_reward + track_reward, 1 
    
    def learn(self, cs: CarState.CarState):
        if (self.last_action != None and self.last_state != None):
            reward, score = self.getRewardScore(cs)
            super().learn(cs, reward, score)

    def getAccel(self, cs: CarState.CarState):
        gas, brake = self.accel_values[self.policy(cs)]
        #self.current_gas += gas_change
        #self.current_brake += brake_change

        #self.current_gas = 1 if self.current_gas > 1 else 0 if self.current_gas < 0 else self.current_gas
        #self.current_brake = 1 if self.current_brake > 1 else 0 if self.current_brake < 0 else self.current_brake

        return gas, brake
