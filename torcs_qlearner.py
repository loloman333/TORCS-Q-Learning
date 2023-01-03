import SimplePythonClient.BaseDriver as BaseDriver
import SimplePythonClient.CarState as CarState
import SimplePythonClient.CarControl as CarControl
from types import SimpleNamespace
import numpy as np

from q_learner import QLearner

NUMBEROFRANGEFINDERSENSORS = 19

class QDriver(QLearner, BaseDriver.BaseDriver):

    substates = [
        SimpleNamespace(name="track_posistion", count=21, min=-1.3, max=1.3),
        SimpleNamespace(name="angle", count=21, min=-np.pi/2, max=np.pi/2)
    ]
    
    # accel, brake, gear, steer, clutch, focus, meta
    controls = [
        CarControl.CarControl(0.25, 0, 1, 0, 0, 0, 0), # Straight
        CarControl.CarControl(0.25, 0, 1, 0.35, 0, 0, 0), # Left
        CarControl.CarControl(0.25, 0, 1, -0.35, 0, 0, 0), # Right
    ]

    def __init__(self, epsilon, alpha, gamma, epsilon_change, epsilon_min):
        self.steeringWheel = 0.0

        num_states = 1
        for substate in self.substates:
            num_states *= substate.count
            substate.bins = np.linspace(substate.min, substate.max, substate.count + 1)
        
        super().__init__(num_states, len(self.controls), epsilon, alpha, gamma, epsilon_change, epsilon_min)
    
    @staticmethod
    def _import() -> QLearner:
        learner = super(QDriver, QDriver)._import()

        num_states = 1
        for substate in learner.substates:
            num_states *= substate.count
            substate.bins = np.linspace(substate.min, substate.max, substate.count + 1)
        learner.num_states = num_states

        return learner
        
    def onShutdown(self):            
        self.end_episode()

    def onRestart(self):
        self.end_episode()
        
    def getInitAngles(self):
        return [0]

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
        
        #if obs.getDistRaced() > self.max_dist_raced:
        #    self.max_dist_raced = obs.getDistRaced()

        track_reward = 10 * (1 - abs(obs.getTrackPos()))
        #dist_reward = 10 * (obs.getDistRaced() / self.max_dist_raced)

        return track_reward, 1 

    def Update(self, buffer):
        cs = CarState.CarState(buffer)

        if (abs(cs.getTrackPos()) >= 1.2):
            self.learn(cs, -1000, 1)
            #return str(CarControl.CarControl(0,0,0,0,0,0,1))

        if (self.last_action != None and self.last_state != None):
            reward, score = self.getRewardScore(cs)
            self.learn(cs, reward, score)

        cc = self.__wDrive(cs)

        return str(cc)   
    
    def action_to_car_control(self, action):
        return self.controls[action]
    
    # put the intelligence here    
    def __wDrive(self, currentCarState):
        action = self.policy(currentCarState)
        return self.action_to_car_control(action)
 