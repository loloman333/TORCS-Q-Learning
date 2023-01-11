import SimplePythonClient.BaseDriver as BaseDriver
import SimplePythonClient.CarState as CarState
import SimplePythonClient.CarControl as CarControl
from QSteerer import QSteerer
from QAccelerator import QAccelerator

class QDriver(BaseDriver.BaseDriver):

    def __init__(self, steerer: QSteerer, accelerator: QAccelerator):
        self.steerer = steerer
        self.accelerator = accelerator
        super().__init__()
        
    def onShutdown(self):        
        self.steerer.end_episode()
        self.accelerator.end_episode()

    def onRestart(self):
        self.steerer.end_episode()
        self.accelerator.end_episode()
        
    def getInitAngles(self):
        return [-90,-80,-70,-60,-50,-40,-30,-20,-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def Update(self, buffer):
        cs = CarState.CarState(buffer)
        self.steerer.learn(cs)
        self.accelerator.learn(cs)
        cc = self.__wDrive(cs)
        return str(cc)   

    # accel, brake, gear, steer, clutch, focus, meta
    controls = [
         CarControl.CarControl(0.25, 0, 1, 0, 0, 0, 0), # Straight
         CarControl.CarControl(0.25, 0, 1, 0.35, 0, 0, 0), # Left
         CarControl.CarControl(0.25, 0, 1, -0.35, 0, 0, 0), # Right
    ]
    current_gas = 1
    current_break = 0
    current_gear = 1

    def action_to_car_control(self, steer_action, accelerate_action):
        action = self.controls[steer_action]

        match accelerate_action:
            case 0: # Hold
                pass
            case 1: # Increase Gas
                self.current_gas += 0.5
                self.current_break = 0
            case 2: # Decrease Gas
                self.current_gas -= 0.05
                self.current_break = 0
            case 3: # Break
                self.current_gas = 0
                self.current_break = 1
        
        self.current_gas = 1 if self.current_gas > 1 else self.current_gas
        self.current_gas = 0 if self.current_gas < 0 else self.current_gas

        action.accel = self.current_gas
        action.brake = self.current_break
        action.gear = self.current_gear

        return action
    
    # put the intelligence here    
    def __wDrive(self, currentCarState: CarState.CarState):

        if currentCarState.getTrackPos() >= 1.2:
            return CarControl.CarControl(0,0,0,0,0,0,0,1)
        
        if currentCarState.getRpm() >= 4500:
            self.current_gear += 1
            self.current_gear = 7 if self.current_gear > 7 else self.current_gear
        if currentCarState.getRpm() <= 1000:
            self.current_gear -= 1

        self.current_gear = 1 if self.current_gear < 1 else self.current_gear

        steer_action = self.steerer.policy(currentCarState)
        accelerate_action = self.accelerator.policy(currentCarState)

        return self.action_to_car_control(steer_action, accelerate_action)
 