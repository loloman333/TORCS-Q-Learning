import SimplePythonClient.BaseDriver as BaseDriver
import SimplePythonClient.CarState as CarState
import SimplePythonClient.CarControl as CarControl
from QSteerer import QSteerer
from QAccelerator import QAccelerator

class QDriver(BaseDriver.BaseDriver):

    def __init__(self, steerer: QSteerer, accelerator: QAccelerator):
        self.steerer = steerer
        self.accelerator = accelerator
        self.keep_learning = True
        self.lap = 0
        self.curLapTime = 0
        super().__init__()
        
    def onShutdown(self):
        self.endEpisode()

    def onRestart(self):
        self.endEpisode()
    
    def endEpisode(self):
        self.lap = 0
        self.curLapTime = 0
        self.steerer.end_episode()
        self.accelerator.end_episode()

    def lapChange(self, cs: CarState.CarState):
        lap_change = 1 if cs.getCurLapTime() < self.curLapTime else 0
        self.curLapTime = cs.getCurLapTime()
        return lap_change       

    def Update(self, buffer):
        cs = CarState.CarState(buffer)
        self.lap += self.lapChange(cs)

        self.steerer.learn(cs)
        self.accelerator.learn(cs)

        cc = self.__wDrive(cs)
        return str(cc)   
    
    def __wDrive(self, currentCarState: CarState.CarState):

        if self.keep_learning and abs(currentCarState.getTrackPos()) >= 1.2 or self.lap >=4:
            return CarControl.CarControl(0,0,0,0,0,0,1)
        
        gear = self.getGear(currentCarState)
        steer = self.steerer.getSteer(currentCarState)
        gas, brake = self.accelerator.getAccel(currentCarState)

        return CarControl.CarControl(gas, brake, gear, steer, 0, 0, 0)

    def stopLearning(self):
        self.keep_learning = False
        self.steerer.stop_learning()
        self.accelerator.stop_learning()
 