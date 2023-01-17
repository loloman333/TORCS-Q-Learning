'''
This class is based on the C++ code of Daniele Loiacono

Created on  03.05.2011
@author: Thomas Fischle
@contact: fisch27@gmx.de
'''

import socket
import subprocess
import time 
import datetime
import SimplePythonClient.BaseDriver as BaseDriver
import SimplePythonClient.SimpleParser as SimpleParser
from QDriver import QDriver
from QSteerer import QSteerer
from QAccelerator import QAccelerator, StaticAccelerator
   
TORCS_PATH = "C:\\Program Files (x86)\\torcs"
SERVER_IP = "127.0.0.1"
CLIENT_PORT = 3002

# Maximal size of file read from socket
BUFSIZE = 1024

#Set default values
maxEpisodes=0
maxSteps=100000
serverPort=3001
hostName = "localhost"
id = "championship2011"
stage = BaseDriver.tstage.WARMUP
trackName = "unknown"

#    noise=false
#    noiseAVG=0
#    noiseSTD=0.05
#    seed=0

server: subprocess.Popen = None
logfile = open("./server.log", "w")

def start_server(visual):
    global server
    if server is not None:
        server.kill()
    
    args: str
    if visual:
        args = "-t 15000000"
    else:
        args = "-T -t 15000000"

    server = subprocess.Popen(f"{TORCS_PATH}\\wtorcs.exe {args}", close_fds=True, cwd=TORCS_PATH, stdout=logfile)    

class client():

    def __init__(self, driver):
        self.timeoutCounter = 0
        self.driver = driver

    def run_episodes(self, num_episodes):
        episode = 1
        while episode <= num_episodes:
            episode += 1
            self.run()
    
    def run(self):

        # Bind client to UDP-Socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.settimeout(2)
        self.s.bind(("", CLIENT_PORT))
        
        '''
        print("***********************************" )
        print("HOST: " , hostName    )
        print("PORT: ", serverPort  )
        print("ID: " , id     )
        print("MAX_STEPS: ", maxSteps  )
        print("MAX_EPISODES: ", maxEpisodes )
        print("TRACKNAME: ", trackName)
        
        if stage == BaseDriver.tstage.WARMUP:
            print("STAGE: WARMUP" )
        elif BaseDriver.tstage.QUALIFYING:
            print("STAGE: QUALIFYING" )
        elif BaseDriver.tstage.RACE:
            print("STAGE: RACE" )
        else:
            print("STAGE: UNKNOWN" )
        '''

        sp = SimpleParser.SimpleParser()
        self.driver.stage = stage
        
        msg_in = ""

        while True:
            # Initialize the angles of rangefinders
            angles = self.driver.getInitAngles()
            
            #self.driver.init(angles);
            initString = sp.stringify("init",angles)
            #print("Sending id to server: ", id)
            initString =  str(id) + initString
            #print("Sending init string to the server: ", initString)
            self.s.sendto((initString.encode()), (SERVER_IP, serverPort))
        
            # Read data from socket
            try:
                #wait to connect to server, without sleep program freezes
                time.sleep(1)
                msg_in, (client_ip, client_port) = self.s.recvfrom(BUFSIZE)

                ## GONZALO
                msg_in = msg_in.decode()

                #print (client_ip," ", client_port, " received:", msg_in[:-1])  # do not print last character "\00"
                
                #remove last character from string, seems to be a new line
                msg_in = msg_in[:-1]
                
                if msg_in== "***identified***":
                    break
            except:
                pass
                #print("no server running")   
                
        currentStep=0
        connection_lost = 0
        # Connected to server 
        while True:
            try:               
                msg_in, (client_ip, client_port) = self.s.recvfrom(BUFSIZE)

                ## GONZALO
                msg_in = msg_in.decode()

                #recTime = time.clock()
                recTime = datetime.datetime.now()
                # print("[",recTime,"]", client_ip," ", client_port," message received; length = ", len(msg_in))
                #remove last character from string, could be a new line character
                msg_in = msg_in[:-1]
            
            except:
                #print("error connection lost")
                connection_lost += 1
                if connection_lost >= 100:
                    break
                            
            if msg_in == "***shutdown***":
                start_server(False)
                time.sleep(1)
                self.driver.onShutdown()
                self.s.close()
                print("Shutdown!")
                break
    
            if msg_in == "***restart***":
                #start_server(False)
                #time.sleep(2)
                self.driver.onRestart()
                self.s.close()
                break
            
            #**************************************************
            #* Compute The Action to send to the solo race server
            #**************************************************
            currentStep = currentStep + 1
            if currentStep != maxSteps:
                action = self.driver.Update(msg_in)

                # write action to buffer
                # print("sending action", action)
                
                # create correct format
                msgBuffer = action
            else:
                # max actions reached
                msgBuffer = "(meta 1)"
            
            #send action    
            self.s.sendto(msgBuffer.encode(), (SERVER_IP, serverPort))

if __name__ == '__main__':
    szenario = 2

    # Train new steerer
    if szenario == 0:
        start_server(False)
        print("Server started!")

        steerer = QSteerer(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.005, epsilon_min=0.001)
        print("Steerer initalized!")

        accelerator = StaticAccelerator()
        print("Accelerator initalized!")

        driver = QDriver(steerer, accelerator)
        print("Driver initialized!")

        print("Start driving...")
        myclient = client(driver)
        myclient.run_episodes(250)
        driver.steerer.plot_stats()

        driver.steerer.epsilon = 0
        myclient.run_episodes(50)
        driver.steerer.plot_stats()

        start_server(True)
        myclient.run_episodes(10)
        
        accelerator = QAccelerator(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.005, epsilon_min=0.001)
        accelerator.print_frequency = 50

    # Test driver from qtable expport
    if szenario == 1:
        steerer = QSteerer(epsilon=0, alpha=0, gamma=0, epsilon_change=0, epsilon_min=0)
        steerer.stop_learning()
        steerer.import_qtable()

        accelerator = StaticAccelerator()

        driver = QDriver(steerer, accelerator)

        myclient = client(driver)
        myclient.run_episodes(20)

    # Train aaccelerator with steerer from export
    if szenario == 2:
        start_server(False)

        steerer = QSteerer(epsilon=0, alpha=0, gamma=0, epsilon_change=0, epsilon_min=0)
        steerer.stop_learning()
        steerer.import_qtable()

        accelerator = QAccelerator(epsilon=1, alpha=0.2, gamma=0.8, epsilon_change=-0.005, epsilon_min=0.001)

        driver = QDriver(steerer, accelerator)

        myclient = client(driver)
        myclient.run_episodes(250)
        driver.accelerator.plot_stats()

        driver.accelerator.epsilon = 0
        myclient.run_episodes(50)
        driver.accelerator.plot_stats()

        driver.accelerator.stop_learning()
        start_server(True)
        myclient.run_episodes(20)
