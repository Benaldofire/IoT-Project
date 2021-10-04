import time
import network
from time import sleep
from machine import Pin
import dht
import urequests

import gc
gc.collect() #garbage collector

#Name and password of desired wifi
ssidRouter     = '' #Enter the router name
passwordRouter = '' #Enter the router password

#API key required for IFTTT so that we can send data to it using post request
api_key = ''
#Obtain ur from IFTT site after registration. e.g 'https://maker.ifttt.com/trigger/project/with/key/'+APIKEY
url_IFTT = ''

#method to connect to the wifi, when given the name and password of that wifi.
def STA_Setup(ssidRouter,passwordRouter):
    print("Setup start")
    sta_if = network.WLAN(network.STA_IF)
    if not sta_if.isconnected():
        print('connecting to',ssidRouter)
        sta_if.active(True)
        sta_if.connect(ssidRouter,passwordRouter)
        while not sta_if.isconnected():
            pass
    print('Connected, IP address:', sta_if.ifconfig())
    print("Setup End")


try:
    STA_Setup(ssidRouter,passwordRouter)
except:
    sta_if.disconnect()
    
DHT = dht.DHT11(Pin(18))
duration = 0; 
try:
    while duration <= 60: #one reading every 2 minutes for 2 hours, 120/2 = 60 iterations/readings
        DHT.measure()
        temp = DHT.temperature()
        humidity = DHT.humidity()
        print("Temperature: %2.2f C" %temp) #.2f to display 2 decimal places (f = precision)
        print("Humidity: %2.2f %%" %humidity)
        
        readings = {'value1':duration, 'value2': temp, 'value3': humidity}
        print(readings) #debugging purposes
        
        #the information required to make the post request to the IFTTT
        request_headers = {'Content-Type':'application/json'}
        request = urequests.post(
            url_IFTT+api_key,
            json=readings,
            headers=request_headers)
        print(request.text) #print request for debugging purposes
        request.close()
        duration = duration + 2 #increment duration/iterations required for the while loop
        sleep(120) #sleep for 120 seconds, then repeat
        
except OSError as e:
    print('Failure in publishing readings.')



    
