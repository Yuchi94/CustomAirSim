from AirSimClient import *
from matplotlib import pyplot as plt
import numpy as np

# connect to the AirSim simulator 
client1 = CarClient(port = 42451)
client1.confirmConnection()
client1.enableApiControl(True)

client2 = CarClient(port = 42452)
client2.confirmConnection()
client2.enableApiControl(True)

client3 = CarClient(port = 42453)
client3.confirmConnection()
client3.enableApiControl(True)

client4 = CarClient(port = 42454)
client4.confirmConnection()
client4.enableApiControl(True)

car_controls = CarControls()
car_controls.throttle = 1

responses1 = client1.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
response1 = responses1[0]

img1d1 = np.fromstring(response1.image_data_uint8, dtype=np.uint8) 

img1 = img1d1.reshape(response1.height, response1.width, 4)  

img1 = np.flipud(img1)


responses2 = client2.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
response2 = responses2[0]

img1d2 = np.fromstring(response2.image_data_uint8, dtype=np.uint8) 

img2 = img1d2.reshape(response2.height, response2.width, 4)  

img2 = np.flipud(img2)


responses3 = client3.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
response3 = responses3[0]

img1d3 = np.fromstring(response3.image_data_uint8, dtype=np.uint8) 

img3 = img1d3.reshape(response3.height, response3.width, 4)  

img3 = np.flipud(img3)


responses4 = client4.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
response4 = responses4[0]

img1d4 = np.fromstring(response4.image_data_uint8, dtype=np.uint8) 

img4 = img1d4.reshape(response4.height, response4.width, 4)  

img4 = np.flipud(img4)


# for idx in range(3):
#     # get state of the car
#     car_state = client.getCarState()
#     print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

#     # go forward
#     car_controls.throttle = 0.5
#     car_controls.steering = 0
#     client.setCarControls(car_controls)
#     print("Go Foward")
#     time.sleep(3)   # let car drive a bit

#     # Go forward + steer right
#     car_controls.throttle = 0.5
#     car_controls.steering = 1
#     client.setCarControls(car_controls)
#     print("Go Foward, steer right")
#     time.sleep(3)   # let car drive a bit

#     # go reverse
#     car_controls.throttle = -0.5
#     car_controls.is_manual_gear = True;
#     car_controls.manual_gear = -1
#     car_controls.steering = 0
#     client.setCarControls(car_controls)
#     print("Go reverse, steer right")
#     time.sleep(3)   # let car drive a bit
#     car_controls.is_manual_gear = False; # change back gear to auto
#     car_controls.manual_gear = 0  

#     # apply breaks
#     car_controls.brake = 1
#     client.setCarControls(car_controls)
#     print("Apply break")
#     time.sleep(3)   # let car drive a bit
#     car_controls.brake = 0 #remove break
    
#     # get camera images from the car
#     responses = client.simGetImages([
#         ImageRequest(0, AirSimImageType.DepthVis),  #depth visualiztion image
#         ImageRequest(1, AirSimImageType.DepthPerspective, True), #depth in perspective projection
#         ImageRequest(1, AirSimImageType.Scene), #scene vision image in png format
#         ImageRequest(1, AirSimImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
#     print('Retrieved images: %d', len(responses))

#     for response in responses:
#         filename = 'c:/temp/py' + str(idx)

#         if response.pixels_as_float:
#             print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#             AirSimClientBase.write_pfm(os.path.normpath(filename + '.pfm'), AirSimClientBase.getPfmArray(response))
#         elif response.compress: #png format
#             print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#             AirSimClientBase.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#         else: #uncompressed array
#             print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#             img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
#             img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
#             img_rgba = np.flipud(img_rgba) #original image is fliped vertically
#             img_rgba[:,:,1:2] = 100 #just for fun add little bit of green in all pixels
#             AirSimClientBase.write_png(os.path.normpath(filename + '.greener.png'), img_rgba) #write to png 


# #restore to original state
# client.reset()

# client.enableApiControl(False)


            
