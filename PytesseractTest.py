import cv2 
import pytesseract
import paho.mqtt.client as mqtt
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def on_connect(client, userdata, flags, rc):
	print("Connection returned result: "+str(rc))

	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	# client.subscribe("ece180d/test")

# The callback of the client when it disconnects.

def on_disconnect(client, userdata, rc):
	if rc != 0:
		print('Unexpected Disconnect')
	else:
		print('Expected Disconnect')

# The default message callback.
# (won’t be used if only publishing, but can still exist)
def on_message(client, userdata, message):
	print('Received message: "' + str(message.payload) + '" on topic "' +
	message.topic + '" with QoS ' + str(message.qos))

# Camera section of the code 
# Once the camera is showing the in the terminal position the camera over the text that you want taken
# press 's' to take the picture
# Press 'q' to quit out 
# You can take an infinite amount of photos than can be taken
while True:
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(1)
    while True:
        try:
            check, frame = webcam.read()
            #print(check) #prints true as long as the webcam is running
            #print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                #print("Processing image...")
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                #print("Converting RGB image to grayscale...")
                gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                #print("Converted RGB image to grayscale...")
                #print("Resizing image to 28x28 scale...")
                img_ = cv2.resize(gray,(28,28))
                #print("Resized...")
                img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                #print("Image saved!")
            
                break
            elif key == ord('q'):
                #print("Turning off camera.")
                webcam.release()
                #print("Camera off.")
                #print("Program ended.")
                cv2.destroyAllWindows()
                exit()
                break
            
        except(KeyboardInterrupt):
            #print("Turning off camera.")
            webcam.release()
            #print("Camera off.")
            #print("Program ended.")
            cv2.destroyAllWindows()
            break

    # Take the saved image and send to ocr pytesseract function to process 
    img = cv2.imread('saved_img.jpg')
	
    # save the processed text in 'text' to send with mqtt
    text = pytesseract.image_to_string(img)
    print(text)

    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    pytesseract.image_to_string(img, config=custom_config)


    # 0. define callbacks - functions that run when events happen.
    # The callback for when the client receives a CONNACK response from the server.


    # 1. create a client instance.
    client = mqtt.Client()
    # add additional client options (security, certifications, etc.)
    # many default options should be good to start off.
    # add callbacks to client.
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # 2. connect to a broker using one of the connect*() functions.
    client.connect_async('mqtt.eclipseprojects.io')

    # 3. call one of the loop*() functions to maintain network traffic flow with the broker.
    client.loop_start()

    # 4. use subscribe() to subscribe to a topic and receive messages.

    # 5. use publish() to publish messages to the broker.
    # payload must be a string, bytearray, int, float or None.
    #text = 'My very photogenic mother died in a freak accident (picnic, lightning) when I was three, and, save for a pocket of warmth in the darkest past, nothing of her subsists within the hollows and dells of memory, over which, if you can still stand my style (I am writing under observation), the sun of my infancy had set: surely, you all know those redolent remnants of day suspended, with the midges, about some hedge in bloom or suddenly entered and traversed by the rambler, at the bottom of a hill, in the summer dusk; a furry warmth, golden midges.'

    client.publish('ece180d/text', text, qos=1)

    # 6. use disconnect() to disconnect from the broker.
    client.loop_stop()
    client.disconnect()
