import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = ''  # ip of raspberry pi
port = 80
s.bind((host, port))

s.listen(5)

while True:

    c, addr = s.accept()
    print('Got connection from', addr)
    #c.send('Thank you for connecting')
    c.close()
