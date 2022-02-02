import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = ''  # ip of raspberry pi
port = 80
s.connect((host, port))
print(s.recv(1024))
s.close()
