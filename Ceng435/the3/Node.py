import json
import sys
import socket
import threading
import time


class Node:
    def __init__(self, port):
        self.node_count = 0
        self.neighbors = []
        self.port = port
        self.listener_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.distance_vector = {}
        self.read(port)

    def read(self, port):
        with open(f"{port}.costs") as f:
            i = 0
            for line in f:
                if i == 0:
                    self.node_count = int(line)
                    for j in range(self.node_count):
                        self.distance_vector[str(3000+j)] = 9999999
                else:
                    node, cost = line.split()
                    self.distance_vector[node] = int(cost)
                    self.distance_vector[str(self.port)] = 0
                    self.neighbors.append(node)
                i += 1

    def send(self):
        time.sleep(0.1)
        for neighbor in self.neighbors:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', int(neighbor)))
            sock.sendall(str.encode(json.dumps(self.distance_vector)))
            sock.close()

    def listen(self):
        #print(self.distance_vector)
        self.listener_sock.bind(('127.0.0.1', int(self.port)))
        self.listener_sock.listen()
        self.listener_sock.settimeout(5)
        while 1:
            try:
                conn, addr = self.listener_sock.accept()
                with conn:
                    data = conn.recv(2048)
                    if self.update_distances(json.loads(data.decode())):
                        self.listener_sock.settimeout(5)
            except socket.timeout:
                self.listener_sock.close()
                self.print()
                break

    def update_distances(self, data):
        ports = [*data]
        costs = [*data.values()]
        sent_from = ""
        for i in range(len(costs)):
            if costs[i] == 0:
                sent_from = ports[i]
                break

        changed = False
        for port, cost in data.items():
            if port == str(self.port):
                continue
            new_cost = self.distance_vector[sent_from] + cost
            if new_cost < self.distance_vector[port]:
                self.distance_vector[port] = new_cost
                changed = True
        if changed:
            self.send()
        return changed

    def print(self):
        for node, cost in self.distance_vector.items():
            print(str(self.port) + "-" + str(node) + "|" + str(cost))

    def run(self):
        listen_thread = threading.Thread(target=self.listen)
        listen_thread.start()
        self.send()
        listen_thread.join(timeout=5)


Node(sys.argv[1]).run()



