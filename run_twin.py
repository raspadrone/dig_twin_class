import time
import json
import uuid
import pika
import numpy as np
import datetime as dt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from config.config import load_config
from queue_routingkeys import TestBenchAUCAE2DofRMQServerRoutingKey

# --- 1. CONFIGURATION ---
m, c, k_nom = 10.0, 15.0, 200.0
dt_step = 0.1

config = load_config("startup.conf")
EXCHANGE_NAME = config["rabbitmq"]["exchange"]

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "rPoynUg-aLtP_o69CjB2_fYAPEIE6JTWyOhKXeYiuwmf1EqVKQRUrwxS8d_AaFZ5gZPzg7Yd9ZwjqZIgKZrbNw=="
INFLUX_ORG = "AU"
INFLUX_BUCKET = "testbench"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# --- 2. KALMAN FILTER ---
class KinematicKalmanFilter:
    def __init__(self, dt_step):
        self.X = np.zeros((2, 1))
        self.A = np.array([[1.0, dt_step], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.P = np.eye(2)
        self.Q = np.eye(2) * 1e-4
        self.R = np.array([[1e-2]])
    def update(self, z):
        X_pred = np.dot(self.A, self.X)
        P_pred = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, X_pred)
        self.X = X_pred + np.dot(K, y)
        self.P = np.dot((np.eye(2) - np.dot(K, self.H)), P_pred)
        return self.X[0, 0], self.X[1, 0]

kf = KinematicKalmanFilter(dt_step)

# --- 3. HARDENED PIKA RPC CLIENT ---
class DigitalTwinRPC:
    def __init__(self):
        rmq_cfg = config["rabbitmq"]
        credentials = pika.PlainCredentials(rmq_cfg["username"], rmq_cfg["password"])
        # Explicitly setting heartbeat and disabling blocked connection timeout
        parameters = pika.ConnectionParameters(
            host=rmq_cfg["ip"], 
            port=rmq_cfg["port"], 
            virtual_host=rmq_cfg["vhost"], 
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=None 
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        result = self.channel.queue_declare(queue='', exclusive=True, auto_delete=True)
        self.reply_queue = result.method.queue
        
        self.channel.basic_consume(
            queue=self.reply_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = json.loads(body)

    def call(self, method_name, args, timeout=1.5):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        
        self.channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=TestBenchAUCAE2DofRMQServerRoutingKey,
            properties=pika.BasicProperties(
                reply_to=self.reply_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps({"method": method_name, "args": args}).encode('utf-8')
        )
        
        # DEFENSIVE TIMEOUT: Never wait longer than 1.5 seconds
        start_time = time.time()
        while self.response is None:
            self.connection.process_data_events(time_limit=0.05)
            if time.time() - start_time > timeout:
                return None  # Break the infinite loop if Mockup drops the packet
                
        return self.response

# --- 4. THE DIGITAL TWIN CONTROL LOOP ---
print(" Booting Hardened Digital Twin Controller...")
dt_rpc = DigitalTwinRPC()
start_time = time.time()
print(" Injecting dynamic sine-wave force and streaming telemetry...")

try:
    while True:
        t = time.time() - start_time
        dynamic_force = 0.05 * np.sin(2.0 * np.pi * 0.5 * t)
        
        # 1. Inject Force
        dt_rpc.call("setForceTaskSpace", {"value": [dynamic_force, -dynamic_force]})
        
        # 2. Get Data
        pos_resp = dt_rpc.call("getPositionTaskSpace", {})
        force_resp = dt_rpc.call("getForceTaskSpace", {})
        
        # 3. SAFETY NET: If a packet was lost, skip this 0.1s frame and keep going
        if pos_resp is None or force_resp is None:
            print(f"Time: {t:05.1f}s |  Skipped frame (Packet lost)")
            continue
            
        raw_x = pos_resp[0] 
        raw_f = force_resp[0] 
        
        # 4. Process Data
        smooth_x, smooth_v = kf.update(raw_x)
        
        k_est = 200.0
        if abs(smooth_x) > 0.001:
            k_est = (raw_f - (c * smooth_v)) / smooth_x
            
        # 5. Save to Database
        point = Point("twin_telemetry") \
            .field("raw_position", float(raw_x)) \
            .field("filtered_position", float(smooth_x)) \
            .field("estimated_stiffness", float(k_est)) \
            .time(dt.datetime.now(dt.timezone.utc), WritePrecision.NS)
        
        write_api.write(INFLUX_BUCKET, INFLUX_ORG, point)
        print(f"Time: {t:05.1f}s | Force: {dynamic_force:>6.1f}N | x: {smooth_x:>7.4f}m | k_est: {k_est:.1f} N/m")
        
        # Keep RabbitMQ alive between loops
        dt_rpc.connection.sleep(dt_step)
        
except KeyboardInterrupt:
    print("\n Stopped Digital Twin.")
finally:
    dt_rpc.connection.close()