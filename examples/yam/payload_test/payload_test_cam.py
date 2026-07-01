"""JPEG inference latency benchmark."""
import requests, time, json, numpy as np, cv2, json_numpy, zmq, pickle, socket, threading, base64
from requests.adapters import HTTPAdapter
import urllib3.connectionpool as cp
from urllib3.connection import HTTPConnection
json_numpy.patch()

URL = "http://195.26.233.28:50726/act"
NUM_STEPS = 10
JPEG_QUALITY = 95
N = 50

# TCP_NODELAY + keepalive
class NoDelayHTTPConnection(HTTPConnection):
    def connect(self):
        super().connect()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

cp.HTTPConnection = NoDelayHTTPConnection

session = requests.Session()
session.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=1))

# keep-warm thread
stop_warm = threading.Event()
def _keep_warm(interval=2.0):
    while not stop_warm.is_set():
        try:
            session.get(URL.replace("/act", "/healthz"), timeout=1)
        except Exception:
            pass
        time.sleep(interval)
threading.Thread(target=_keep_warm, daemon=True).start()

# camera server
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 500)
sock.connect("tcp://127.0.0.1:5555")

def get_frames():
    sock.send(pickle.dumps({"cmd": "obs"}))
    resp = pickle.loads(sock.recv())
    if not resp.get("ok"):
        raise RuntimeError(f"Camera error: {resp.get('error')}")
    f = resp["frames"]
    return f["left_camera"], f["front_camera"], f["right_camera"]

def b64_jpeg(rgb: np.ndarray, quality: int = 95) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()

def call():
    left, front, right = get_frames()
    t_encode = time.perf_counter()
    payload = json_numpy.dumps({
        "images": [b64_jpeg(front, JPEG_QUALITY), b64_jpeg(left, JPEG_QUALITY), b64_jpeg(right, JPEG_QUALITY)],
        "instruction": "pick up the object",
        "state": [0.0]*14,
        "num_steps": NUM_STEPS,
    })
    encode_ms = (time.perf_counter() - t_encode) * 1000
    t0 = time.perf_counter()
    r = session.post(URL, data=payload, timeout=30)
    wall = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        print("SERVER ERROR:", r.status_code, r.text)
        raise RuntimeError("bad request")
    server = json_numpy.loads(r.text).get("dt_ms", -1)
    return wall, server, wall - server, encode_ms

# warmup
left, front, right = get_frames()
_, buf = cv2.imencode('.jpg', left, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
print(f"Frame: {left.shape} | JPEG@{JPEG_QUALITY}: {len(buf)/1024:.1f}KB | NFE={NUM_STEPS}")
print("Warming up...")
for _ in range(5): call()

walls, servers, rtts = [], [], []
for i in range(N):
    w, s, rtt, enc_ms = call()
    walls.append(w); servers.append(s); rtts.append(rtt)
    print(f"  {i+1:2d}/{N} wall={w:.0f}ms server={s:.0f}ms RTT={rtt:.0f}ms encode={enc_ms:.1f}ms")

walls  = np.array(walls)
rtts   = np.array(rtts)
servers = np.array(servers)
print(f"\n  wall   mean={walls.mean():.0f}ms std={walls.std():.0f}ms min={walls.min():.0f}ms max={walls.max():.0f}ms")
print(f"  RTT    mean={rtts.mean():.0f}ms  std={rtts.std():.0f}ms  min={rtts.min():.0f}ms  max={rtts.max():.0f}ms")
print(f"  server mean={servers.mean():.0f}ms std={servers.std():.0f}ms min={servers.min():.0f}ms max={servers.max():.0f}ms")

stop_warm.set()