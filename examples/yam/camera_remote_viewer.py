import zmq, pickle, cv2

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5556")
sock.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    msg = sock.recv()
    data = pickle.loads(msg)
    img = data["frames"]["front_camera"]
    cv2.imshow("front_camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()