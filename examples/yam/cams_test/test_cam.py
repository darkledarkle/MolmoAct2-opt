import zmq, pickle, cv2, numpy as np

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.connect('tcp://127.0.0.1:5555')

sock.send(pickle.dumps({'cmd': 'obs'}))
resp = pickle.loads(sock.recv())

if resp['ok']:
    for name, frame in resp['frames'].items():
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = f'/home/pantheon/molmoact2/MolmoAct2-opt/examples/yam/cams_test/cams_testcam_{name}.jpg'
        cv2.imwrite(path, bgr)
        print(f'Saved {path} shape={frame.shape}')
else:
    print('Error:', resp['error'])