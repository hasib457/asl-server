import socketio
from preprocess import Sign_predict, Sign_text

sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio, static_files={
    '/':'./public/'})
text_to_sign = Sign_text()
sign_predict = Sign_predict()

@sio.event
def connect(sid, environ):
    print(f'[INFO] client connected: {sid}')

@sio.event
def disconnect(sid):
    print(f'[INFO] client disconnected: {sid}') 

@sio.event
def stream_text(sid, data):
    """ 
        INPUT: sid, txt
        OUTPUT: emit signs frames for the recived text
    """
    for sign_frame in text_to_sign.sign_gen(data.get("data")):
        sio.emit("stream_text", {'data': sign_frame, "id":data.get("id")}, to=sid) 
    sio.emit("send","id": data.get("id"), to=sid)
   
@sio.event
def stream_sign(sid, data):
    pred = sign_predict.sign_predict(data.get("landmarks"))
    sio.emit("stream_sign", {"text": pred, "id": data.get("id")},to= sid)
