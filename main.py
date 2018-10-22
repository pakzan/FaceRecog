from flask import Flask, render_template, Response, jsonify
from multiprocessing import Process, Queue
import FaceRecog

app = Flask(__name__)

qFrame = Queue()
qStatus = Queue()
qPlayer = Queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Run FaceRecog multiprocessor
    process = Process(target=FaceRecog.main, args=(
        qFrame, qStatus, qPlayer))
    process.start()

    def gen():
        while True:
            yield qFrame.get()
    
    # Get frame or player info from FaceRecog module
    # getFrameOrInfo function will keep yielding jpeg frame until all player information is found
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    

@app.route('/audio_feed')
def audio_feed():
    # Server Send Event for sending current status to frontend
    # status will then be converted to speech afterwards
    def gen():
        while True:
            yield 'data: {}\n\n'.format(qStatus.get())
    return Response(gen(), mimetype='text/event-stream')


@app.route('/player_info')
def info():
    # FaceRecog.getFrameOrInfo() will return values after all players' faces found for 2 seconds
    # player_info = {'Player 1': [location(cm), angle(degree)], 'Player 2': ...}
    # format: DICTIONARY = {STRING: [float, float]}
    player_info = qPlayer.get()
    return jsonify(player_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

