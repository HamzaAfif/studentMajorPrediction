from flask import Flask, jsonify, request
from flask_socketio import SocketIO
import eventlet
from brain import predict_student_major
from flask_cors import CORS

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/api/message', methods=['GET'])
def send_json_message():
    
    response = {
        "status": "success",
        "message": "Hello, this is your JSON response from the API!"
    }
    return jsonify(response), 200


@app.route('/api/receive', methods=['POST'])
def receive_data():
    data = request.json
    print(f"Received data from Node.js: {data}")
    
    socketio.emit('update', {'message': f"New data received: {data['text']}"})
    
    
    return jsonify({"status": "success", "received": data['text']}), 200

@socketio.on('connect')
def test_connect():
    print('Client connected')
    socketio.emit('update', {'message': 'Test message from server'})
   


@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')
    


@app.route('/api/predict', methods=['GET'])
def predict():
    
    student_data = [80, 75, 85, 60, 78, 1]  

    if not student_data:
        return jsonify({"error": "Missing student data"}), 400

    predicted_major = predict_student_major(student_data)

    return jsonify({"predicted_major": predicted_major}), 200


@app.route('/api/notes', methods=['POST'])
def receive_notes():
    
    data = request.json
    print(f"Received notes from front-end: {data}")
    
    data_values = list(data.values())[:-1]
    userID = list(data.values())[-1]
    
    print(userID)
    predicted_major = predict_student_major(data_values, userID)
    
    return jsonify({"predicted_major": predicted_major}), 200


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

