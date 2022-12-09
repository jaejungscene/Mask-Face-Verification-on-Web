from flask import Flask, render_template, jsonify, request
# from register import face_vectorization
# from identifier import face_verifier
import json

app = Flask(__name__)


@app.route('/register', methods=['GET'])
def register_face():
    print("==== operating register.py ====")
    # model = face_vectorization()
    # model.capturing()
    print("==== register fininshed ====")
    return jsonify({
        'result': "done"
    })


@app.route('/verify', methods=['GET'])
def register_faces():
    print("==== operating register.py ====")
    # model = face_verifier()
    # model.capturing()
    print("==== register fininshed ====")
    return jsonify({
        'result': "done"
    })


if __name__ == "__main__":
    app.run()