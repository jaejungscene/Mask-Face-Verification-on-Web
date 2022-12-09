from flask import Flask, render_template, jsonify, request
from register import face_vectorization
from identifier import face_verifier
import json

app = Flask(__name__)


@app.route('/register/<newID>', methods=['GET']) # param: UID
def register_face(newID):
    print("==== operating register.py ====")
    uid = newID
    print("user ID : ", newID)
    model = face_vectorization()
    model.capturing()
    emb = [1,]
    print("==== register fininshed ====")
    return jsonify({
        'result': '1',
    })


@app.route('/verify', methods=['GET'])  # param: UID, emb
def register_face():
    print("==== operating register.py ====")
    model = face_verifier()
    model.capturing()
    emb = [1,]
    print("==== register fininshed ====")
    return jsonify({
        'result': 1, # 성공시
        'emb' : emb,
    })


if __name__ == "__main__":
    app.run()