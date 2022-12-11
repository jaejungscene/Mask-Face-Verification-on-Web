from flask import Flask, render_template, jsonify, request
from register import face_vectorization
from identifier import face_verifier
import json
from flask_cors import CORS, cross_origin
import os

base_dir = "C:/Users/gmk_0/source/repos/pythonProject/IT2/Computer-Vision-Project/"

def is_registered(userId):
    npy_path = os.path.join(base_dir, 'images/')
    # 유저가 얼굴을 등록했으면 true, 등록하지 않았으면 false을 리턴
    npys = os.listdir(npy_path)
    if userId+'.npy' in npys:
        return True
    else:
        return False

app = Flask(__name__)
cors = CORS(app, resources={
    r"*": { "origin": "*"},
})
app.config['CORS_HEADER']='Content-Type'

@app.route('/register/<newID>', methods=['GET']) # param: UID
@cross_origin()
def register_face(newID):
    print('user ID : ', str(newID))
    print("==== operating register.py ====")
    model = face_vectorization()
    emb = model.capturing(newID)
    print("User "+newID+"'s ID : \n", emb)
    print("==== register fininshed ====")
    return jsonify({
        'result': '1'
    })


@app.route('/verify/<userID>', methods=['GET'])  # param: UID, emb
@cross_origin()
def verify_face(userID):
    if not is_registered(userID):
        print('얼굴 미등록 아이디')
        emb = 2
    else:
        print("==== operating identifier.py ====")
        model = face_verifier()
        emb = model.capturing(userID)
        if emb == 1:
            print("match success")
        else:
            print("match failed")
        print("==== verification fininshed ====")
    return jsonify({
        'result': str(emb), # 성공시 1, 실패시 0
    })


if __name__ == "__main__":
    app.run()