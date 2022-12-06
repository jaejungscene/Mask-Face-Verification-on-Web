import "./CSS/styles.css";
import React from "react";
import { db } from "./firebase";
import styled from "styled-components";
import { useRecoileState, useSetRecoilState, useResetRecoilState } from "recoil";
import { usersAtom } from "./atom";
import { useState, useEffect } from "react";
import { async } from "@firebase/util";
import { collection, getDocs, addDoc } from "firebase/firestore";
import bg from "./image/bg.jpg";

function App() {
  const [newUID, setNewUID] = useState("");
  const [newPass, setNewPass] = useState(0);

  const [users, setUsers] = useState([]);
  const usersCollectionRef = collection(db, "users");

  const Container = styled.div`
    display: flex;
    position: relative;
    width: 100%;
    height: calc(100vh);
    background: url(${bg});
    background-size: cover;
    background-origin: content-box;
    align-items: center;
    justify-content: center;
  `;

  const createUser = async () => {
    await addDoc(usersCollectionRef, { UID: newUID });
    console.log("Creat new ID :", newUID);
  };

  const verifyPW = async () => {
    // await addDoc(usersCollectionRef, { password: newPass });
    console.log("verifyPW");
    console.log("Userlen :", users.length);
    var temp = users[0].password;
    console.log("Password len", Object.keys(temp).length);
  };

  useEffect(() => {
    const getUsers = async () => {
      const usersCollection = await getDocs(usersCollectionRef);
      setUsers(usersCollection.docs.map((doc) => ({ ...doc.data(), id: doc.id })));
      console.log(usersCollection);
    };

    getUsers();
  }, []);

  return (
    <Container>
      <div className="content">
        <div className="box-s">
          <div className="bg-L"></div>
          <div className="bg-R">
            <div className="login-wrapper">
              <h1 className="title">아주대학교 통합인증</h1>
              <div>
                <input
                  className="inputbox"
                  type="text"
                  placeholder="사용자 ID를 입력해주세요."
                  // onChange={(e) => {
                  //   setNewUID(e.target.value);
                  // }}
                />
                {/* 이 부분의 버튼을 누르면 FaceID인증page 이동 */}
                <button className="main-btn" onClick={createUser}>
                  FACE ID
                </button>
              </div>
              <div>
                <input
                  className="inputbox"
                  type="password"
                  placeholder="비밀번호를 입력해주세요."
                  onChange={(e) => {
                    // setNewPass(e.target.value);
                  }}
                />
                {/* 위 FaceID를 통해 인증이 되면 자동적으로 password가 입력되며 Login btn을 클릭해 로그인 성공 */}
                <button className="main-btn" onClick={verifyPW}>
                  LOGIN
                </button>
                {users.map((user) => {
                  return (
                    <div>
                      <h1>UID : {user.UID}</h1>
                      {/* <h1>
              Password : {user.password["1"]}, {user.password["2"]}
            </h1> */}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Container>
  );
}

export default App;

// npm install firebase
// npm install uid
