import "./CSS/styles.css";
import React from "react";
import styled from "styled-components";
import { useState, useEffect } from "react";
import bg from "./image/bg.jpg";

function App() {
  const [newUID, setNewUID] = useState("");
  const [users, setUsers] = useState([]);

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

  const isFaceID = (e) => {
    e.preventDefault();
    console.log(newUID);
    fetch(`http://localhost:5000/${newUID}`)
      .then((res) => res.json())
      .then((res) => {
        console.log(res);
        if (res.result === "1") {
          alert("인증되었습니다.");
        } else {
          alert("인증에 실패하였습니다.");
        }
      })
      .catch((err) => console.log(err));
  };

  const inLogin = (e) => {
    e.preventDefault();
  };

  return (
    <div className="container">
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
                  required
                  value={newUID}
                  onChange={(e) => {
                    setNewUID(e.target.value);
                    console.log(newUID);
                  }}
                />
                {/* 이 부분의 버튼을 누르면 FaceID인증page 이동 */}
                <button className="main-btn" onClick={isFaceID}>
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
                <button className="main-btn" onClick={inLogin}>
                  LOGIN
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

// npm install firebase
// npm install uid
