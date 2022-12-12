import "./CSS/styles.css";
import React from "react";
import styled from "styled-components";
import { useState, useEffect } from "react";
import bg from "./image/bg.jpg";
import Modal from '@mui/material/Modal';
import Box from '@mui/material/Box';

function App() {
  const [newUID, setNewUID] = useState("");
  const [newPW, setNewPW] = useState("");
  const [registerUID, setRegisterUID] = useState("");
  const [registerPW, setRegisterPW] = useState("");
  const [users, setUsers] = useState([]);
  const [openModal, setOpenModal] = useState(false);
  const [openModal2, setOpenModal2] = useState(false);
  

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

  const BoxStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 400,
    bgcolor: 'background.paper',
    border: '2px solid #000',
    boxShadow: 24,
    p: 4,
  }

  const isFaceID = (e) => {
    e.preventDefault();
    console.log(newUID);
    // faceauth로 post 요청 날림
    fetch(`http://localhost:4004/faceauth`, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Content-Type' : 'application/json',
      },
      body: JSON.stringify({
        'UID' : newUID,
      })
    })
      .then((res) => res.json())
      .then((res) => {
        if (res.status === "error") {
          alert(res.error);
        } else if (res.status === "success") {
          alert("로그인 성공");
        }
      })
      .catch((err) => console.log(err));
  };

  const inLogin = (e) => {
    e.preventDefault();
    // firebase에 ID-PW 매칭 후 로그인 진행
    fetch(`http://localhost:4004/login`, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Content-Type' : 'application/json',
      },
      body: JSON.stringify({
        'UID' : newUID,
        'password' : newPW,
      })
    }).then((res) => res.json())
      .then((res) => {
        if(res.status === "success") {
          alert("로그인 성공");
          setNewPW("");
          setNewUID("");
          window.open("https://mportal.ajou.ac.kr/main.do")
        } else if (res.status === "error") {
          alert(res.error);
          setNewPW("");
        }
      })
      .catch((err) => console.log(err));

  };
  const handleJoin = () => {
    // DB에 아이디&PW 등록
    fetch(`http://localhost:4004/signup`, {
      method: 'POST',
      headers: {
        'Content-Type' : 'application/json',
      },
      body: JSON.stringify({
        'UID': registerUID,
        'password' : registerPW,
      })
    }).then((res) => res.json())
      .then((res) => {
        if (res.status === 200 || res.status === "success") {
          alert("회원가입 완료");
        }
        else if (res.status === 400 || res.status === 'error' ) {
          alert(res.error)
        }
      })
      .catch((err) => console.log(err));
    setOpenModal(false);
    setRegisterUID("");
    setRegisterPW("");
  }

  const handleRegiFace = () => {
    fetch(`http://localhost:4004/faceid`, {
      method: 'POST',
      headers: {
        'Content-Type' : 'application/json',
      },
      body: JSON.stringify({
        'UID': registerUID,
        'password' : registerPW,
      })
    })
    .then((res) => res.json())  
    .then((res) => {
      if (res.status === "success") {
        alert("Face ID 등록 완료");
      }
      else if (res.status === 'error' ) {
        alert(res.error)
      }
    })
    .catch((err) => console.log(err));
    setOpenModal2(false);
    setRegisterUID("");
    setRegisterPW("");
  }

  const handleOpenModal = () => setOpenModal(true);
  const handleCloseModal = () => setOpenModal(false);
  const handleOpenModal2 = () => setOpenModal2(true);
  const handleCloseModal2 = () => setOpenModal2(false);

  const handleFind = () => {
    console.log("find ID/PW")
    alert("서비스 준비중입니다.")
  }

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
                    setNewPW(e.target.value);
                  }}
                />
                {/* 위 FaceID를 통해 인증이 되면 자동적으로 password가 입력되며 Login btn을 클릭해 로그인 성공 */}
                <button className="main-btn" onClick={inLogin}>
                  LOGIN
                </button>
                <ul className="join-opts">
                  <li>
                  <a className="join" href='javascript:void(0);' onClick={handleOpenModal}>회원가입</a>
                  </li>
                  <li>
                  <a className="register_faceID" href ='javascript:void(0);' onClick={handleOpenModal2}>FaceID 등록</a>
                  </li>
                  <li>
                  <a className="register_faceID" href ='javascript:void(0);' onClick={handleFind}>ID/PW 찾기</a>
                  </li>
                </ul>
                
                

              </div>
            </div>
          </div>
        </div>
        <Modal
          open={openModal}
          onClose={handleCloseModal}
          aria-labelledby="modal-modal-title"
          aria-describedby="modal-modal-description"
        >
          
          <Box sx = {BoxStyle}>
            <div className="data-form">
              <dl>
                <dt className="data-col">아이디 </dt>
                <dd>
                  <input
                  className="register-input"
                  type="text"
                  placeholder="사용자 ID를 입력해주세요."
                  required
                  value={registerUID}
                  onChange={(e) => {
                    setRegisterUID(e.target.value);
                    console.log(newUID);
                  }}
                  />
                </dd>
              </dl>
              <dl>
                <dt className="data-col">비밀번호 </dt>
                <dd>
                  <input
                  className="register-input"
                  type="password"
                  placeholder="비밀번호를 입력해주세요."
                  onChange={(e) => {
                    setRegisterPW(e.target.value);
                  }}
                  />
                </dd>
              </dl>
              <button className="submit" onClick={handleJoin}>Submit</button>
            </div>
          </Box>
        </Modal>
        <Modal
          open={openModal2}
          onClose={handleCloseModal2}
          aria-labelledby="modal-modal-title"
          aria-describedby="modal-modal-description"
        >
          <Box sx = {BoxStyle}>
            <div className="data-form">
              <dl>
                <dt className="data-col">아이디 </dt>
                <dd>
                  <input
                  className="register-input"
                  type="text"
                  placeholder="사용자 ID를 입력해주세요."
                  required
                  value={registerUID}
                  onChange={(e) => {
                    setRegisterUID(e.target.value);
                    console.log(registerUID);
                  }}
                  />
                </dd>
              </dl>
              <dl>
                <dt className="data-col">비밀번호 </dt>
                <dd>
                  <input
                  className="register-input"
                  type="password"
                  placeholder="비밀번호를 입력해주세요."
                  onChange={(e) => {
                    setRegisterPW(e.target.value);
                  }}
                  />
                </dd>
              </dl>
              <button className="submit" onClick={handleRegiFace}>Face ID 등록</button>
            </div>
          </Box>
        </Modal>
      </div>
    </div>
  );
}

export default App;

// npm install firebase
// npm install uid
