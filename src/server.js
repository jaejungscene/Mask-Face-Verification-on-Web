import express from "express";
import morgan from "morgan";
import { collection, getDocs, addDoc, setDoc, updateDoc, deleteDoc, doc } from "firebase/firestore";
import { db } from "./config";

// import globalRouter from "./routers/globalRouter.js";

const cors = require("cors");
const PORT = 4004 || process.env.PORT;
const app = express();
const logger = morgan("dev");
const usersCollectionRef = collection(db, "users");

app.set("view engine", "html");
// app.set("views", process.cwd() + "/src/views");
app.use(express.json());
app.use(cors());
app.use(logger);

app.get("/", async (req, res) => {
  const snapshot = await getDocs(usersCollectionRef);
  const ids = snapshot.docs.map((doc) => doc.id); // id traceback
  console.log(ids);
  const list = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() })); // data traceback
  res.send(list);
});

//Create new user POST msg
app.post("/create", async (req, res) => {
  const data = req.body;
  if (!req.body.UID) {
    return res.status(400).json({
      status: "error",
      error: "Please provide a valid data",
    });
  } else {
    console.log("Data of users:", data);
    await addDoc(usersCollectionRef, { UID: data.UID, password: data.password, faceid: data.faceid });
    res.send("Creat new ID :" + data.UID);

    return res.status(200).json({
      status: "success",
    });
  }
});

//UPDATE user POST msg
app.post("/update", async (req, res) => {
  const id = req.body.id;
  if (!req.body) {
    return res.status(400).json({
      status: "error",
      error: "Please provide a valid data",
    });
  }
  res.status(200).json({
    status: "success",
    id: id,
  });
  console.log("Before deleting ID", req.body);
  delete req.body.id;
  console.log("After deleting ID", req.body);
  const data = req.body;
  await updateDoc(doc(db, "users", id), data);
  res.send({ msg: "User Updated!" });
});

//DELETE user POST msg
app.post("/delete", async (req, res) => {
  const id = req.body.id;
  if (!req.body) {
    return res.status(400).json({
      status: "error",
      error: "Please provide a valid data",
    });
  }
  res.status(200).json({
    status: "success",
    id: id,
  });

  await deleteDoc(doc(db, "users", id));
  res.send({ msg: "User Deleted!" });
});

//회원 등록 POST : /signup
//새로운 UID와 password를 입력하면, firebase에 저장되고, 그 UID를 가진 사용자의 정보를 출력
//data = {UID: "inputuid"}
//중복된 UID 입력시, 에러 메시지 출력
app.post("/signup", async (req, res) => {
  const data = req.body;
  const snapshot = await getDocs(usersCollectionRef);
  const uidList = snapshot.docs.map((doc) => doc.data().UID); // UID traceback
  console.log(uidList);

  if (uidList.includes(data.UID)) {
    return res.status(400).json({
      status: "error",
      error: "중복된 UID입니다. 다른 UID를 입력해주세요.",
    });
  } else {
    const docRef = doc(db, "users", data.UID);
    await setDoc(docRef, { UID: data.UID, password: data.password });
    const list = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() })); // data traceback

    return res.status(200).json({
      status: "success",
      data: data,
      list: list,
    });
  }
});

//얼굴 등록 POST : /faceid
//UID(string)와 faceid(숫자) request를 보내면, firebase에 저장되고, 그 UID를 가진 사용자의 정보를 출력
//숫자가 입력되면 기존 faceid list에 추가로 저장됨.
app.post("/faceid", async (req, res) => {
  const data = req.body;
  const snapshot = await getDocs(usersCollectionRef);
  const uidList = snapshot.docs.map((doc) => doc.data().UID); // UID traceback
  var faceidList = [];
  snapshot.docs.map((doc) => {
    if (doc.data().UID == data.UID) {
      faceidList = doc.data().faceid;
    }
  }); // faceid traceback

  if (!uidList.includes(data.UID)) {
    return res.status(400).json({
      status: "error",
      error: "회원가입을 먼저 해주세요.",
    });
  } else {
    const docRef = doc(db, "users", data.UID);
    faceidList.push(data.faceid);
    console.log(faceidList);
    await updateDoc(docRef, { faceid: faceidList });
    console.log(data.UID, data.faceid);
    const list = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() })); // data traceback

    return res.status(200).json({
      status: "success",
      data: data,
      list: list,
    });
  }
});

//얼굴 인증 POST : /faceauth
//인증성공시 200, 실패시 400
//data = {UID: "inputUid", faceid: InputFaceidEmbbedingVector}
app.post("/faceauth", async (req, res) => {
  const data = req.body;
  const snapshot = await getDocs(usersCollectionRef);
  var faceidList = [];
  snapshot.docs.map((doc) => {
    if (doc.data().UID == data.UID) {
      faceidList = doc.data().faceid; // UID가 갖고있는 embbeding faceid list

      /*여기서 특정 threshold를 넘는지, 즉 faceid로 로그인이 성공하는지 boolean값 반환*/
      flag = true;
      if (flag) {
        return res.status(200).json({
          status: "success",
          data: data,
        });
      } else {
        return res.status(400).json({
          status: "error",
          error: "faceid가 일치하지 않습니다.",
        });
      }
    } else {
      return res.status(400).json({
        status: "error",
        error: "회원가입을 먼저 해주세요.",
      });
    }
  }); // faceid traceback;
});

//로그인 POST msg : /login
//UID와 password가 일치하면 200, 일치하지 않으면 400
app.post("/login", async (req, res) => {
  const data = req.body;
  const snapshot = await getDocs(usersCollectionRef);
  snapshot.docs.map((doc) => {
    if (doc.data().UID == data.UID) {
      if (doc.data().password == data.password) {
        return res.status(200).json({
          status: "success",
          data: data,
        });
      } else {
        return res.status(400).json({
          status: "error",
          error: "비밀번호가 일치하지 않습니다.",
        });
      }
    } else {
      return res.status(400).json({
        status: "error",
        error: "회원가입을 먼저 해주세요.",
      });
    }
  }); // faceid traceback;
});

const handleListening = () => console.log(`server listening on port http://localhost:${PORT} 💥`);

app.listen(PORT, handleListening);
