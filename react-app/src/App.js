import "./App.css";
import React from "react";
import { db } from "./firebase";
import { useRecoileState, useSetRecoilState, useResetRecoilState } from "recoil";
import { usersAtom } from "./atom";
import { useState, useEffect } from "react";
import { async } from "@firebase/util";
import { collection, getDocs, addDoc } from "firebase/firestore";

function App() {
  const [newUID, setNewUID] = useState("");
  const [newPass, setNewPass] = useState(0);

  const [users, setUsers] = useState([]);
  const usersCollectionRef = collection(db, "users");

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
    <div className="App">
      <input
        placeholder="ID..."
        onChange={(e) => {
          setNewUID(e.target.value);
        }}
      />
      <button onClick={createUser}>FACE ID</button>
      <input
        placeholder="Password..."
        onChange={(e) => {
          // setNewPass(e.target.value);
        }}
      />
      <button onClick={verifyPW}>LOGIN</button>
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
  );
}

export default App;

// npm install firebase
// npm install uid
